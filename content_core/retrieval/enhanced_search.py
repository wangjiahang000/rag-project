"""增强检索引擎：实体软加权 + 时间衰减 + 查询改写 + MMR 多路召回

在已有 hybrid_search 结果基础上叠加动态因子：

1. 实体软加权：文档文本中包含查询实体的，分数提升 (×1.2)
2. 时间衰减：距查询年份越远的文档，分数按指数衰减
3. 查询改写：LLM 将原问题改写为多个搜索变体，每路独立检索后合并
4. MMR 去重：兼顾相关性与多样性，消除冗余结果
"""

import math
import json
import os
import logging
from typing import List, Dict, Optional, Tuple

from content_core.data.vector_store import VectorStore
from content_core.tools.process.entity_extractor import (
    extract_entities,
    extract_time_info,
    format_time_query,
)
import content_core.config as cfg

logger = logging.getLogger(__name__)


def entity_boost(
    results: List[Tuple[str, float, dict]],
    entities: List[str],
    boost: float = cfg.ENTITY_BOOST_FACTOR,
) -> List[Tuple[str, float, dict]]:
    """实体软加权：文档文本中出现查询实体的，分数提升

    Args:
        results: [(文档文本, 分数, 元数据), ...]
        entities: 实体列表
        boost: 提升倍数

    Returns:
        调整后的 [(文档文本, 分数, 元数据), ...]
    """
    if not entities or not results:
        return results

    boosted = []
    for item in results:
        doc, score = item[0], item[1]
        meta = item[2] if len(item) > 2 else {}
        matched = sum(1 for e in entities if e in doc)
        if matched > 0:
            factor = 1.0 + (boost - 1.0) * matched / len(entities)
            boosted.append((doc, score * factor, meta))
        else:
            boosted.append(item)

    # 重新排序
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def time_decay(
    results: List[Tuple[str, float, dict]],
    query_year: int,
    decay: float = cfg.TIME_DECAY_LAMBDA,
    max_range: int = cfg.MAX_YEAR_RANGE,
) -> List[Tuple[str, float, dict]]:
    """时间衰减：距查询年份越远分数越低

    score *= exp(-decay * |query_year - doc_year|)

    若无法获取 doc_year，则分数不变。

    Args:
        results: [(文档文本, 分数, 元数据), ...]
        query_year: 查询年份
        decay: 衰减系数
        max_range: 超过此年差则分数置 0（硬截断）

    Returns:
        调整后的 [(文档文本, 分数, 元数据), ...]
    """
    if not results:
        return results

    decayed = []
    for item in results:
        doc, score = item[0], item[1]
        meta = item[2] if len(item) > 2 else {}
        doc_year = _extract_year(doc)
        if doc_year is None:
            decayed.append(item)
            continue

        years_apart = abs(query_year - doc_year)
        if years_apart > max_range:
            continue  # 硬截断：不加入结果

        factor = math.exp(-decay * years_apart)
        decayed.append((doc, score * factor, meta))

    decayed.sort(key=lambda x: x[1], reverse=True)
    return decayed


def _extract_year(text: str) -> Optional[int]:
    """从文档文本中尝试提取第一出现的 4 位数字年份"""
    import re
    match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if match:
        return int(match.group(1))
    return None


def enhanced_hybrid_search(
    query: str,
    vector_store: VectorStore,
    k: int = 10,
    entities: Optional[List[str]] = None,
    time_info: Optional[List[dict]] = None,
    query_year: Optional[int] = None,
    use_rewrite: bool = True,
    use_mmr: bool = True,
) -> Dict:
    """增强版混合检索：改写 → 多路检索 → 合并 → 实体提权 → 时间衰减 → MMR

    这是意图调度层下发给检索层的统一入口。

    Args:
        query: 用户查询
        vector_store: 向量存储实例
        k: 返回结果数
        entities: 实体列表（用于软加权）
        time_info: 时间表达式列表（结构化）
        query_year: 查询年份（未指定则从 time_info 推断或默认）
        use_rewrite: 是否启用查询改写
        use_mmr: 是否启用 MMR 去重

    Returns:
        {
            "results": [(文档文本, 分数, 元数据), ...],
            "entities_used": [...],
            "time_info_used": {...},
            "queries_used": [...],
        }
    """
    # 1. 查询改写：生成多个检索变体
    if use_rewrite:
        queries = rewrite_query(query)
    else:
        queries = [query]

    # 2. 多路检索 + 合并（去重保留最高分）
    seen_docs = {}
    for q in queries:
        raw = vector_store.hybrid_search(q, k=k * 2)
        for item in raw:
            doc = item[0]
            if doc not in seen_docs or item[1] > seen_docs[doc][1]:
                seen_docs[doc] = item

    merged = sorted(seen_docs.values(), key=lambda x: x[1], reverse=True)
    if not merged:
        return {"results": [], "entities_used": entities or [], "time_info_used": time_info or [], "queries_used": queries}

    # 3. 实体软加权
    if entities:
        merged = entity_boost(merged, entities)

    # 4. 时间衰减
    if query_year is not None:
        merged = time_decay(merged, query_year)
    elif time_info:
        inferred = _infer_query_year(time_info)
        if inferred:
            merged = time_decay(merged, inferred)

    # 5. MMR 多样性重排序
    mmr_k = k * 3  # 候选池大一些，MMR 从中选 top-k
    if use_mmr and len(merged) > k:
        merged = mmr_rerank(merged, query, top_k=k)
    else:
        merged = merged[:k]

    return {
        "results": merged[:k],
        "entities_used": entities or [],
        "time_info_used": time_info or [],
        "queries_used": queries,
    }


def _infer_query_year(time_info: List[dict]) -> Optional[int]:
    """从结构化时间表达式推断查询年份"""
    for e in time_info:
        rel = e.get("time_relation")
        if rel == "year_range":
            return e.get("end_year") or e.get("start_year")
        elif rel == "after_year":
            return e.get("year")
        elif rel == "single_year":
            return e.get("year")
        elif rel == "recent_years":
            return cfg.DEFAULT_QUERY_YEAR
    return cfg.DEFAULT_QUERY_YEAR


# ── 查询改写 ──────────────────────────────

def rewrite_query(query: str, n_variants: int = 3) -> List[str]:
    """LLM 将用户问题改写为多个搜索变体，提升召回覆盖率

    改写策略：
    - 保持原始 query 作为第一个变体
    - 其余变体从不同角度改写：关键词扩展、同义替换、子问题拆分
    - 改写后的 query 并发检索，结果合并排序

    Returns:
        [原始query, 变体1, 变体2, ...]
    """
    if not cfg.QUERY_REWRITE_ENABLED:
        return [query]

    prompt = f"""你是一个搜索优化专家。请将用户的问题改写为{n_variants}个不同的搜索查询，用于学术论文检索。

要求：
- 保持原始含义，但用不同的关键词和表达方式
- 适当扩展技术术语和同义词
- 每个变体应覆盖原始问题的不同侧面
- 不要改变原始问题的信息内容
- 直接输出查询，不要编号和解释

原始问题: {query}

请输出{n_variants}个搜索查询，每行一个："""

    try:
        from litellm import completion
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL"),
            timeout=15,
        )
        content = response.choices[0].message.content
        if not content:
            return [query]

        variants = [line.strip().strip("\"'") for line in content.strip().split("\n") if line.strip()]
        # 过滤掉明显不是查询的行（如编号、说明文字）
        variants = [v for v in variants if len(v) >= 4 and not v.startswith(("1.", "2.", "3.", "(", "-"))]

        # 去重，与原始 query 合并
        seen = set()
        result = [query]
        for v in variants:
            if v.lower() not in seen and v != query:
                seen.add(v.lower())
                result.append(v)

        logger.info("查询改写: '%s' → %s", query[:50], result)
        return result[:n_variants]
    except Exception as e:
        logger.warning("查询改写失败，使用原始 query: %s", e)
        return [query]


# ── MMR 多路召回 ──────────────────────────

def mmr_rerank(
    results: List[Tuple[str, float, dict]],
    query: str,
    lambda_param: float = None,
    top_k: int = 10,
) -> List[Tuple[str, float, dict]]:
    """最大边际相关性（MMR）重排序：兼顾相关性与多样性

    MMR = λ * rel_score - (1-λ) * max_sim_to_selected

    λ=1 → 纯相关性排序，λ=0 → 纯多样性排序，λ=0.5 → 等权重

    使用 TF-IDF 向量计算文档间相似度。
    """
    if not cfg.MMR_ENABLED or len(results) <= top_k:
        return results[:top_k]

    lambda_param = lambda_param if lambda_param is not None else cfg.MMR_LAMBDA

    # 构建 TF-IDF 向量
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        docs = [r[0] for r in results]
        vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer="char_wb",
            ngram_range=(1, 2),
            stop_words=None,
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        doc_norms = np.asarray((tfidf_matrix @ tfidf_matrix.T).toarray())

        # MMR 贪心选择
        selected = []
        remaining = list(range(len(results)))

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_score = -float("inf")

            for i in remaining:
                rel_score = results[i][1]

                if selected:
                    sim_scores = [doc_norms[i][j] for j in selected]
                    max_sim = max(sim_scores)
                else:
                    max_sim = 0

                mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [results[i] for i in selected]
    except ImportError:
        logger.warning("scikit-learn 未安装，跳过 MMR 去重")
        return results[:top_k]


def build_enhanced_context(
    query: str,
    vector_store: VectorStore,
    tasks: List[str],
    k: int = 10,
) -> Dict:
    """构建带结构化引用的上下文（供 chat 路由调用）

    根据意图调整检索参数，集成查询改写和 MMR 多样性去重。
    返回格式化上下文和引用列表。

    Returns:
        {
            "context": 格式化的上下文字符串,
            "citations": [CitationInfo, ...],
        }
    """
    # 按意图调整 k 值
    intent_k = {
        "summarize": max(k, 20),   # 总结需要更多上下文
        "compare": max(k, 15),     # 对比也需要较多
    }
    effective_k = max(intent_k.get(t, k) for t in tasks) if tasks else k

    # 提取实体和时间信息
    entities = extract_entities(query)
    time_info = extract_time_info(query)

    # 是否为复杂查询（多意图或含对比/推理）→ 启用改写 + MMR
    is_complex = len(tasks) > 1 or "compare" in tasks or "reason" in tasks

    # 增强检索（集成改写 + MMR）
    enhanced = enhanced_hybrid_search(
        query=query,
        vector_store=vector_store,
        k=effective_k,
        entities=entities,
        time_info=time_info,
        use_rewrite=is_complex,
        use_mmr=is_complex,
    )

    results = enhanced["results"]
    if not results:
        return {"context": "暂无相关文献", "citations": []}

    # 相关性分数阈值过滤：最高分低于阈值视为无相关结果
    top_score = results[0][1]
    if top_score < cfg.MIN_RELEVANCE_SCORE:
        return {"context": "暂无相关文献", "citations": []}

    # 构建结构化上下文
    context_parts = []
    citations = []
    for i, item in enumerate(results, 1):
        doc, score = item[0], item[1]
        meta = item[2] if len(item) > 2 else {}
        # 优先用元数据中的 source（arxiv ID），回退到文本首行
        source = meta.get("source", "") or doc.strip().split("\n")[0][:80]
        chapter = meta.get("chapter", "")
        source_display = source
        if chapter:
            source_display = f"{source} (§{chapter})"
        year = meta.get("year") or _extract_year(doc)
        citations.append({
            "index": i,
            "source": source_display,
            "year": year,
        })
        context_parts.append(f"[{i}] (相关度: {score:.3f})\n  {doc}")

    context = "\n\n".join(context_parts)
    return {"context": context, "citations": citations}
