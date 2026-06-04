"""结构化上下文构建器

将检索结果转换为带引用编号的结构化上下文，供 LLM 生成使用。

设计：
- 每个文档块附带 [来源/年份] 标记
- LLM 生成时需要在答案中标注引用编号 [1][2][3]
- 最终输出带 Citations 列表供前端展示
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def build_context(
    results: List[tuple],
    query: str = "",
    include_metadata: bool = True,
    max_docs: int = 15,
) -> Dict:
    """将检索结果构建为结构化上下文

    Args:
        results: [(文档文本, 分数), ...] 或 [(文档文本, 分数, 元数据字典), ...]
        query: 原始查询（仅用于日志）
        include_metadata: 是否在上下文中嵌入元数据
        max_docs: 最大文档数

    Returns:
        {
            "context": 格式化上下文字符串,
            "citations": [{"index": 1, "source": "...", "year": ...}, ...],
        }
    """
    if not results:
        return {"context": "暂无相关文献", "citations": []}

    results = results[:max_docs]
    context_parts = []
    citations = []

    for i, item in enumerate(results, 1):
        if len(item) == 3:
            doc, score, meta = item
        else:
            doc, score = item
            meta = {}

        source = meta.get("source", meta.get("arxiv_id", ""))
        year = meta.get("year", meta.get("pub_year"))
        chapter = meta.get("chapter", "")

        # 构建元数据前缀
        meta_tags = []
        if source:
            meta_tags.append(f"来源: {source}")
        if year:
            meta_tags.append(str(year))
        if chapter:
            meta_tags.append(f"第{chapter}章")

        meta_str = f" ({' | '.join(meta_tags)})" if meta_tags else ""
        score_str = f"(相关度: {score:.3f})"
        header = f"[{i}]{meta_str} {score_str}"

        context_parts.append(f"{header}\n  {doc}")

        citations.append({
            "index": i,
            "source": source or doc[:60],
            "year": year,
        })

    context = "\n\n".join(context_parts)
    return {"context": context, "citations": citations}


def make_citation_prompt(citations: List[dict]) -> str:
    """生成引用说明 prompt 片段（注入 system message 用）"""
    if not citations:
        return ""
    return (
        "在回答中请用 [1][2] 等编号标注信息来源。"
        "每个引用编号对应上方文献列表中对应的编号。"
    )


SYSTEM_PROMPT_TPL = """你是一个学术助手。基于以下文献回答问题。
若文献不足以回答，请如实说明。

文献：
{context}

{cite_instruction}
请用中文回答。"""
