import os
import re
import time
import logging
from typing import List
from litellm import completion
import content_core.config as cfg

logger = logging.getLogger(__name__)

# ── BERT NER 实体抽取（延迟加载） ──

_ner_pipeline = None
_HF_AVAILABLE = None


def _hf_reachable() -> bool:
    """快速检测 HuggingFace 是否可达，避免 pipeline 加载长时间阻塞"""
    global _HF_AVAILABLE
    if _HF_AVAILABLE is not None:
        return _HF_AVAILABLE
    try:
        import urllib.request
        import socket
        socket.setdefaulttimeout(3)
        mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        urllib.request.urlopen(f"{mirror}/dslim/bert-base-NER/resolve/main/config.json", timeout=3)
        _HF_AVAILABLE = True
    except Exception:
        logger.info("HuggingFace 不可达，跳过 BERT NER 加载")
        _HF_AVAILABLE = False
    finally:
        socket.setdefaulttimeout(None)
    return _HF_AVAILABLE


def _get_ner():
    global _ner_pipeline
    if _ner_pipeline is None:
        if not _hf_reachable():
            _ner_pipeline = False
            return None
        try:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from transformers import pipeline
            _ner_pipeline = pipeline(
                "ner", model="dslim/bert-base-NER",
                aggregation_strategy="simple",
            )
        except Exception as e:
            logger.warning("BERT NER 加载失败: %s，使用规则兜底", e)
            _ner_pipeline = False
    return _ner_pipeline if _ner_pipeline is not False else None


def _split_fallback(query: str) -> List[str]:
    """规则兜底：按对比标记切分提取候选实体

    例: "LoRA和QLoRA有什么区别" → ["LoRA", "QLoRA"]
         "对比一下RAG和微调"   → ["RAG", "微调"]
         "Transformer和Bert比起来哪个好"
           → 去噪: "Transformer和Bert" → 切分: ["Transformer", "Bert"]
    """
    # 先移除对比意图本身的标记词，避免它们进入候选实体
    # 涵盖书面语 + 口语表达
    cleaned = re.sub(
        r'(?:(你|我)觉得|你认为|推荐|各自|分别|'
        r'你看|我们来看|我们来|'
        r'对比|比较|区别|差异|不同|有什么区别|有什么不同|有什么差异|'
        r'哪个好|谁优谁劣|谁更强|怎么样|如何|怎么|什么|哪些|'
        r'一下|分析|优缺点|特点|优劣|'
        r'有啥|比起来|比一下|说起来|聊一聊|说说|谈谈|'
        r'哪个更靠谱|哪个更实用|哪个更好用|哪个更合适|'
        r'哪个更|哪个比较|怎么选|怎么挑|选哪个|选谁|谁更好|谁厉害|谁的|'
        r'哪个最|更推荐|更实用|更好用|哪个好用|更靠谱|更合适|更优秀|更有效|'
        r'更简单|更复杂|更流行|更常用)',
        '', query
    )
    # 切分标记覆盖口语词
    parts = re.split(
        r'\s*(?:vs|VS|v\.s|v\.s\.|pk|PK|和|与|跟|还是|或者|'
        r'、|相比|比起|较之)\s*',
        cleaned
    )
    seen = set()
    entities = []
    for p in parts:
        p = p.strip().strip("?？，,。.!！、：:;；的了吧吗你我用来")
        if p and len(p) >= 1 and p not in seen and not re.match(r'^[\s的了吗呢是吧你我好这个]$', p):
            seen.add(p)
            entities.append(p)
    return entities


def extract_entities(query: str) -> List[str]:
    """使用 BERT NER 从查询中提取实体，失败时使用规则兜底"""
    if not query:
        return []
    ner = _get_ner()
    if ner is None:
        return _split_fallback(query)
    try:
        results = ner(query)
        entities = list(set(r["word"] for r in results if r.get("word")))
        if not entities:
            entities = _split_fallback(query)
        return entities
    except Exception as e:
        logger.warning("BERT NER 抽取异常: %s，使用规则兜底", e)
        return _split_fallback(query)


def compare(docs_a: list, docs_b: list, query: str = "") -> str:
    """对比两组文档，从核心观点、方法、结论三方面分析异同"""
    if not docs_a or not docs_b:
        return "对比数据不足"

    # 提取实体作为对比参考
    entities = extract_entities(query) if query else []

    entity_hint = ""
    if entities:
        entity_hint = f"\n\n请特别关注以下实体的对比分析：{' / '.join(entities[:4])}"

    prompt = f"""对比以下两组文档：

A组：
{chr(10).join(docs_a)}

B组：
{chr(10).join(docs_b)}

从核心观点、方法、结论三方面对比异同。{entity_hint}"""

    for attempt in range(1 + cfg.LLM_RETRY_COUNT):
        try:
            response = completion(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                api_base=os.getenv("DEEPSEEK_BASE_URL"),
                timeout=cfg.LLM_TIMEOUT,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("compare LLM 第 %d/%d 次失败: %s",
                           attempt + 1, cfg.LLM_RETRY_COUNT + 1, e)
            if attempt < cfg.LLM_RETRY_COUNT:
                time.sleep(cfg.LLM_RETRY_DELAY * (attempt + 1))
    logger.error("compare LLM 全部 %d 次尝试均失败", cfg.LLM_RETRY_COUNT + 1)
    return "[对比生成失败]"
