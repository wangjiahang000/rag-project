"""重排序工具

支持 BM25 和 Cross-Encoder 两种模式，由全局配置控制。
"""

import logging
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Optional

import content_core.config as cfg

logger = logging.getLogger(__name__)

# 模块级 Cross-Encoder 实例（延迟初始化）
_ce_reranker = None


def _get_ce_reranker():
    """延迟加载 Cross-Encoder"""
    global _ce_reranker
    if cfg.RERANK_MODE != "cross_encoder":
        return None
    if _ce_reranker is not None:
        return _ce_reranker
    try:
        from content_core.reranking.reranker import CrossEncoderReranker
        _ce_reranker = CrossEncoderReranker(
            model_path=cfg.CROSS_ENCODER_MODEL,
            device=cfg.CROSS_ENCODER_DEVICE,
        )
        return _ce_reranker
    except Exception as e:
        logger.error("Cross-Encoder 初始化失败: %s", e)
        return None


def rerank(docs: List[str], query: str) -> List[str]:
    """对文档列表重排序

    当 RERANK_MODE 为 "cross_encoder" 且模型可用时使用 Cross-Encoder，
    否则回退到 BM25。
    """
    if not docs:
        return []

    # 尝试 Cross-Encoder
    ce = _get_ce_reranker()
    if ce is not None and ce.available:
        logger.debug("使用 Cross-Encoder 重排序（%d 条文档）", len(docs))
        return ce.rerank(query, docs)

    # 回退 BM25
    tokenized = [list(jieba.cut(doc)) for doc in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(list(jieba.cut(query)))
    sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_pairs]
