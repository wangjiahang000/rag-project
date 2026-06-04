"""Cross-Encoder 重排序器

提供 Cross-Encoder 和 BM25 两种重排序实现。

Cross-Encoder 直接对 (query, doc) 对打分，精度高于双塔/BM25，
但速度较慢。适用于候选数不多的精排阶段（50条以内）。

模型下载（用户手动执行）：
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
    model.save_pretrained("./models/bge-reranker-v2-m3")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    tokenizer.save_pretrained("./models/bge-reranker-v2-m3")
"""

import logging
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """基于 Cross-Encoder 的重排序器

    使用方式：
        reranker = CrossEncoderReranker(model_path="BAAI/bge-reranker-v2-m3")
        reranked = reranker.rerank(query, docs)

    若模型未下载或加载失败，自动降级到 BM25。
    """

    def __init__(self, model_path: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载 Cross-Encoder 模型，失败时记录警告"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=self.device,
            )
            self.model.eval()
            logger.info("Cross-Encoder 模型加载成功: %s", self.model_path)
        except Exception as e:
            logger.warning("Cross-Encoder 加载失败，降级到 BM25: %s", e)
            self.model = None
            self.tokenizer = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        """Cross-Encoder 重排序

        Args:
            query: 查询字符串
            docs: 文档列表

        Returns:
            按相关度降序排列的文档列表
        """
        if not docs:
            return []

        if not self.available:
            logger.warning("Cross-Encoder 不可用，返回原始顺序")
            return docs

        try:
            import torch

            pairs = [[query, doc] for doc in docs]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).tolist()

            if isinstance(scores, float):
                scores = [scores]

            sorted_pairs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in sorted_pairs]

        except Exception as e:
            logger.error("Cross-Encoder 推理失败: %s", e)
            return docs

    def rerank_with_scores(self, query: str, docs: List[str]) -> List[tuple]:
        """重排序并返回分数

        Returns:
            [(分数, 文档), ...] 按分数降序
        """
        if not docs:
            return []

        if not self.available:
            return [(0.0, d) for d in docs]

        try:
            import torch

            pairs = [[query, doc] for doc in docs]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).tolist()

            if isinstance(scores, float):
                scores = [scores]

            sorted_pairs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return sorted_pairs

        except Exception as e:
            logger.error("Cross-Encoder 推理失败: %s", e)
            return [(0.0, d) for d in docs]


class BM25Reranker:
    """基于 BM25 的重排序器（轻量，无需模型）"""

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        if not docs:
            return []
        tokenized = [list(jieba.cut(doc)) for doc in docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(list(jieba.cut(query)))
        sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_pairs]


def create_reranker(mode: str = "bm25", model_path: Optional[str] = None):
    """工厂函数：创建重排序器实例

    Args:
        mode: "bm25" 或 "cross_encoder"
        model_path: Cross-Encoder 模型路径（仅 mode="cross_encoder" 时需要）
    """
    if mode == "cross_encoder":
        return CrossEncoderReranker(model_path=model_path or "BAAI/bge-reranker-v2-m3")
    return BM25Reranker()
