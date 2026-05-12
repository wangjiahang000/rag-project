import os
import pickle
import jieba
import numpy as np
from typing import List, Tuple, Optional
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


class VectorStore:
    """纯 Python 实现的混合检索引擎（ChromaDB + BM25）"""

    def __init__(self, persist_dir: str = "./chroma_data",
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 device: str = "cpu"):
        self.persist_dir = persist_dir
        self.bm25_path = os.path.join(persist_dir, "bm25.pkl")
        self.docs_path = os.path.join(persist_dir, "docs.pkl")
        os.makedirs(persist_dir, exist_ok=True)

        # 初始化 ChromaDB 持久化客户端
        self.client = PersistentClient(path=persist_dir)
        # 嵌入函数（中文用 bge-small-zh，或者保留多语言 MiniLM）
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model, device=device
        )
        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name="docs", embedding_function=self.embed_fn
        )

    def add_texts(self, texts: List[str], metadatas: List[dict] = None, ids: List[str] = None):
        """添加文本到向量库和 BM25 索引"""
        if not texts:
            return

        # 自动生成 IDs
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

        # 加入 ChromaDB
        self.collection.add(documents=texts, metadatas=metadatas or [{}]*len(texts), ids=ids)

        # 更新 BM25 索引
        all_docs = self._load_docs_cache()
        all_docs.extend(texts)
        self._save_docs_cache(all_docs)
        self._build_bm25(all_docs)

    def _load_docs_cache(self) -> List[str]:
        if os.path.exists(self.docs_path):
            with open(self.docs_path, 'rb') as f:
                return pickle.load(f)
        return []

    def _save_docs_cache(self, docs: List[str]):
        with open(self.docs_path, 'wb') as f:
            pickle.dump(docs, f)

    def _build_bm25(self, corpus: List[str]):
        tokenized = [list(jieba.cut(doc)) for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(bm25, f)

    def _load_bm25(self) -> Tuple[Optional[BM25Okapi], List[str]]:
        if not os.path.exists(self.bm25_path) or not os.path.exists(self.docs_path):
            return None, []
        with open(self.bm25_path, 'rb') as f:
            bm25 = pickle.load(f)
        with open(self.docs_path, 'rb') as f:
            docs = pickle.load(f)
        return bm25, docs

    def hybrid_search(self, query: str, k: int = 5,
                      vec_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Tuple[str, float]]:
        """
        混合检索：向量 + BM25 合并打分
        返回：[(文本内容, 得分), ...]
        """
        from collections import defaultdict
        scores = defaultdict(float)

        # ---- 向量检索 ----
        vec_results = self.collection.query(query_texts=[query], n_results=k*2)
        vec_docs = vec_results.get("documents", [[]])[0]
        vec_distances = vec_results.get("distances", [[]])[0]
        for doc, dist in zip(vec_docs, vec_distances):
            # 距离转分数（距离越小越好）
            vec_score = 1.0 / (1.0 + dist) if dist is not None else 0
            scores[doc] += vec_weight * vec_score

        # ---- BM25 检索 ----
        bm25, all_docs = self._load_bm25()
        if bm25 and all_docs:
            tokens = list(jieba.cut(query))
            bm_scores = bm25.get_scores(tokens)
            max_bm = max(bm_scores) if max(bm_scores) > 0 else 1.0
            top_idx = np.argsort(bm_scores)[-k*2:][::-1]
            for idx in top_idx:
                if bm_scores[idx] <= 0:
                    continue
                doc = all_docs[idx]
                bm_norm = bm_scores[idx] / max_bm
                scores[doc] += bm25_weight * bm_norm

        # 排序取 top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]