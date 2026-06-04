import os
import pickle
import logging
import time
import jieba
import numpy as np
from typing import Dict, List, Tuple, Optional
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# HuggingFace 国内镜像（不干扰用户已有配置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── BM25 全局缓存（进程级，避免每次查询反序列化 pickle）──
_bm25_cache: dict[str, tuple[float, float, Optional[BM25Okapi], list]] = {}
"""key -> (bm25_mtime, docs_mtime, bm25, docs)"""


def _get_bm25_cached(bm25_path: str, docs_path: str) -> tuple[Optional[BM25Okapi], list]:
    """带文件变更检测的 BM25 缓存"""
    cache_key = f"{bm25_path}|{docs_path}"
    bm25_mtime = os.path.getmtime(bm25_path) if os.path.exists(bm25_path) else 0
    docs_mtime = os.path.getmtime(docs_path) if os.path.exists(docs_path) else 0

    cached = _bm25_cache.get(cache_key)
    if cached and cached[0] == bm25_mtime and cached[1] == docs_mtime:
        return cached[2], cached[3]

    if not os.path.exists(bm25_path) or not os.path.exists(docs_path):
        _bm25_cache[cache_key] = (bm25_mtime, docs_mtime, None, [])
        return None, []

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    with open(docs_path, 'rb') as f:
        docs = pickle.load(f)

    _bm25_cache[cache_key] = (bm25_mtime, docs_mtime, bm25, docs)
    logger.debug("BM25 缓存已更新: %s", cache_key)
    return bm25, docs


def _invalidate_bm25_cache(bm25_path: str, docs_path: str):
    """重建/修改 BM25 后清除缓存"""
    cache_key = f"{bm25_path}|{docs_path}"
    _bm25_cache.pop(cache_key, None)


class VectorStore:
    """纯 Python 实现的混合检索引擎（ChromaDB + BM25）"""

    # 默认模型路径：优先本地，回退 HF 模型名
    _DEFAULT_MODEL = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models", "bge-small-zh-v1.5"
    )
    if not os.path.exists(_DEFAULT_MODEL):
        _DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, persist_dir: str = "./chroma_data",
                 embedding_model: str = None,
                 device: str = "cpu"):
        if embedding_model is None:
            embedding_model = self._DEFAULT_MODEL
        self.persist_dir = persist_dir
        self.bm25_path = os.path.join(persist_dir, "bm25.pkl")
        self.docs_path = os.path.join(persist_dir, "docs.pkl")
        self.embedding_model = embedding_model
        self.device = device
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

    def load_and_index(
        self,
        txt_dir: str,
        source_prefix: str = "",
        chunk_strategy: str = "structure",
        clear_existing: bool = False,
        max_chunk_size: int = 512,
        batch_size: int = 50,
        max_files: int = 0,
    ) -> Dict:
        """批量加载 TXT 文件，结构化分块后写入向量库

        每处理 batch_size 个文件后写入一次向量库，避免内存占用过大。
        支持断点续传：已写入的文件会跳过。

        Args:
            txt_dir: 包含 .txt 文件的目录
            source_prefix: 来源前缀（如 "arxiv"）
            chunk_strategy: 分块策略 ("structure" | "recursive")
            clear_existing: 是否清空现有索引
            max_chunk_size: 单块最大字符数
            batch_size: 每处理多少文件后写入一次（默认 50）
            max_files: 最多处理文件数（0 = 全部处理）

        Returns:
            {"total_files": N, "total_chunks": N, "skipped": N, "files": [...]}
        """
        import glob
        from content_core.data.chunker import chunk_text as do_chunk

        if clear_existing:
            self.clear()

        txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
        if max_files > 0:
            txt_files = txt_files[:max_files]
        if not txt_files:
            logger.warning("未找到 TXT 文件: %s", txt_dir)
            return {"total_files": 0, "total_chunks": 0, "skipped": 0, "files": []}

        all_texts = []
        all_metadatas = []
        all_ids = []
        file_stats = []
        total_chunks = 0
        skipped = 0

        for idx, fp in enumerate(txt_files):
            filename = os.path.basename(fp)
            source = f"{source_prefix}/{os.path.splitext(filename)[0]}" if source_prefix else os.path.splitext(filename)[0]

            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                logger.error("读取失败 %s: %s", fp, e)
                skipped += 1
                continue

            if not text.strip():
                skipped += 1
                continue

            chunks = do_chunk(
                text,
                source=source,
                strategy=chunk_strategy,
                max_chunk_size=max_chunk_size,
            )

            if not chunks:
                skipped += 1
                continue

            for c in chunks:
                all_texts.append(c["text"])
                all_metadatas.append(c["metadata"])
                all_ids.append(f"{source}_chunk_{c['metadata']['chunk_index']}")

            file_stats.append({"file": filename, "chunks": len(chunks)})
            total_chunks += len(chunks)

            # 每 batch_size 个文件写入一次
            if (idx + 1) % batch_size == 0 or (idx + 1) == len(txt_files):
                if all_texts:
                    self.add_texts(all_texts, all_metadatas, all_ids)
                    logger.info(
                        "批次写入: 已处理 %d/%d 个文件, 累计 %d chunks",
                        idx + 1, len(txt_files), total_chunks,
                    )
                    all_texts.clear()
                    all_metadatas.clear()
                    all_ids.clear()

        result = {
            "total_files": len(file_stats),
            "total_chunks": total_chunks,
            "skipped": skipped,
            "files": file_stats,
        }
        logger.info("批量索引完成: 文件=%d, chunks=%d, 跳过=%d", len(file_stats), total_chunks, skipped)
        return result

    def _existing_sources(self) -> set:
        """返回已索引的 source 集合（用于增量跳过）"""
        try:
            all_meta = self.collection.get(include=["metadatas"])
            sources = set()
            for m in all_meta.get("metadatas", []) or []:
                if m and "source" in m:
                    sources.add(m["source"])
            return sources
        except Exception:
            return set()

    def incremental_index(
        self,
        txt_dir: str,
        source_prefix: str = "",
        chunk_strategy: str = "structure",
        max_chunk_size: int = 512,
        batch_size: int = 50,
        max_files: int = 0,
    ) -> Dict:
        """增量索引：只处理尚未入库的 TXT 文件

        通过比对 ChromaDB 中已存在的 source 元数据来跳过已索引的文件。
        最后增量重建 BM25（全量 corpus + 新增合并）。
        """
        import glob
        from content_core.data.chunker import chunk_text as do_chunk

        existing = self._existing_sources()
        logger.info("增量索引: 已有 %d 个 source 在库中", len(existing))

        txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
        if max_files > 0:
            txt_files = txt_files[:max_files]
        if not txt_files:
            return {"total_files": 0, "total_chunks": 0, "skipped": 0, "new_files": 0, "files": []}

        all_texts = []
        all_metadatas = []
        all_ids = []
        file_stats = []
        total_chunks = 0
        new_files = 0

        for idx, fp in enumerate(txt_files):
            filename = os.path.basename(fp)
            source = f"{source_prefix}/{os.path.splitext(filename)[0]}" if source_prefix else os.path.splitext(filename)[0]

            # 跳过已索引的文件
            if source in existing:
                continue

            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                logger.error("读取失败 %s: %s", fp, e)
                continue

            if not text.strip():
                continue

            chunks = do_chunk(text, source=source, strategy=chunk_strategy, max_chunk_size=max_chunk_size)
            if not chunks:
                continue

            for c in chunks:
                all_texts.append(c["text"])
                all_metadatas.append(c["metadata"])
                all_ids.append(f"{source}_chunk_{c['metadata']['chunk_index']}")

            file_stats.append({"file": filename, "chunks": len(chunks)})
            total_chunks += len(chunks)
            new_files += 1

            if (idx + 1) % batch_size == 0 or (idx + 1) == len(txt_files):
                if all_texts:
                    self.add_texts(all_texts, all_metadatas, all_ids)
                    logger.info("增量写入: %d/%d 文件, %d chunks", idx + 1, len(txt_files), total_chunks)
                    all_texts.clear()
                    all_metadatas.clear()
                    all_ids.clear()

        # 增量重建 BM25
        if new_files > 0:
            old_docs = self._load_docs_cache()
            self._build_bm25(old_docs)  # _build_bm25 会写入 pickle 并失效缓存

        result = {
            "total_files": len(txt_files),
            "new_files": new_files,
            "total_chunks": total_chunks,
            "files": file_stats,
        }
        logger.info("增量索引完成: 新增 %d/%d 文件, %d chunks", new_files, len(txt_files), total_chunks)
        return result

    def clear(self):
        """清空向量库和 BM25 索引"""
        try:
            self.client.delete_collection("docs")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="docs", embedding_function=self.embed_fn
        )
        # 删除 BM25 缓存文件
        for p in [self.bm25_path, self.docs_path]:
            if os.path.exists(p):
                os.remove(p)
        _invalidate_bm25_cache(self.bm25_path, self.docs_path)
        logger.info("向量库已清空")

    def count(self) -> int:
        """返回当前文档数"""
        return self.collection.count()

    def add_texts(self, texts: List[str], metadatas: List[dict] = None, ids: List[str] = None):
        """添加文本到向量库和 BM25 索引

        ChromaDB 单次写入上限约 5461 条，超过时分批写入。
        """
        if not texts:
            return

        # 自动生成 IDs
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

        metadatas = metadatas or [{}] * len(texts)

        # ChromaDB 分批写入（上限 5000 条/批）
        BATCH_SIZE = 5000
        for start in range(0, len(texts), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(texts))
            self.collection.add(
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

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
        _invalidate_bm25_cache(self.bm25_path, self.docs_path)

    def _load_bm25(self) -> Tuple[Optional[BM25Okapi], List[str]]:
        return _get_bm25_cached(self.bm25_path, self.docs_path)

    def hybrid_search(self, query: str, k: int = 5,
                      vec_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Tuple[str, float, dict]]:
        """
        混合检索：向量 + BM25 合并打分
        返回：[(文本内容, 得分, 元数据字典), ...]
        """
        from collections import defaultdict
        scores = defaultdict(float)
        doc_meta = {}  # text -> metadata

        # ---- 向量检索 ----
        vec_results = self.collection.query(query_texts=[query], n_results=k*2)
        vec_docs = vec_results.get("documents", [[]])[0]
        vec_distances = vec_results.get("distances", [[]])[0]
        vec_metadatas = vec_results.get("metadatas", [[]])[0]
        for doc, dist, meta in zip(vec_docs, vec_distances, vec_metadatas):
            vec_score = 1.0 / (1.0 + dist) if dist is not None else 0
            scores[doc] += vec_weight * vec_score
            if meta:
                doc_meta[doc] = meta

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
                if doc not in doc_meta:
                    doc_meta[doc] = {}

        # 排序取 top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(text, score, doc_meta.get(text, {})) for text, score in sorted_items[:k]]