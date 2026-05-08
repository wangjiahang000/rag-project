import os
import re
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional


class SemanticChunker:
    """基于语义相似度的文本分块器。

    将文本切分为句子，用嵌入模型计算相邻句子的语义相似度，
    相似度低于阈值处断开，形成语义独立的块。
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        max_chunk_size: int = 512,
        min_chunk_size: int = 50,
        threshold: float = 0.35,
    ):
        self.embeddings = embeddings
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.threshold = threshold

    def split_text(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [s for s in sentences if s.strip()]

        # 批量嵌入
        emb_list = self.embeddings.embed_documents(sentences)

        # 计算相邻句子余弦相似度
        sims = []
        for i in range(len(emb_list) - 1):
            sims.append(self._cosine(emb_list[i], emb_list[i + 1]))

        # 低于阈值的点为断点
        breaks = [i + 1 for i, s in enumerate(sims) if s < self.threshold]

        # 按断点切分
        raw_chunks = []
        start = 0
        for b in breaks:
            seg = "".join(sentences[start:b]).strip()
            if seg:
                raw_chunks.append(seg)
            start = b
        seg = "".join(sentences[start:]).strip()
        if seg:
            raw_chunks.append(seg)

        # 合并过小块 + 切割过大块
        return self._normalize(raw_chunks)

    def _split_sentences(self, text: str) -> List[str]:
        # 按句尾标点分割，保留分隔符
        parts = re.split(r'(?<=[。！？\n])', text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _normalize(self, chunks: List[str]) -> List[str]:
        # 合并过小的块（低于 min_chunk_size 的往前/后合并）
        merged = []
        buf = ""
        for c in chunks:
            if len(buf) + len(c) < self.min_chunk_size:
                buf += c
            else:
                if buf:
                    merged.append(buf)
                buf = c
        if buf:
            merged.append(buf)

        # 切割过大的块（超过 max_chunk_size）
        result = []
        for c in merged:
            if len(c) > self.max_chunk_size:
                result.extend(self._fallback_split(c))
            else:
                result.append(c)
        return result

    def _fallback_split(self, text: str) -> List[str]:
        """大块按段落/句子回退切割"""
        parts = re.split(r'(?<=[。！？\n\n])', text)
        chunks = []
        buf = ""
        for p in parts:
            if len(buf) + len(p) > self.max_chunk_size and buf:
                chunks.append(buf)
                buf = p
            else:
                buf += p
        if buf:
            chunks.append(buf)
        return chunks or [text]


class DocumentLoader:
    def __init__(self, embeddings: HuggingFaceEmbeddings = None,
                 chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = embeddings

    def load_file(self, path: str) -> List[Document]:
        if path.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding='utf-8')
        return loader.load()

    def process(self, path: str, metadata: dict = None) -> List[Document]:
        docs = self.load_file(path)
        full_text = "\n".join(d.page_content for d in docs)

        # 语义分块
        if self.embeddings:
            chunker = SemanticChunker(
                embeddings=self.embeddings,
                max_chunk_size=self.chunk_size,
            )
            texts = chunker.split_text(full_text)
        else:
            # 无嵌入模型时回退到 RecursiveCharacterTextSplitter
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
            texts = splitter.split_text(full_text)

        chunks = []
        for i, text in enumerate(texts):
            meta = dict(metadata) if metadata else {}
            meta['chunk_index'] = i
            chunks.append(Document(page_content=text, metadata=meta))
        return chunks
