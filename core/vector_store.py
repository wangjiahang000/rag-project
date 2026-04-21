import os
import pickle
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Tuple, Optional

class VectorStoreManager:
    def __init__(self, model_path: str, persist_dir: str, device: str = "cpu"):
        self.persist_dir = persist_dir
        self.bm25_path = os.path.join(persist_dir, "bm25.pkl")
        self.docs_cache = os.path.join(persist_dir, "docs.pkl")
        
        os.makedirs(persist_dir, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            try:
                self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
            except:
                pass
    
    def add_documents(self, docs: List[Document]):
        if not docs:
            return
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_dir)
        else:
            self.vectorstore.add_documents(docs)
        self.vectorstore.persist()
        self._update_bm25(docs)
    
    def _update_bm25(self, new_docs: List[Document]):
        all_docs = []
        if os.path.exists(self.docs_cache):
            with open(self.docs_cache, 'rb') as f:
                all_docs = pickle.load(f)
        all_docs.extend(new_docs)
        with open(self.docs_cache, 'wb') as f:
            pickle.dump(all_docs, f)
        
        corpus = [list(jieba.cut(doc.page_content)) for doc in all_docs]
        bm25 = BM25Okapi(corpus)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(bm25, f)
    
    def _load_bm25(self) -> Tuple[Optional[BM25Okapi], List[Document]]:
        if not os.path.exists(self.bm25_path) or not os.path.exists(self.docs_cache):
            return None, []
        with open(self.bm25_path, 'rb') as f:
            bm25 = pickle.load(f)
        with open(self.docs_cache, 'rb') as f:
            docs = pickle.load(f)
        return bm25, docs
    
    def hybrid_search(self, query: str, k: int = 5, 
                     vec_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Document]:
        results = {}
        # 向量检索
        if self.vectorstore:
            vec_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
            for doc, score in vec_results:
                if score > 1.5:
                    continue
                vec_score = 1.0 / (1.0 + score)
                key = doc.page_content
                results[key] = {'doc': doc, 'score': vec_weight * vec_score}
        
        # BM25检索
        bm25, docs = self._load_bm25()
        if bm25 and docs:
            tokens = list(jieba.cut(query))
            scores = bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[-k*2:][::-1]
            max_score = max(scores) if max(scores) > 0 else 1.0
            for idx in top_idx:
                if scores[idx] <= 0:
                    continue
                doc = docs[idx]
                key = doc.page_content
                bm25_norm = scores[idx] / max_score
                if key in results:
                    results[key]['score'] += bm25_weight * bm25_norm
                else:
                    results[key] = {'doc': doc, 'score': bm25_weight * bm25_norm}
        
        sorted_docs = sorted(results.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs[:k]]