import os
import pdfplumber
from typing import List, Optional, Dict
from .loader import DocumentLoader
from .searcher import ArxivSearcher
from .vector_store import VectorStoreManager
from .mysql_client import MySQLClient
from .models import PaperInfo, ImportResult
from .utils import normalize_arxiv_id
from config import settings
from openai import OpenAI

class RAGSystem:
    def __init__(self):
        self.loader = DocumentLoader(settings.chunk_size, settings.chunk_overlap)
        self.searcher = ArxivSearcher()
        self.vector_store = VectorStoreManager(
            model_path=settings.embedding_model,
            persist_dir=settings.chroma_dir,
            device=settings.embedding_device
        )
        self.mysql = MySQLClient(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset='utf8mb4'
        )
        self.mysql.init_db()
        
        self.llm = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url
        )
        
        os.makedirs(settings.papers_pdf_dir, exist_ok=True)
        os.makedirs(settings.papers_txt_dir, exist_ok=True)
    
    # ---------- 文件上传 ----------
    def upload(self, file_path: str, metadata: dict = None) -> int:
        chunks = self.loader.process(file_path, metadata)
        self.vector_store.add_documents(chunks)
        return len(chunks)
    
    # ---------- arXiv 相关 ----------
    def search_arxiv(self, query: str, max_results: int = 10) -> List[PaperInfo]:
        return self.searcher.search(query, max_results)
    
    def import_arxiv(self, paper: PaperInfo, category: str = "manual") -> ImportResult:
        arxiv_id = paper.arxiv_id
        pdf_path = self.searcher.download(arxiv_id, settings.papers_pdf_dir, paper.title)
        if not pdf_path:
            return ImportResult(arxiv_id, paper.title, False, 0, "下载失败")
        
        txt_path = os.path.join(settings.papers_txt_dir, f"{arxiv_id}.txt")
        if not os.path.exists(txt_path):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as e:
                return ImportResult(arxiv_id, paper.title, False, 0, f"PDF解析失败: {e}")
        
        self.mysql.save_paper({
            'arxiv_id': arxiv_id,
            'title': paper.title,
            'authors': ', '.join(paper.authors),
            'year': paper.year,
            'category': category,
            'pdf_path': pdf_path,
            'txt_path': txt_path
        })
        
        chunks = self.loader.process(txt_path, {
            'arxiv_id': arxiv_id,
            'title': paper.title,
            'category': category,
            'source': f"{arxiv_id}.txt"
        })
        self.vector_store.add_documents(chunks)
        self.mysql.mark_vectorized([arxiv_id])
        return ImportResult(arxiv_id, paper.title, True, len(chunks))
    
    def import_arxiv_category(self, category: str, target: int = 100, year: Optional[int] = None) -> List[ImportResult]:
        papers = self.searcher.search_by_category(category, target, year)
        results = []
        for p in papers:
            result = self.import_arxiv(p, category)
            results.append(result)
            print(f"📥 {result.arxiv_id}: {'✅' if result.success else '❌'} {result.chunks} chunks")
        return results
    
    # ---------- 问答 ----------
    def chat(self, question: str, history: List[tuple] = None) -> str:
        docs = self.vector_store.hybrid_search(
            question, 
            k=settings.retrieval_k,
            vec_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight
        )
        if not docs:
            context = "暂无相关文献"
            sources = []
        else:
            by_source = {}
            for doc in docs:
                src = doc.metadata.get('arxiv_id', 'unknown')
                by_source.setdefault(src, []).append(doc.page_content)
            context = "\n\n---\n\n".join(["\n".join(chunks) for chunks in by_source.values()])
            sources = list(by_source.keys())
        
        messages = [{
            "role": "system",
            "content": f"你是一个学术助手。基于以下文献回答问题，若无法回答请说明。\n\n文献：\n{context}"
        }]
        if history:
            for u, a in history[-5:]:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": question})
        
        resp = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7
        )
        answer = resp.choices[0].message.content
        
        if sources:
            titles = self.mysql.get_titles(sources)
            refs = []
            for i, sid in enumerate(sources, 1):
                t = titles.get(sid, sid)
                if len(t) > 80:
                    t = t[:77] + "..."
                refs.append(f"{i}. [{sid}] {t}")
            answer += "\n\n---\n📚 参考来源：\n" + "\n".join(refs)
        return answer