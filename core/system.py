import os
import logging
import pdfplumber
from typing import List, Optional, Dict, Tuple
from .loader import DocumentLoader
from .searcher import ArxivSearcher
from .vector_store import VectorStoreManager
from .mysql_client import MySQLClient
from .models import PaperInfo, ImportResult
from .utils import normalize_arxiv_id
from config import settings
from openai import OpenAI

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.loader = DocumentLoader(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
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
    
    # ---------- PDF 提取（含表格转 Markdown）----------
    @staticmethod
    def _extract_pdf_with_tables(pdf_path: str) -> str:
        import pdfplumber
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 1. 常规文本
                body = page.extract_text() or ""

                # 2. 检测表格 → 转 Markdown
                tables = page.extract_tables()
                table_mds = []
                for table in tables:
                    if not table or len(table) < 1:
                        continue
                    rows = []
                    for row in table:
                        cells = [cell.replace("\n", " ") if cell else "" for cell in row]
                        rows.append("| " + " | ".join(cells) + " |")
                    # 加分隔行（第二行为 ---）
                    header = rows[0]
                    sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
                    md = header + "\n" + sep
                    if len(rows) > 1:
                        md += "\n" + "\n".join(rows[1:])
                    table_mds.append(md)

                # 3. 合并：文本 + 表格
                page_text = body
                if table_mds:
                    page_text += "\n\n【表格】\n" + "\n\n".join(table_mds)
                pages_text.append(page_text)
        return "\n\n".join(pages_text)

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
                text = self._extract_pdf_with_tables(pdf_path)
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
        logger.info("=" * 60)
        logger.info("用户问题: %s", question)
        logger.info("=" * 60)

        results = self.vector_store.hybrid_search(
            question,
            k=settings.retrieval_k,
            vec_weight=settings.vector_weight,
            bm25_weight=0
        )
        docs = [d for d, _ in results]

        logger.info("-" * 60)
        logger.info("检索到的文献片段 (共 %d 条):", len(results))
        logger.info("-" * 60)
        for i, (doc, score) in enumerate(results, 1):
            src = doc.metadata.get('source', doc.metadata.get('arxiv_id', 'unknown'))
            chunk_idx = doc.metadata.get('chunk_index', '?')
            logger.info("--- [%d] 相关度: %.4f | 来源: %s | 分块: %s ---", i, score, src, chunk_idx)
            logger.info(doc.page_content)
            logger.info("")

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

        logger.info("-" * 60)
        logger.info("发送给 LLM 的上下文 (按来源归并):")
        logger.info("-" * 60)
        logger.info(context)
        
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