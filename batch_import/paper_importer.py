import os
import PyPDF2
from typing import Dict, Optional
from .config import PAPER_TXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .utils import normalize_arxiv_id, safe_filename
from .mysql_manager import paper_exists, save_paper, log_import
from .semantic_scholar_client import download_pdf


def pdf_to_txt(pdf_path: str, arxiv_id: str) -> Optional[str]:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    txt_filename = f"{safe_filename(arxiv_id)}.txt"
    txt_path = os.path.join(PAPER_TXT_DIR, txt_filename)
    
    if os.path.exists(txt_path):
        print(f"  TXT 已存在: {txt_filename}")
        return txt_path
    
    try:
        text_content = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(f"--- 第 {page_num} 页 ---\n")
                    text_content.append(text)
                    text_content.append("\n\n")
        
        if text_content:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(''.join(text_content))
            print(f"  TXT 提取成功: {txt_filename}")
            return txt_path
        else:
            print(f"  TXT 提取失败: 无文本内容")
            return None
    except Exception as e:
        print(f"  TXT 提取异常: {e}")
        return None


def vectorize_paper(txt_path: str, arxiv_id: str, title: str):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from backend.vector_store import get_vector_store, create_vector_store
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) < 100:
            print(f"  ⚠️ 文本过短，跳过向量化")
            return False
        
        doc = Document(
            page_content=content,
            metadata={"source": f"{arxiv_id}.txt", "arxiv_id": arxiv_id, "title": title}
        )
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = splitter.split_documents([doc])
        
        existing_store = get_vector_store()
        if existing_store:
            existing_store.add_documents(chunks)
            existing_store.persist()
            print(f"  ✅ 向量化成功: {arxiv_id} (追加，{len(chunks)} 块)")
        else:
            create_vector_store(chunks)
            print(f"  ✅ 向量化成功: {arxiv_id} (新建，{len(chunks)} 块)")
        
        return True
    except Exception as e:
        print(f"  ❌ 向量化失败: {e}")
        return False


def import_single_paper(paper_info: Dict, category: str) -> bool:
    arxiv_id = normalize_arxiv_id(paper_info.get("arxiv_id", ""))
    
    if not arxiv_id:
        log_import("unknown", category, "failed", "缺少 arXiv ID")
        return False
    
    if paper_exists(arxiv_id):
        print(f"  ⏭️ 跳过: {arxiv_id} 已存在")
        return True
    
    print(f"\n📥 导入: {arxiv_id}")
    title = paper_info.get('title', '')
    print(f"   标题: {title[:60]}..." if len(title) > 60 else f"   标题: {title}")
    
    pdf_path = download_pdf(arxiv_id)
    if not pdf_path:
        log_import(arxiv_id, category, "failed", "PDF 下载失败")
        return False
    
    txt_path = pdf_to_txt(pdf_path, arxiv_id)
    if not txt_path:
        log_import(arxiv_id, category, "failed", "PDF 转 TXT 失败")
        return False
    
    paper_data = {
        "arxiv_id": arxiv_id,
        "title": paper_info.get("title", "")[:5000],
        "authors": str(paper_info.get("authors", [])),
        "year": paper_info.get("year"),
        "journal": "",
        "doi": paper_info.get("doi", ""),
        "categories": category,
        "citation_count": 0,
        "pdf_path": pdf_path,
        "txt_path": txt_path
    }
    
    if not save_paper(paper_data):
        log_import(arxiv_id, category, "failed", "MySQL 保存失败")
        return False
    
    if not vectorize_paper(txt_path, arxiv_id, paper_info.get("title", "")):
        log_import(arxiv_id, category, "failed", "向量化失败")
        return False
    
    log_import(arxiv_id, category, "success")
    print(f"  ✅ 导入成功: {arxiv_id}")
    return True