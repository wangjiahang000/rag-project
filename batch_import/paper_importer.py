import os
import PyPDF2
from typing import Dict, Optional, List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import PAPER_TXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .utils import normalize_arxiv_id, safe_filename
from .mysql_manager import (
    paper_exists, save_paper, log_import, mark_as_vectorized,
    paper_exists_and_vectorized
)
from .semantic_scholar_client import download_pdf


def pdf_to_txt(pdf_path: str, arxiv_id: str) -> Optional[str]:
    """PDF 转 TXT 文件"""
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
            # 验证 PDF 有效性
            if len(reader.pages) == 0:
                print(f"  ❌ PDF 无效: 没有页面")
                return None
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(f"--- 第 {page_num} 页 ---\n")
                    text_content.append(text)
                    text_content.append("\n\n")
        
        if text_content:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(''.join(text_content))
            print(f"  TXT 提取成功: {txt_filename} ({len(text_content)} 字符)")
            return txt_path
        else:
            print(f"  TXT 提取失败: 无文本内容")
            return None
    except Exception as e:
        print(f"  TXT 提取异常: {e}")
        return None


def load_and_chunk(
    txt_path: str, 
    arxiv_id: str, 
    title: str, 
    category: str, 
    authors: str = "", 
    year: int = 0
) -> List[Document]:
    """
    读取 TXT 文件并分块，添加完整元数据
    
    返回: List[Document] 每个 Document 包含 page_content 和 metadata
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) < 100:
            print(f"  ⚠️ 文本过短 ({len(content)} 字符)，跳过向量化")
            return []
        
        # 创建单个 Document（包含完整论文内容）
        doc = Document(
            page_content=content,
            metadata={
                "source": f"{arxiv_id}.txt",
                "paper_id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": title[:500] if title else "",
                "category": category,
                "authors": authors[:300] if authors else "",
                "year": year if year else 0
            }
        )
        
        # 分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = splitter.split_documents([doc])
        
        # 为每个 chunk 添加索引
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_total"] = len(chunks)
        
        print(f"  📊 分块完成: {len(chunks)} 个 chunks")
        return chunks
        
    except Exception as e:
        print(f"  ❌ 分块失败: {e}")
        return []


def import_single_paper(paper_info: Dict, category: str) -> Tuple[bool, str, Optional[List[Document]], Optional[Dict]]:
    """
    导入单篇论文，返回 (success, arxiv_id, chunks, metadata)
    
    参数:
        paper_info: 论文元数据字典
        category: 分类标识
    
    返回:
        success: bool - 是否成功
        arxiv_id: str - 论文ID
        chunks: List[Document] | None - 分块后的文档列表（成功时返回）
        metadata: Dict | None - 论文元数据（成功时返回）
    """
    arxiv_id = normalize_arxiv_id(paper_info.get("arxiv_id", ""))
    
    if not arxiv_id:
        log_import("unknown", category, "failed", "缺少 arXiv ID")
        return (False, arxiv_id, None, None)
    
    # 检查是否已存在及是否已向量化
    exists, is_vectorized = paper_exists_and_vectorized(arxiv_id)
    
    if exists and is_vectorized:
        print(f"  ⏭️ 跳过: {arxiv_id} 已完成向量化")
        return (True, arxiv_id, None, None)
    
    if exists and not is_vectorized:
        print(f"  🔄 重试: {arxiv_id} 已入库但未向量化，重新处理")
    
    print(f"\n📥 导入: {arxiv_id}")
    title = paper_info.get('title', '')
    print(f"   标题: {title[:60]}..." if len(title) > 60 else f"   标题: {title}")
    
    # 1. 下载 PDF
    pdf_path = download_pdf(arxiv_id)
    if not pdf_path:
        log_import(arxiv_id, category, "failed", "PDF 下载失败")
        return (False, arxiv_id, None, None)
    
    # 2. PDF 转 TXT
    txt_path = pdf_to_txt(pdf_path, arxiv_id)
    if not txt_path:
        log_import(arxiv_id, category, "failed", "PDF 转 TXT 失败")
        return (False, arxiv_id, None, None)
    
    # 3. 准备数据库记录
    authors = str(paper_info.get("authors", []))
    year = paper_info.get("year")
    
    paper_data = {
        "arxiv_id": arxiv_id,
        "title": paper_info.get("title", "")[:5000],
        "authors": authors,
        "year": year,
        "journal": "",
        "doi": paper_info.get("doi", ""),
        "categories": category,
        "citation_count": 0,
        "pdf_path": pdf_path,
        "txt_path": txt_path
    }
    
    if not save_paper(paper_data):
        log_import(arxiv_id, category, "failed", "MySQL 保存失败")
        return (False, arxiv_id, None, None)
    
    # 4. 分块
    chunks = load_and_chunk(txt_path, arxiv_id, title, category, authors, year)
    
    if not chunks:
        log_import(arxiv_id, category, "failed", "分块失败")
        return (False, arxiv_id, None, None)
    
    # 5. 准备元数据
    metadata = {
        "arxiv_id": arxiv_id,
        "title": title[:100],
        "category": category,
        "authors": authors[:100],
        "year": year,
        "chunk_count": len(chunks)
    }
    
    log_import(arxiv_id, category, "success")
    print(f"  ✅ 导入成功: {arxiv_id} ({len(chunks)} chunks)")
    
    return (True, arxiv_id, chunks, metadata)