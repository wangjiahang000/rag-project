import arxiv
import time
from typing import List, Dict
from .config import API_DELAY_SECONDS
from .utils import normalize_arxiv_id


def search_by_category(category: str, limit: int = 50, year: int = None) -> List[Dict]:
    """
    按分类和年份搜索
    
    参数:
        category: 分类标识，如 cs.CL, cs.CV
        limit: 返回数量
        year: 年份，如 2024，不指定则搜索所有年份
    """
    if year:
        # 构建带年份的查询
        start_date = f"{year}01010000"
        end_date = f"{year}12312359"
        query = f"cat:{category} AND submittedDate:[{start_date} TO {end_date}]"
        print(f"  🔍 搜索: 分类={category}, 年份={year}, 数量={limit}")
    else:
        query = f"cat:{category}"
        print(f"  🔍 搜索: 分类={category}, 数量={limit}")
    
    client = arxiv.Client(
        page_size=min(limit, 100),
        delay_seconds=API_DELAY_SECONDS,
        num_retries=3
    )
    
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    try:
        for paper in client.results(search):
            papers.append({
                "arxiv_id": normalize_arxiv_id(paper.get_short_id()),
                "title": paper.title,
                "authors": [str(author) for author in paper.authors],
                "year": paper.published.year if paper.published else year,
                "citation_count": 0,
                "doi": paper.doi if hasattr(paper, 'doi') else None,
                "categories": [category],  # 保存分类信息
                "pdf_url": paper.pdf_url
            })
        
        print(f"    找到 {len(papers)} 篇新论文")
        return papers
        
    except Exception as e:
        print(f"    搜索失败: {e}")
        return []


def download_pdf(arxiv_id: str, pdf_url: str = None) -> str:
    """下载 PDF"""
    from .config import PAPER_PDF_DIR
    import os
    
    arxiv_id = normalize_arxiv_id(arxiv_id)
    filename = f"{arxiv_id}.pdf"
    filepath = os.path.join(PAPER_PDF_DIR, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    try:
        paper = next(client.results(search))
        paper.download_pdf(dirpath=PAPER_PDF_DIR, filename=filename)
        print(f"    PDF 下载成功: {filename}")
        return filepath
    except Exception as e:
        print(f"    PDF 下载失败: {e}")
        return None