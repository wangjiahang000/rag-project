import arxiv
import time
import os
from typing import List, Dict, Optional

from .config import API_DELAY_SECONDS
from .utils import normalize_arxiv_id, safe_pdf_filename

# 模块级单例客户端，复用连接
_client: Optional[arxiv.Client] = None

def get_arxiv_client() -> arxiv.Client:
    """获取复用的 arXiv 客户端实例"""
    global _client
    if _client is None:
        _client = arxiv.Client(
            page_size=100,
            delay_seconds=API_DELAY_SECONDS,
            num_retries=3
        )
    return _client


def search_by_category(category: str, limit: int = 50, year: int = None) -> List[Dict]:
    """
    按分类和年份搜索

    参数:
        category: 分类标识，如 cs.CL, cs.CV
        limit: 返回数量
        year: 年份，如 2024，不指定则搜索所有年份
    """
    if year:
        start_date = f"{year}01010000"
        end_date = f"{year}12312359"
        query = f"cat:{category} AND submittedDate:[{start_date} TO {end_date}]"
        print(f"  🔍 搜索: 分类={category}, 年份={year}, 数量={limit}")
    else:
        query = f"cat:{category}"
        print(f"  🔍 搜索: 分类={category}, 数量={limit}")

    client = get_arxiv_client()
    client.page_size = min(limit, 100)  # 动态调整每页大小

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
                "categories": [category],
                "pdf_url": paper.pdf_url
            })

        print(f"    找到 {len(papers)} 篇新论文")
        return papers

    except Exception as e:
        print(f"    搜索失败: {e}")
        return []


def download_pdf(arxiv_id: str, max_retries: int = 3) -> Optional[str]:
    """
    下载论文 PDF，带重试和完整性校验

    参数:
        arxiv_id: arXiv 论文 ID
        max_retries: 最大重试次数（默认 3）

    返回:
        成功时返回文件路径，失败返回 None
    """
    from .config import PAPER_PDF_DIR

    arxiv_id = normalize_arxiv_id(arxiv_id)
    os.makedirs(PAPER_PDF_DIR, exist_ok=True)

    client = get_arxiv_client()

    for attempt in range(max_retries):
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search))

            # 生成安全的文件名（包含标题和哈希防冲突）
            filename = safe_pdf_filename(paper.title, arxiv_id)
            filepath = os.path.join(PAPER_PDF_DIR, filename)

            # 如果文件已存在且大小正常，直接返回
            if os.path.exists(filepath) and os.path.getsize(filepath) > 10240:
                print(f"    PDF 已存在且有效: {filename}")
                return filepath

            # 下载 PDF
            paper.download_pdf(dirpath=PAPER_PDF_DIR, filename=filename)

            # 验证下载完整性（至少 10KB）
            if os.path.exists(filepath) and os.path.getsize(filepath) > 10240:
                print(f"    PDF 下载成功: {filename}")
                return filepath
            else:
                # 删除不完整的文件
                if os.path.exists(filepath):
                    os.unlink(filepath)
                print(f"    下载文件无效 (尝试 {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避: 2, 4, 8 秒
                continue

        except StopIteration:
            print(f"    论文 {arxiv_id} 在 arXiv 中不存在")
            return None
        except arxiv.HTTPError as e:
            print(f"    arXiv API 错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (2 ** attempt))
        except Exception as e:
            print(f"    PDF 下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None