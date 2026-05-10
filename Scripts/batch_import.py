#!/usr/bin/env python
"""
批量导入 arXiv 论文（稳健版：内置限流、重试与宽泛的ID校验）
"""
import sys
import os
import re
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# 确保可以导入项目 core 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import RAGSystem
from core.utils import normalize_arxiv_id

# ========== 配置 ==========
CATEGORIES = {
    "cs.CL": 400,
    "cs.CV": 300,
    "cs.LG": 100,
    "cs.AI": 100,
    "cs.RO": 100,
}
BATCH_SIZE = 50           # 每次API请求获取的论文数
BASE_DELAY = 3.0          # 请求间基础延迟（秒）
MAX_RETRIES = 5           # 最大重试次数
BACKOFF_FACTOR = 2        # 退避因子（延迟倍增）

# 宽泛的ID正则：匹配新格式（如2401.00001）和旧格式（如astro-ph/9603021）
BROAD_ID_PATTERN = re.compile(r'^\d{4}\.\d{4,5}$|^[a-z\-]+/\d{7}$|^[a-z\-]+\d{7}$')
def is_broadly_valid(arxiv_id: str) -> bool:
    """接受新旧两种格式的arXiv ID"""
    return bool(BROAD_ID_PATTERN.match(arxiv_id))

def create_requests_session() -> requests.Session:
    """创建带重试策略的 requests Session"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

def fetch_batch(category: str, start: int, max_results: int, session: requests.Session) -> list:
    """
    获取单批论文，返回元数据列表（字典形式）
    """
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()

    import xml.etree.ElementTree as ET
    ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(resp.content)
    papers = []
    for entry in root.findall('arxiv:entry', ns):
        arxiv_id = entry.find('arxiv:id', ns).text.split('/')[-1]
        title = entry.find('arxiv:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('arxiv:summary', ns).text.strip().replace('\n', ' ')
        authors = [au.find('arxiv:name', ns).text for au in entry.findall('arxiv:author', ns)]
        categories = [cat.get('term') for cat in entry.findall('arxiv:category', ns)]
        published = entry.find('arxiv:published', ns).text
        papers.append({
            'arxiv_id': arxiv_id,
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'categories': categories,
            'published': published
        })
    return papers

def fetch_papers_for_category(category: str, target: int) -> list:
    session = create_requests_session()
    collected = []
    start = 0
    retry_count = 0

    while len(collected) < target:
        need = target - len(collected)
        limit = min(BATCH_SIZE, need)

        try:
            papers_meta = fetch_batch(category, start, limit, session)
        except Exception as e:
            print(f"   ⚠️ 批次请求失败 (start={start}): {e}")
            retry_count += 1
            wait = BASE_DELAY * (BACKOFF_FACTOR ** retry_count)
            print(f"   ⏳ 等待 {wait:.1f} 秒后重试...")
            time.sleep(wait)
            continue

        if not papers_meta:
            break

        # ========= 临时：接受所有论文，不过滤ID =========
        valid_papers = papers_meta
        skipped = 0
        # 打印样例ID以便后续分析
        if len(papers_meta) > 0:
            sample_ids = [p['arxiv_id'] for p in papers_meta[:3]]
            print(f"   🔍 示例ID: {sample_ids}")
        # =============================================

        collected.extend(valid_papers)
        print(f"   📥 获取 {len(valid_papers)} 篇（累计 {len(collected)}/{target}）")

        start += limit
        if len(papers_meta) < limit:
            break

        time.sleep(BASE_DELAY)
        retry_count = 0

    return collected[:target]
def import_category(rag: RAGSystem, category: str, target: int):
    """导入指定分类的论文"""
    from types import SimpleNamespace

    print(f"\n{'='*60}")
    print(f"📚 导入分类: {category}，目标: {target} 篇")
    print(f"{'='*60}")

    papers = fetch_papers_for_category(category, target)
    if not papers:
        print(f"⚠️ 未获取到任何论文，跳过 {category}")
        return

    print(f"\n🚀 开始导入 {len(papers)} 篇论文...")
    success = 0
    fail = 0

    for paper_dict in tqdm(papers, desc=category):
        try:
            # 将字典转换为 SimpleNamespace 对象，使其具有属性访问方式
            arxiv_id = normalize_arxiv_id(paper_dict['arxiv_id'])
            year = int(paper_dict['published'][:4]) if paper_dict.get('published') else None
            paper_obj = SimpleNamespace(
                arxiv_id=arxiv_id,
                title=paper_dict['title'],
                abstract=paper_dict['abstract'],
                authors=paper_dict['authors'],
                categories=paper_dict['categories'],
                published=paper_dict['published'],
                year=year
            )
            result = rag.import_arxiv(paper_obj, category)
            if result.success:
                success += 1
            else:
                fail += 1
                print(f"  ❌ 失败: {paper_dict['arxiv_id']} - {result.error}")
        except Exception as e:
            fail += 1
            print(f"  ❌ 异常: {paper_dict['arxiv_id']} - {str(e)}")

    print(f"\n✅ {category} 完成: 成功 {success} 篇，失败 {fail} 篇")
def main():
    rag = RAGSystem()   # 正确初始化 RAGSystem
    for cat, target in CATEGORIES.items():
        try:
            import_category(rag, cat, target)
        except KeyboardInterrupt:
            print(f"\n⚠️ 用户中断，跳过 {cat}")
            continue
        except Exception as e:
            print(f"\n❌ {cat} 导入异常: {e}")
            continue
    print("\n🎉 所有分类导入完成！")

if __name__ == "__main__":
    main()