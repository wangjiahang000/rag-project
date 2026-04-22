#!/usr/bin/env python
"""
批量导入 arXiv 论文（按年份倒序搜索，凑够目标数量）
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RAGSystem
from tqdm import tqdm

# 配置
CATEGORIES = {
    "cs.CL": 400,
    "cs.CV": 300,
    "cs.LG": 100,
    "cs.AI": 100,
    "cs.RO": 100,
}
START_YEAR = 2026  # 从今年开始倒序搜索
BATCH_SIZE = 50    # 每次搜索的数量


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """检查是否为新的 arXiv ID 格式（YYMM.NNNNN）"""
    return bool(re.match(r'^\d{4}\.\d{4,5}$', arxiv_id))


def import_category(rag: RAGSystem, category: str, target: int):
    """按分类导入，倒序搜索年份直到凑够目标数量"""
    print(f"\n{'='*60}")
    print(f"📚 导入分类: {category}，目标: {target} 篇")
    print(f"{'='*60}")
    
    collected_papers = []
    current_year = START_YEAR
    
    # 倒序搜索，直到凑够目标数量
    while len(collected_papers) < target and current_year >= 2000:
        need = target - len(collected_papers)
        limit = min(BATCH_SIZE, need)
        
        print(f"\n📅 搜索年份: {current_year}，需要 {need} 篇，本次搜索 {limit} 篇")
        
        papers = rag.searcher.search_by_category(category, limit=limit, year=current_year)
        
        # 过滤掉旧格式 ID
        valid_papers = [p for p in papers if is_valid_arxiv_id(p.arxiv_id)]
        skipped = len(papers) - len(valid_papers)
        
        if skipped > 0:
            print(f"   ⚠️ 跳过 {skipped} 篇旧格式论文")
        
        collected_papers.extend(valid_papers)
        print(f"   ✅ 收集 {len(valid_papers)} 篇有效论文，累计 {len(collected_papers)}/{target}")
        
        current_year -= 1
    
    # 截取目标数量
    papers_to_import = collected_papers[:target]
    print(f"\n🚀 开始导入 {len(papers_to_import)} 篇论文...")
    
    success_count = 0
    fail_count = 0
    
    for paper in tqdm(papers_to_import, desc=category):
        result = rag.import_arxiv(paper, category)
        if result.success:
            success_count += 1
        else:
            fail_count += 1
            print(f"  ❌ 失败: {paper.arxiv_id} - {result.error}")
    
    print(f"\n✅ {category} 完成: 成功 {success_count} 篇，失败 {fail_count} 篇")


def main():
    rag = RAGSystem()
    
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