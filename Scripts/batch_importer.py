import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import RAGSystem
from config import settings
from tqdm import tqdm

def main():
    rag = RAGSystem()
    categories = {
        "cs.CL": 400,
        "cs.CV": 300,
        "cs.LG": 100,
    }
    for cat, target in categories.items():
        print(f"\n📚 导入分类 {cat}，目标 {target} 篇")
        papers = rag.searcher.search_by_category(cat, target)
        for paper in tqdm(papers, desc=cat):
            result = rag.import_arxiv(paper, cat)
            if not result.success:
                print(f"  失败: {paper.arxiv_id} - {result.error}")

if __name__ == "__main__":
    main()