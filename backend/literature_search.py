import arxiv
import os
from typing import List, Dict, Optional

class LiteratureSearcher:
    """arXiv文献检索器 (兼容 arxiv 3.0)"""
    
    def __init__(self):
        # arxiv 3.0 客户端配置
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        检索arXiv论文
        
        参数:
            query: 搜索关键词
            max_results: 最大返回数量
        
        返回:
            论文列表，每篇包含元数据
        """
        # arxiv 3.0 的 Search 对象
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        try:
            for paper in self.client.results(search):
                # 提取 arXiv ID（兼容不同格式）
                paper_id = paper.entry_id.split("/abs/")[-1] if "/abs/" in paper.entry_id else paper.entry_id
                paper_id = paper_id.split("/pdf/")[-1] if "/pdf/" in paper_id else paper_id
                
                results.append({
                    "id": paper_id,
                    "title": paper.title,
                    "authors": [str(author) for author in paper.authors],
                    "summary": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else "未知",
                    "pdf_url": paper.pdf_url,
                    "primary_category": str(paper.primary_category) if paper.primary_category else "未知"
                })
        except Exception as e:
            print(f"检索失败: {e}")
        
        return results
    
    def search_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        根据 arXiv ID 获取单篇论文信息
        """
        search = arxiv.Search(id_list=[paper_id])
        try:
            paper = next(self.client.results(search))
            return {
                "id": paper_id,
                "title": paper.title,
                "authors": [str(author) for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "未知",
                "pdf_url": paper.pdf_url
            }
        except Exception as e:
            print(f"获取论文失败: {e}")
            return None
    
    def download_paper(self, paper_id: str, download_path: str = "./uploads") -> Optional[str]:
        """
        下载论文PDF
        
        参数:
            paper_id: arXiv ID
            download_path: 保存路径
        
        返回:
            保存的文件路径，失败返回None
        """
        # 确保下载目录存在
        os.makedirs(download_path, exist_ok=True)
        
        search = arxiv.Search(id_list=[paper_id])
        try:
            paper = next(self.client.results(search))
            # 生成安全的文件名
            safe_title = "".join(c for c in paper.title if c.isalnum() or c in " .-_")[:50]
            filename = f"{paper_id}_{safe_title}.pdf"
            filepath = os.path.join(download_path, filename)
            
            # arxiv 3.0 下载方式
            paper.download_pdf(dirpath=download_path, filename=filename)
            
            return filepath if os.path.exists(filepath) else None
        except Exception as e:
            print(f"PDF下载失败: {e}")
            return None


# 单例实例
_searcher = None

def get_searcher() -> LiteratureSearcher:
    global _searcher
    if _searcher is None:
        _searcher = LiteratureSearcher()
    return _searcher