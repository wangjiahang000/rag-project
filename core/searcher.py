import arxiv
import time
import os
from typing import List, Optional
from .models import PaperInfo
from .utils import normalize_arxiv_id, safe_filename

class ArxivSearcher:
    def __init__(self, delay: float = 3.0):
        self.client = arxiv.Client(page_size=100, delay_seconds=delay, num_retries=3)
    
    def search(self, query: str, max_results: int = 10) -> List[PaperInfo]:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for paper in self.client.results(search):
            results.append(PaperInfo(
                arxiv_id=normalize_arxiv_id(paper.get_short_id()),
                title=paper.title,
                authors=[str(a) for a in paper.authors],
                year=paper.published.year if paper.published else None,
                summary=paper.summary,
                pdf_url=paper.pdf_url
            ))
        return results
    
    def search_by_category(self, category: str, limit: int = 50, year: Optional[int] = None) -> List[PaperInfo]:
        if year:
            query = f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"
        else:
            query = f"cat:{category}"
        return self.search(query, limit)
    
    def download(self, arxiv_id: str, save_dir: str, title: str = "", max_retries: int = 3) -> Optional[str]:
        os.makedirs(save_dir, exist_ok=True)
        filename = safe_filename(title, arxiv_id) if title else f"{arxiv_id}.pdf"
        filepath = os.path.join(save_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10240:
            return filepath
        
        for attempt in range(max_retries):
            try:
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(self.client.results(search))
                paper.download_pdf(dirpath=save_dir, filename=filename)
                if os.path.getsize(filepath) > 10240:
                    return filepath
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5 * (attempt + 1))
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None