from fastapi import APIRouter, Depends
from core import RAGSystem, PaperInfo
from backend.dependencies import get_rag
from backend.models import SearchRequest, SearchResponse, AddPaperRequest

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search_arxiv(request: SearchRequest, rag: RAGSystem = Depends(get_rag)):
    papers = rag.search_arxiv(request.query, request.max_results)
    paper_dicts = [
        {
            "id": p.arxiv_id,
            "title": p.title,
            "authors": p.authors,
            "authors_display": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
            "summary": p.summary[:500],
            "published": str(p.year) if p.year else "未知",
            "pdf_url": p.pdf_url
        }
        for p in papers
    ]
    return SearchResponse(status="success", papers=paper_dicts, count=len(paper_dicts))

@router.post("/add")
async def add_paper(request: AddPaperRequest, rag: RAGSystem = Depends(get_rag)):
    paper = PaperInfo(
        arxiv_id=request.paper_id,
        title=request.paper_title,
        category=request.category
    )
    result = rag.import_arxiv(paper, request.category)
    return {
        "status": "success" if result.success else "error",
        "message": f"已添加 {result.arxiv_id}" if result.success else result.error,
        "chunks": result.chunks
    }