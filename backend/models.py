from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class UploadResponse(BaseModel):
    status: str
    chunks: int
    message: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

class PaperResponse(BaseModel):
    id: str
    title: str
    authors: List[str]
    authors_display: str
    summary: str
    published: str
    pdf_url: Optional[str] = None

class SearchResponse(BaseModel):
    status: str
    papers: List[Dict[str, Any]]
    count: int

class AddPaperRequest(BaseModel):
    paper_id: str
    paper_title: str = ""
    category: str = "manual"

class ChatRequest(BaseModel):
    question: str
    history: List[List[str]] = []

class ChatResponse(BaseModel):
    answer: str