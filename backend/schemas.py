from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    session_id: str = "anonymous"


class CitationInfo(BaseModel):
    index: int
    source: str
    title: str = ""
    year: Optional[int] = None
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    user_tasks: List[str]
    plan: list
    answer: str
    citations: List[CitationInfo] = []
    source: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0"
    onnx_mode: bool = False
