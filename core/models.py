from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PaperInfo:
    arxiv_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    category: str = ""
    summary: str = ""
    pdf_url: Optional[str] = None

@dataclass
class ImportResult:
    arxiv_id: str
    title: str
    success: bool
    chunks: int
    error: Optional[str] = None