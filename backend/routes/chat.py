from fastapi import APIRouter, Depends
from core import RAGSystem
from backend.dependencies import get_rag
from backend.models import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, rag: RAGSystem = Depends(get_rag)):
    answer = rag.chat(request.question, request.history)
    return ChatResponse(answer=answer)