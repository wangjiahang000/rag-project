from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import upload_router, arxiv_router, chat_router

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, tags=["Upload"])
app.include_router(arxiv_router, prefix="/arxiv", tags=["arXiv"])
app.include_router(chat_router, tags=["Chat"])

@app.get("/health")
async def health():
    return {"status": "ok"}