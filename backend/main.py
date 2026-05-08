from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.routes import upload_router, arxiv_router, chat_router
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error("=" * 60)
    logger.error("请求验证失败! 错误详情: %s", exc.errors())
    logger.error("原始请求体: %s", body.decode("utf-8", errors="replace"))
    logger.error("=" * 60)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

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

# 托管 React 前端静态文件
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend_react", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    logger.info("React 前端已挂载: %s", frontend_dist)