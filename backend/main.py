import os
import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from backend.routers.chat import router as chat_router
from backend.routers.health import router as health_router
from backend.routers.profile import router as profile_router

logger = logging.getLogger(__name__)

app = FastAPI(title="MyRAG API", version="2.0")

# 前后端分离：允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静态文件（前端） ──
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_frontend_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dir, "assets")), name="assets")
    logger.info("前端静态资源已挂载: %s", _frontend_dir)


# ── 监控中间件 ──

class MetricsMiddleware(BaseHTTPMiddleware):
    """请求延迟、状态码、路径分布打点"""

    async def dispatch(self, request: Request, call_next):
        from backend.metrics import metrics as m

        start = time.time()
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000

        path = request.url.path
        method = request.method
        status = response.status_code

        m.inc("http_requests_total", {"method": method, "path": path, "status": str(status)})
        m.record("http_request_duration_ms", elapsed_ms, {"path": path})

        # 慢查询警告
        if elapsed_ms > 3000:
            logger.warning("[SLOW] %s %s → %d (%.0fms)", method, path, status, elapsed_ms)
        elif elapsed_ms > 500:
            logger.info("[METRIC] %s %s → %d (%.0fms)", method, path, status, elapsed_ms)

        # 响应头带耗时
        response.headers["X-Process-Time-Ms"] = str(int(elapsed_ms))
        return response


app.add_middleware(MetricsMiddleware)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(profile_router)


# ── /metrics 端点 ──

@app.get("/metrics")
def get_metrics():
    from backend.metrics import metrics as m
    return m.snapshot()


@app.post("/metrics/reset")
def reset_metrics():
    from backend.metrics import metrics as m
    m.reset()
    return {"status": "ok"}


# ── SPA 兜底路由（前端 history 路由支持） ──
_index_path = os.path.join(_frontend_dir, "index.html") if os.path.isdir(_frontend_dir) else None


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if _index_path and not full_path.startswith(("api/", "openapi", "docs", "redoc")):
        return FileResponse(_index_path)
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": "Not Found"}, status_code=404)

