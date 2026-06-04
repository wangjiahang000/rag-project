"""FastAPI 依赖注入 —— 单例管理 RAG 核心组件"""

from functools import lru_cache

from content_core.task_router import TaskRouter
from content_core.graph_generator import GraphGenerator
from content_core.graph_executor import ToolRegistry, GraphExecutor
from content_core.data.vector_store import VectorStore
from content_core.tools.search.hybrid_search import hybrid_search, set_vector_store
from content_core.tools.process.rerank import rerank
from content_core.tools.process.compare import compare
from content_core.tools.process.summarize import summarize
from content_core.tools.process.extract import extract
from content_core.tools.process.reason import reason


@lru_cache
def get_vector_store() -> VectorStore:
    vs = VectorStore(persist_dir="./chroma_data")
    set_vector_store(vs)
    return vs


@lru_cache
def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("hybrid_search", hybrid_search)
    registry.register("rerank", rerank)
    registry.register("compare", compare)
    registry.register("summarize", summarize)
    registry.register("extract", extract)
    registry.register("reason", reason)
    return registry


@lru_cache
def get_router() -> TaskRouter:
    return TaskRouter()


@lru_cache
def get_generator() -> GraphGenerator:
    return GraphGenerator()


@lru_cache
def get_executor() -> GraphExecutor:
    return GraphExecutor(get_tool_registry())
