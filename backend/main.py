from fastapi import FastAPI
from pydantic import BaseModel

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

app = FastAPI()

# ── 持久化向量存储 ──
vector_store = VectorStore(chroma_path="./chroma_data")
set_vector_store(vector_store)

# ── 注册工具 ──
registry = ToolRegistry()
registry.register("hybrid_search", hybrid_search)
registry.register("rerank", rerank)
registry.register("compare", compare)
registry.register("summarize", summarize)
registry.register("extract", extract)
registry.register("reason", reason)

executor = GraphExecutor(registry)
generator = GraphGenerator()
router = TaskRouter()


class Query(BaseModel):
    question: str


class Response(BaseModel):
    user_tasks: list
    plan: list
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=Response)
def chat(query: Query):
    result = router.route(query.question)
    tasks = result["user_tasks"]

    graphs = generator.generate(tasks, query.question)
    answers = []
    all_plan = []
    for graph in graphs:
        results = executor.execute(graph)
        last_id = graph.nodes[-1].id
        answers.append(str(results.get(last_id, "")))
        all_plan.extend(
            {"id": n.id, "op": n.op, "args": n.args} for n in graph.nodes
        )
    answer = "\n\n---\n\n".join(answers)

    return Response(
        user_tasks=tasks,
        plan=all_plan,
        answer=answer,
    )
