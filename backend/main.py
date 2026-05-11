from fastapi import FastAPI
from pydantic import BaseModel
from content_core.task_router import TaskRouter
from content_core.graph_generator import GraphGenerator
from content_core.graph_executor import ToolRegistry, GraphExecutor

app = FastAPI()

registry = ToolRegistry()
executor = GraphExecutor(registry)
generator = GraphGenerator(model="deepseek/deepseek-chat")
router = TaskRouter()

class Query(BaseModel):
    question: str

class Response(BaseModel):
    user_tasks: list
    plan: list
    answer: str

@app.post("/chat", response_model=Response)
def chat(query: Query):
    tasks, _ = router.route(query.question)
    graph = generator.generate(tasks, query.question)
    results = executor.execute(graph)

    last_node = graph.nodes[-1]
    answer = results.get(last_node.id, "抱歉，无法回答")

    return Response(
        user_tasks=tasks,
        plan=[{"id": n.id, "op": n.op, "args": n.args} for n in graph.nodes],
        answer=str(answer)
    )