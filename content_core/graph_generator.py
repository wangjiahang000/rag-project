import json
import os
from dotenv import load_dotenv
from litellm import completion
from content_core.types import TaskGraph, TaskNode

load_dotenv()

class GraphGenerator:
    def __init__(self, model: str = "deepseek/deepseek-chat"):
        self.model = model

    def generate(self, user_tasks: list, query: str) -> TaskGraph:
        prompt = f"""
你是任务规划器。用户任务标签：{user_tasks}。

可用工具：
- hybrid_search: 混合检索，参数 query(必填), k(默认5), vec_weight(默认0.7), bm25_weight(默认0.3)
- rerank: 重排序，参数 docs(文档列表), query(原始问题)
- compare: 对比两组文档，参数 docs_a, docs_b
- summarize: 汇总文档，参数 docs
- extract: 抽取值，参数 docs, target(要抽取的内容)

用户问题：{query}

返回纯 JSON（不要 ``` 标记）：
{{
  "user_tasks": {user_tasks},
  "nodes": [
    {{"id":"1","op":"工具名","args":{{"参数":"值"}},"depends_on":[]}}
  ]
}}

规则：
- 对比类任务必须分别检索两个对象，最后调用 compare
- 检索后必须接 rerank
- 节点 id 从 1 递增
"""
        response = completion(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL")
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        nodes = [TaskNode(**n) for n in data["nodes"]]
        return TaskGraph(user_tasks=data["user_tasks"], nodes=nodes)