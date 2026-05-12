import json
import os
from typing import List
from litellm import completion
from content_core.types import TaskGraph, TaskNode


class GraphGenerator:
    """使用 LLM 动态生成任务图（TaskGraph）"""

    def __init__(self, tool_registry=None):
        """
        tool_registry: ToolRegistry 实例，用于获取工具描述
        """
        self.tool_registry = tool_registry

    def generate(
        self,
        user_tasks: List[str],
        query: str,
        resource_hint: str = "doc",
        entities: List[str] = None,
        complexity: str = "single_step"
    ) -> TaskGraph:
        # 获取工具描述
        tools_desc = self.tool_registry.get_descriptions() if self.tool_registry else ""
        entities_str = ", ".join(entities) if entities else "无"

        prompt = f"""你是任务规划器。根据用户意图和可用工具，生成一个执行计划（TaskGraph）。

用户意图标签：{user_tasks}
资源类型提示：{resource_hint}
已识别实体：{entities_str}
问题复杂度：{complexity}

可用工具：
{tools_desc}

用户原始问题：{query}

请生成一个 JSON 格式的执行计划，结构如下：
{{
  "user_tasks": {json.dumps(user_tasks)},
  "nodes": [
    {{
      "id": "1",
      "op": "工具名",
      "args": {{"参数名": "参数值"}},
      "depends_on": []          // 依赖的前置节点 id 列表，无依赖则空
    }}
  ]
}}

规则：
1. 节点 ID 从 "1" 开始递增。
2. 如果有 retrieve 意图，应先调用 hybrid_search 检索文档，然后通常跟 rerank 重排序。
3. 如果有 compare 意图，应分别检索两个对象（如果只有模糊对象，可进行一次检索后提取再对比），最后调用 compare 工具。
4. 如果有 summarize 意图，检索后调用 summarize 工具（如果未实现 summarize，暂时可省略或使用 extract 替代，但尽量避免）。
5. 如果有 reason 意图，可能需要多跳检索，先查原因再查机制。
6. 如果有 extract 意图，检索后调用 extract 工具（暂未实现，可用检索结果直接回答，但不要添加到节点中）。
7. 如果只有 howto 意图，直接检索相关教程。
8. 节点参数中可以用占位符 `"{{{{1}}}}"` 引用前面节点的输出（例如 "{{{{1}}}}" 表示节点1的整个结果）。args 中的具体参数值根据上下文填充，不要使用占位符表示具体查询词，而应直接写出查询词。
9. 如果 complexity 为 "multi_step"，可生成多个节点，否则尽量保持简单（1-3个节点）。
10. 只返回纯 JSON，不要包含任何 Markdown 标记或解释。

生成的 JSON："""

        # 调用 LLM
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL"),
        )
        content = response.choices[0].message.content.strip()

        # 清理可能的 Markdown 代码块标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，返回一个最简单的 fallback 计划：检索+直接返回
            fallback_nodes = [
                TaskNode(id="1", op="hybrid_search", args={"query": query, "k": 5})
            ]
            # 如果有 compare 且实体够，尝试拆分
            if "compare" in user_tasks and entities and len(entities) >= 2:
                fallback_nodes = [
                    TaskNode(id="1", op="hybrid_search", args={"query": f"{entities[0]} {query}", "k": 5}),
                    TaskNode(id="2", op="hybrid_search", args={"query": f"{entities[1]} {query}", "k": 5}),
                    TaskNode(id="3", op="rerank", args={"docs": "{{1}}", "query": query}, depends_on=["1"]),
                    TaskNode(id="4", op="rerank", args={"docs": "{{2}}", "query": query}, depends_on=["2"]),
                    TaskNode(id="5", op="compare", args={"docs_a": "{{3}}", "docs_b": "{{4}}"},
                             depends_on=["3", "4"]),
                ]
            return TaskGraph(user_tasks=user_tasks, nodes=fallback_nodes)

        # 将 dict 转为 TaskNode 列表
        nodes = [TaskNode(**n) for n in data.get("nodes", [])]
        return TaskGraph(user_tasks=data.get("user_tasks", user_tasks), nodes=nodes)