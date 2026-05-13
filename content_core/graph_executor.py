import logging
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表，管理所有可执行工具的统一调度"""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self._tools[name] = func

    def run(self, op: str, **kwargs) -> Any:
        func = self._tools.get(op)
        if func is None:
            raise ValueError(f"未知工具: {op}，请先 register")
        return func(**kwargs)


class GraphExecutor:
    """DAG 执行引擎，按节点拓扑顺序依次执行"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, graph) -> Dict[str, Any]:
        results = {}
        for node in graph.nodes:
            args = {}
            for k, v in node.args.items():
                if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                    dep_id = v.strip("{}")
                    args[k] = results.get(dep_id, "")
                else:
                    args[k] = v
            try:
                results[node.id] = self.registry.run(node.op, **args)
            except Exception as e:
                logger.error("节点 %s(%s) 执行失败: %s", node.id, node.op, e)
                results[node.id] = f"[执行错误] {e}"
        return results
