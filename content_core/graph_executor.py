import asyncio
import logging
from typing import Any, Dict, Callable, List

import content_core.config as cfg

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

    async def run_async(self, op: str, **kwargs) -> Any:
        """异步执行工具调用，包装同步函数到线程池"""
        func = self._tools.get(op)
        if func is None:
            raise ValueError(f"未知工具: {op}，请先 register")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))


class GraphExecutor:
    """异步 DAG 执行引擎 — 无依赖节点并行执行

    调度策略：
    - 按拓扑层分组（同层节点无依赖关系，可并行）
    - 每层内的节点通过 asyncio.gather 并发执行
    - 串行节点保持拓扑顺序
    - 失败节点支持重试
    """

    def __init__(self, registry: ToolRegistry, max_workers: int = None):
        self.registry = registry
        self._semaphore = asyncio.Semaphore(max_workers or cfg.DAG_MAX_WORKERS)

    async def execute(self, graph) -> Dict[str, Any]:
        results = {}
        nodes = list(graph.nodes)

        # 按拓扑层分组
        layers = self._topological_layers(nodes)

        for layer in layers:
            # 同一层节点无依赖关系，并行执行
            tasks = {}
            async with self._semaphore:
                for node in layer:
                    args = self._resolve_args(node, results)
                    tasks[node.id] = asyncio.create_task(
                        self._run_with_retry(node, args)
                    )

                # 等待当前层所有节点完成
                for node_id, task in tasks.items():
                    try:
                        results[node_id] = await task
                    except Exception as e:
                        logger.error("节点 %s 执行失败: %s", node_id, e)
                        results[node_id] = f"[执行错误] {e}"

        return results

    async def _run_with_retry(self, node, args: dict) -> Any:
        """执行单个节点，带重试"""
        last_error = None
        for attempt in range(1 + cfg.DAG_RETRY_COUNT):
            try:
                return await self.registry.run_async(node.op, **args)
            except Exception as e:
                last_error = e
                logger.warning(
                    "节点 %s(%s) 第 %d/%d 次失败: %s",
                    node.id, node.op, attempt + 1, cfg.DAG_RETRY_COUNT + 1, e,
                )
                if attempt < cfg.DAG_RETRY_COUNT:
                    await asyncio.sleep(1.0 * (attempt + 1))
        logger.error("节点 %s(%s) 全部重试失败: %s", node.id, node.op, last_error)
        raise last_error

    def _topological_layers(self, nodes: list) -> List[list]:
        """将节点按拓扑顺序分层，同一层内可并行执行"""
        node_map = {n.id: n for n in nodes}
        in_degree = {n.id: len(n.depends_on) for n in nodes}
        depends_map = {n.id: set(n.depends_on) for n in nodes}

        layers = []
        remaining = set(node_map.keys())

        while remaining:
            current = {nid for nid in remaining if in_degree[nid] == 0}
            if not current:
                logger.warning("检测到循环依赖，强制取出: %s", remaining)
                current = {min(remaining)}

            sorted_current = sorted(current, key=lambda nid: list(node_map.keys()).index(nid))
            layers.append([node_map[nid] for nid in sorted_current])

            for nid in sorted_current:
                remaining.remove(nid)
                for other in remaining:
                    if nid in depends_map[other]:
                        in_degree[other] -= 1

        return layers

    def _resolve_args(self, node, results: dict) -> dict:
        """解析节点参数，将 {{node_id}} 替换为实际结果"""
        args = {}
        for k, v in node.args.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                dep_id = v.strip("{}")
                args[k] = results.get(dep_id, "")
            else:
                args[k] = v
        return args

    def shutdown(self):
        """兼容旧版 shutdown 调用 — 异步化后无需关闭线程池"""
        pass
