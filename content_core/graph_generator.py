import json
import os
import re
import time
import logging
from typing import List, Optional
from litellm import completion
from content_core.types import TaskGraph, TaskNode
from content_core.tools.process.entity_extractor import (
    extract_entities as tech_extract_entities,
    extract_time_info,
    format_time_query,
)
import content_core.config as cfg

logger = logging.getLogger(__name__)


class GraphGenerator:
    """使用预设模板将意图组合映射为可执行任务图（TaskGraph）

    设计原则：
    - 离线、零延迟：已知组合无需调用 LLM
    - 可预测、可调试：相同意图组合总是生成相同任务图
    - 易扩展：新增意图或工具链只需修改映射表或添加构建方法
    - 未定义组合：LLM 拆解为多个合法子问题，各自走模板
    """

    # 意图组合（排序后） → 构建方法名
    TEMPLATES = {
        # ── 单意图 ──
        ("retrieve",): "_simple_search",
        ("compare",): "_build_compare",
        ("summarize",): "_build_summarize",
        ("howto",): "_simple_search",
        ("reason",): "_build_reason",
        ("extract",): "_build_extract",

        # ── 双意图（retrieve 被其他意图吸收） ──
        ("retrieve", "summarize"): "_build_summarize",
        ("compare", "retrieve"): "_build_compare",
        ("extract", "retrieve"): "_build_extract",
        ("reason", "retrieve"): "_build_reason",
        ("howto", "retrieve"): "_simple_search",
        ("compare", "reason"): "_build_compare_reason",
        ("compare", "extract"): "_build_extract_compare",

        # ── 三意图 ──
        ("compare", "retrieve", "summarize"): "_build_summarize_compare",
        ("compare", "extract", "retrieve"): "_build_extract_compare",
        ("reason", "retrieve", "summarize"): "_build_reason_summarize",
        ("compare", "reason", "retrieve"): "_build_compare_reason",
    }

    DEFAULT_K = 10

    def generate(
        self,
        user_tasks: List[str],
        query: str,
        entities: Optional[List[str]] = None,
        complexity: str = "single_step",
    ) -> List[TaskGraph]:
        """主入口：返回一个或多个 TaskGraph

        - 已知模板 → 返回 [单图]
        - 未定义组合 → LLM 拆解 → 返回多图
        """
        entities = entities or []
        key = tuple(sorted(user_tasks))

        # 提取时间约束并缓存在实例变量中，供后续搜索使用
        time_entities = extract_time_info(query)
        self._time_entities = time_entities
        self._time_suffix = format_time_query(time_entities)
        if time_entities:
            logger.info("查询中包含时间约束: %s → '%s'", time_entities, self._time_suffix)

        method_name = self.TEMPLATES.get(key)
        if method_name:
            method = getattr(self, method_name)
            nodes = method(query, entities)
            return [TaskGraph(user_tasks=user_tasks, nodes=nodes)]

        # 未定义组合 → LLM 拆解
        logger.info("未定义意图组合 %s，尝试 LLM 分解", key)
        llm_graphs = self._llm_decompose(query, user_tasks)
        if llm_graphs:
            return llm_graphs

        # LLM 全部失败 → 各意图独立成单意图图
        logger.warning("LLM 分解失败，各意图独立执行")
        graphs = []
        for task in user_tasks:
            method = getattr(self, self.TEMPLATES[(task,)])
            nodes = method(query, [])
            graphs.append(TaskGraph(user_tasks=[task], nodes=nodes))
        return graphs

    # ── LLM 问题分解 ──────────────────────

    def _llm_decompose(self, query: str, user_tasks: List[str]) -> Optional[List[TaskGraph]]:
        """LLM 将 undefined 意图组合拆解为多个合法子问题"""
        valid_combos = self._format_valid_combos()

        prompt = f"""你是一个问题分解专家。用户的原始问题包含多个意图，需要拆解为独立的子问题。

可用意图标签: retrieve, compare, summarize, howto, reason, extract

合法组合（每个子问题必须是其中之一）:
{valid_combos}

原始问题: {query}
原始意图: {user_tasks}

请分解为多个独立的子问题，使每个子问题只对应一个合法组合。
不要改变意图含义，只做拆分。

返回 JSON 数组 (纯 JSON，无 markdown):
[
  {{"sub_query": "拆分后的子问题文本", "user_tasks": ["retrieve", "compare"]}},
  ...
]"""

        for attempt in range(1 + cfg.LLM_RETRY_COUNT):
            try:
                response = completion(
                    model="deepseek/deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    api_base=os.getenv("DEEPSEEK_BASE_URL"),
                    timeout=cfg.LLM_TIMEOUT,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM 返回了空白内容")
                content = content.strip()
                # 清理 Markdown 代码块标记
                for prefix in ["```json\n", "```json", "```"]:
                    if content.startswith(prefix):
                        content = content[len(prefix):]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("LLM 返回的不是数组")

                # 验证每个子问题
                valid_items = []
                seen_combos = set()
                for item in data:
                    sub_query = (item.get("sub_query") or "").strip()
                    sub_tasks = item.get("user_tasks") or []
                    key = tuple(sorted(sub_tasks))
                    if key in self.TEMPLATES and sub_query and key not in seen_combos:
                        # 跳过与原始输入完全相同的拆解（没实际分解）
                        if set(sub_tasks) == set(user_tasks):
                            continue
                        valid_items.append((sub_query, sub_tasks))
                        seen_combos.add(key)

                if valid_items:
                    graphs = []
                    for sub_query, sub_tasks in valid_items:
                        sub_graphs = self.generate(sub_tasks, sub_query)
                        graphs.extend(sub_graphs)
                    return graphs

                logger.warning("LLM 分解结果无合法子问题: %s", content[:200])

            except Exception as e:
                logger.warning("问题分解第 %d/%d 次失败: %s",
                               attempt + 1, cfg.LLM_RETRY_COUNT + 1, e)
                if attempt < cfg.LLM_RETRY_COUNT:
                    time.sleep(cfg.LLM_RETRY_DELAY * (attempt + 1))

        logger.error("问题分解全部 %d 次尝试均失败", cfg.LLM_RETRY_COUNT + 1)
        return None

    @classmethod
    def _format_valid_combos(cls) -> str:
        """从 TEMPLATES 生成可读的合法组合列表"""
        seen = set()
        lines = []
        for key in cls.TEMPLATES:
            sorted_key = tuple(sorted(key))
            if sorted_key in seen:
                continue
            seen.add(sorted_key)
            if len(key) == 1:
                lines.append(f"  - {key[0]}")
            else:
                lines.append(f"  - {'+'.join(sorted(key))}")
        # 按长度再按字母排序
        lines.sort(key=lambda x: (len(x), x))
        return "\n".join(lines)

    # ── 基础工具链模板 ────────────────────

    def _simple_search(self, query: str, k: Optional[int] = None) -> List[TaskNode]:
        """hybrid_search → rerank"""
        k = k or self.DEFAULT_K
        ts = getattr(self, '_time_suffix', '')
        search_query = f"{query} {ts}".strip() if ts else query
        return [
            TaskNode(id="1", op="hybrid_search", args={"query": search_query, "k": k}),
            TaskNode(id="2", op="rerank", args={"docs": "{{1}}", "query": query}, depends_on=["1"]),
        ]

    def _search_rerank_then(
        self,
        tool: str,
        query: str,
        k: Optional[int] = None,
        extra_args: Optional[dict] = None,
    ) -> List[TaskNode]:
        """hybrid_search → rerank → {tool}"""
        k = k or self.DEFAULT_K
        extra = extra_args or {}
        nodes = self._simple_search(query, k)
        last_id = nodes[-1].id
        new_id = str(int(last_id) + 1)
        tool_node = TaskNode(
            id=new_id,
            op=tool,
            args={"docs": "{{%s}}" % last_id, **extra},
            depends_on=[last_id],
        )
        return nodes + [tool_node]

    @staticmethod
    def _is_person_name(word: str) -> bool:
        """首字母大写纯字母 → 猜测为人名"""
        return bool(re.match(r'^[A-Z][a-z]+$', word))

    @staticmethod
    def _get_extractor():
        from content_core.tools.process.entity_extractor import get_extractor
        return get_extractor()

    def _extract_context(self, query: str, exclude: List[str]) -> str:
        """提取 UNKNOWN 实体作为领域上下文，排除搜索词自身"""
        ext = self._get_extractor()
        entities = ext.extract(query)
        exclude_lower = {x.lower() for x in exclude}
        context = []
        for e in entities:
            if e["type"] not in ("TECH", "TIME") and e["text"].lower() not in exclude_lower:
                context.append(e["text"])
        return " ".join(context[:2])

    def _search_query(self, entity: str, context: str = "") -> str:
        """按实体类型构造搜索词"""
        if entity.lower() in self._get_extractor().tech_dict:
            suffix = f" {context}".rstrip()
            return f"{entity}{suffix}"
        if self._is_person_name(entity):
            return f"{entity} 论文"
        return entity

    def _build_entity_node(self, node_id: str, entity: str, context: str, k: int) -> TaskNode:
        """为单个实体构建搜索节点：人名优先走索引，否则 hybrid_search"""
        if self._is_person_name(entity):
            from data.metadata_index import person_search
            time_entities = getattr(self, '_time_entities', [])
            result = person_search(entity, time_entities)
            if result is not None:
                # 人名索引命中 → 根据时间约束做不同查找
                logger.info("人名索引命中: %s", entity)
                # 暂不实现具体逻辑，等人名索引就绪后按 time_relation 分支
                # 例：
                #   year_range  → 过滤 start_year~end_year 的论文
                #   after_year  → 查找 year 之后的论文
                #   before_year → 查找 year 之前的论文
                #   recent_years→ 查找最近 years_back 年的论文

        query = self._search_query(entity, context)
        ts = getattr(self, '_time_suffix', '')
        search_query = f"{query} {ts}".strip() if ts else query
        return TaskNode(id=node_id, op="hybrid_search", args={"query": search_query, "k": k})

    def _compare_chain(self, query: str, entities: List[str], k: Optional[int] = None) -> List[TaskNode]:
        """hybrid_search ×2（按实体拆分） → rerank ×2 → compare"""
        k = k or self.DEFAULT_K
        e0 = entities[0] if len(entities) > 0 else ""
        e1 = entities[1] if len(entities) > 1 else ""
        ctx = self._extract_context(query, entities)
        return [
            self._build_entity_node("1", e0, ctx, k),
            self._build_entity_node("2", e1, ctx, k),
            TaskNode(id="3", op="rerank", args={"docs": "{{1}}", "query": query}, depends_on=["1"]),
            TaskNode(id="4", op="rerank", args={"docs": "{{2}}", "query": query}, depends_on=["2"]),
            TaskNode(id="5", op="compare", args={"docs_a": "{{3}}", "docs_b": "{{4}}"}, depends_on=["3", "4"]),
        ]

    def _fallback(self, query: str, k: Optional[int] = None) -> List[TaskNode]:
        """未定义组合 / 降级策略 → hybrid_search → rerank"""
        return self._simple_search(query, k)

    # ── 单意图构建器 ──────────────────────

    def _ensure_entities(self, query: str, entities: List[str]) -> List[str]:
        """确保至少有 2 个实体，不足时用 BERT 抽取"""
        if len(entities) >= 2:
            return entities
        extracted = tech_extract_entities(query)
        if len(extracted) >= 2:
            logger.info("BERT 从查询中抽取到实体: %s", extracted[:4])
            return extracted
        return entities

    def _build_compare(self, query: str, entities: List[str]) -> List[TaskNode]:
        entities = self._ensure_entities(query, entities)
        if len(entities) >= 2:
            return self._compare_chain(query, entities)
        logger.warning("compare 实体不足（%d 个），降级为简单检索", len(entities))
        return self._fallback(query)

    def _build_summarize(self, query: str, entities: List[str]) -> List[TaskNode]:
        return self._search_rerank_then("summarize", query)

    def _build_reason(self, query: str, entities: List[str]) -> List[TaskNode]:
        return self._search_rerank_then("reason", query)

    def _build_extract(self, query: str, entities: List[str]) -> List[TaskNode]:
        return self._search_rerank_then("extract", query, extra_args={"target": query})

    # ── 双意图构建器 ──────────────────────

    def _build_compare_reason(self, query: str, entities: List[str]) -> List[TaskNode]:
        """compare 链路 → reason（先对比，再分析原因）"""
        entities = self._ensure_entities(query, entities)
        if len(entities) < 2:
            logger.warning("compare+reason 实体不足，降级")
            return self._fallback(query)
        nodes = self._compare_chain(query, entities)
        last_id = nodes[-1].id
        new_id = str(int(last_id) + 1)
        nodes.append(TaskNode(
            id=new_id, op="reason",
            args={"docs": "{{%s}}" % last_id},
            depends_on=[last_id],
        ))
        return nodes

    def _build_extract_compare(self, query: str, entities: List[str]) -> List[TaskNode]:
        """hybrid_search(e0) → rerank → extract(e0)
           hybrid_search(e1) → rerank → extract(e1)
           compare(extract(e0), extract(e1))
        """
        entities = self._ensure_entities(query, entities)
        if len(entities) < 2:
            logger.warning("extract+compare 实体不足（%d 个），降级", len(entities))
            return self._fallback(query)
        e0, e1 = entities[0], entities[1]
        k = self.DEFAULT_K
        ctx = self._extract_context(query, entities)
        return [
            self._build_entity_node("1", e0, ctx, k),
            self._build_entity_node("2", e1, ctx, k),
            TaskNode(id="3", op="rerank", args={"docs": "{{1}}", "query": query}, depends_on=["1"]),
            TaskNode(id="4", op="rerank", args={"docs": "{{2}}", "query": query}, depends_on=["2"]),
            TaskNode(id="5", op="extract", args={"docs": "{{3}}", "target": e0}, depends_on=["3"]),
            TaskNode(id="6", op="extract", args={"docs": "{{4}}", "target": e1}, depends_on=["4"]),
            TaskNode(id="7", op="compare", args={"docs_a": "{{5}}", "docs_b": "{{6}}", "query": query}, depends_on=["5", "6"]),
        ]

    # ── 三意图构建器 ──────────────────────

    def _build_summarize_compare(self, query: str, entities: List[str]) -> List[TaskNode]:
        """summarize 链路 → compare（先归纳，再对比归纳结果）"""
        entities = self._ensure_entities(query, entities)
        if len(entities) < 2:
            logger.warning("summarize+compare 实体不足，降级")
            return self._fallback(query)
        nodes = self._build_summarize(query, entities)
        last_id = nodes[-1].id
        new_id = str(int(last_id) + 1)
        nodes.append(TaskNode(
            id=new_id, op="compare",
            args={"docs_a": "{{%s}}" % last_id, "docs_b": "{{%s}}" % last_id,
                  "query": query},
            depends_on=[last_id],
        ))
        return nodes

    def _build_reason_summarize(self, query: str, entities: List[str]) -> List[TaskNode]:
        """reason 链路 → summarize（先分析原因，再总结结论）"""
        nodes = self._build_reason(query, entities)
        last_id = nodes[-1].id
        new_id = str(int(last_id) + 1)
        nodes.append(TaskNode(
            id=new_id, op="summarize",
            args={"docs": "{{%s}}" % last_id},
            depends_on=[last_id],
        ))
        return nodes
