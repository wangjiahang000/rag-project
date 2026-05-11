import re
import json
import os
from typing import List, Dict
from litellm import completion
from content_core.models.sbert_classifier import SBERTClassifier


class TaskRouter:
    """三级意图路由：规则 → SBERT → LLM"""

    # 关键词规则
    RULES = [
        (r"对比|区别|比较|vs|不同|差异|不一样", "compare"),
        (r"总结|汇总|概括|归纳|概述|理一理", "summarize"),
        (r"找|检索|搜索|查|查找|什么是|定义", "retrieve"),
        (r"怎么|如何|步骤|教程|配置|部署|实现", "howto"),
        (r"为什么|原因|导致|背后|原理|造成", "reason"),
        (r"提取|抽取|获取|参数|指标|数值|多少", "extract"),
    ]
    CHITCHAT_PATTERNS = [
        r"你是谁",
        r"你叫什么",
        r"你是什么",
        r"你是.{0,3}吗",
        r"我是.{0,3}吗",
        r"你好|嗨|hello|hi",
        r"再见|拜拜|bye",
        r"谢谢|多谢|感谢",
        r"聊天|闲聊",
    ]
    # 资源类型规则
    RESOURCE_RULES = [
        (r"论文|文献|paper|article", "paper"),
        (r"代码|github|实现|编程|code", "code"),
        (r"知识图谱|图数据库|graph|neo4j", "kg"),
    ]

    def __init__(self):
        self.sbert = SBERTClassifier()
    def _is_chitchat(self, query: str) -> bool:
        import re
        for p in self.CHITCHAT_PATTERNS:
            if re.search(p, query):
                return True
        return False
    
    # ── 第一层：规则 ─────────────────────
    def _rule_match(self, query: str) -> List[str]:
        tasks = []
        for pattern, task in self.RULES:
            if re.search(pattern, query):
                tasks.append(task)
        seen = set()
        result = []
        for t in tasks:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def _resource_hint(self, query: str) -> str:
        for pattern, hint in self.RESOURCE_RULES:
            if re.search(pattern, query):
                return hint
        return "doc"

    def _extract_entities(self, query: str) -> List[str]:
        english = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        chinese = re.findall(r'(?:找|和|与|跟|对比|比较)\s*([\u4e00-\u9fa5]{2,3})', query)
        stop_words = {'一下', '怎么', '什么', '这个', '那个', '为啥', '区别', '不一样', '对比下'}
        chinese = [w for w in chinese if w not in stop_words]
        return list(set(english + chinese))

    # ── 第二层：SBERT ────────────────────
    def _sbert_match(self, query: str) -> Dict[str, float]:
        tasks, confs = self.sbert.classify(query)  # 不再传 threshold
        return dict(zip(tasks, confs))

    # ── 第三层：LLM 兜底 ─────────────────
    def _llm_fallback(self, query: str, rule_tasks: list, sbert_scores: dict) -> List[str]:
        prompt = f"""你是意图分类专家。判断用户问题意图，可多选。

可选意图：retrieve, compare, summarize, howto, reason, extract

用户问题：{query}
关键词命中：{rule_tasks or "无"}
语义模型：{sbert_scores or "无"}

返回纯JSON：{{"tasks": ["xxx"], "confidence": 0.9}}"""

        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL")
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("tasks", ["retrieve"])

    # ── 主入口 ───────────────────────────
    def route(self, query: str) -> dict:
        # 并行获取规则和语义结果
        if self._is_chitchat(query):
            return {
               "user_tasks": ["chitchat"],
                "resource_hint": "none",
                "complexity": "single_step",
                "entities": [],
                "source": "rule",
            }
        rule_tasks = self._rule_match(query)
        sbert_scores = self._sbert_match(query)

        # 合并分数
        merged = {}
        for t in rule_tasks:
            merged[t] = 0.95
        for t, c in sbert_scores.items():
            if c > 0.55 or t in rule_tasks:    # 高分或规则已命中才保留
                merged[t] = max(merged.get(t, 0), c)

        sorted_pairs = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        tasks = [t for t, _ in sorted_pairs][:3]
        top_conf = sorted_pairs[0][1] if sorted_pairs else 0

        # 决定来源和是否触发 LLM
        if rule_tasks and sbert_scores:
            source = "rule+sbert"
        elif rule_tasks:
            source = "rule"
        elif sbert_scores:
            source = "sbert"
        else:
            source = "none"

        if not tasks or top_conf < 0.50:
            tasks = self._llm_fallback(query, rule_tasks, sbert_scores)
            source = "llm"

        return {
            "user_tasks": tasks,
            "resource_hint": self._resource_hint(query),
            "complexity": "multi_step" if len(tasks) > 1 else "single_step",
            "entities": self._extract_entities(query),
            "source": source,
        }