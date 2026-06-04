# content_core/task_router.py
import re
import json
import os
import numpy as np
from typing import List, Dict
from litellm import completion
from content_core.models.sbert_classifier import SBERTClassifier
import content_core.config as cfg


class TaskRouter:
    """三级意图路由（规则提供信号，SBERT语义裁决，LLM兜底）"""

    # ── 关键词规则 ──────────────────────
    RULES = [
        (r"对比|区别|比较|vs|不同|差异|不一样|更|哪个", "compare"),
        (r"总结|汇总|概括|归纳|概述|理一理", "summarize"),
        # extract 排在 retrieve 之前，避免"找出/查出"被 retrieve 的"找"截胡
        (r"提取|抽取|获取|找出|查出|参数|指标|数值|多少", "extract"),
        (r"找|检索|搜|搜索|查|查找|什么是|是什么|定义|介绍", "retrieve"),
        (r"怎么|如何|步骤|教程|配置|部署|实现|导出|转换", "howto"),
        (r"为什么|原因|导致|背后|原理|造成", "reason"),
    ]

    def __init__(self):
        self.sbert = SBERTClassifier()

    # ── 自适应幂次缩放（拉大分数差距） ──
    @staticmethod
    def _adaptive_power_scale(scores_dict: Dict[str, float], power: float = 2.0) -> Dict[str, float]:
        scores = np.array(list(scores_dict.values()))
        smin, smax = scores.min(), scores.max()
        if smax - smin < 1e-10:
            return dict(scores_dict)
        # 归一化 → 幂次缩放 → 映射回 [smin, 1.0]（top 意图 = 1.0，拉开分差）
        norm = (scores - smin) / (smax - smin)
        scaled = norm ** power
        target_range = 1.0 - smin
        return dict(zip(scores_dict.keys(), smin + scaled * target_range))

    # ── 闲聊检测 ────────────────────────
    def _is_chitchat(self, query: str) -> bool:
        for p in cfg.CHITCHAT_PATTERNS:
            if re.search(p, query):
                return True
        return False

    # ── 第一层：关键词信号 ──────────────
    def _rule_match(self, query: str) -> List[str]:
        tasks = []
        for pattern, task in self.RULES:
            if re.search(pattern, query):
                tasks.append(task)
        # 去重保序
        seen = set()
        result = []
        for t in tasks:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    # ── 资源类型识别 ────────────────────
    def _resource_hint(self, query: str) -> str:
        for pattern, hint in cfg.RESOURCE_RULES:
            if re.search(pattern, query):
                return hint
        return "doc"

    # ── 简单实体抽取 ────────────────────
    def _extract_entities(self, query: str) -> List[str]:
        english = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        chinese = re.findall(r'(?:找|和|与|跟|对比|比较)\s*([\u4e00-\u9fa5]{2,3})', query)
        stop_words = {'一下', '怎么', '什么', '这个', '那个', '为啥', '区别',
                      '不一样', '对比下', '几篇', '一篇', '关于', '一些'}
        chinese = [w for w in chinese if w not in stop_words]
        return list(set(english + chinese))

    # ── 第二层：SBERT语义分数 ──────────
    def _sbert_match(self, query: str) -> Dict[str, float]:
        tasks, confs = self.sbert.classify(query)
        return dict(zip(tasks, confs))

    # ── 第三层：LLM兜底 ─────────────────
    def _llm_fallback(self, query: str, rule_signals: list, sbert_scores: dict) -> List[str]:
        prompt = f"""你是学术文献检索意图分类专家。首先判断用户输入是否属于闲聊、问候、无意义调侃或与学术技术完全无关的内容。
如果是，请直接返回：{{"tasks": ["chitchat"], "confidence": 1.0}}
如果不是闲聊，则判断用户问题的真实意图，可多选
可选意图：retrieve, compare, summarize, howto, reason, extract

用户问题：{query}
关键词命中：{rule_signals or "无"}
语义模型分数：{sbert_scores or "无"}

返回纯JSON：{{"tasks": ["xxx"], "confidence": 0.9}}"""

        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL"),
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("tasks", ["retrieve"])

    # ── 主入口 ──────────────────────────
    def route(self, query: str) -> dict:
        # 闲聊直接拦截
        if self._is_chitchat(query):
            return {
                "user_tasks": ["chitchat"],
                "resource_hint": "none",
                "complexity": "single_step",
                "entities": [],
                "source": "rule",
            }

        # 1. 收集关键词信号（只作为信号，不直接决策）
        rule_signals = self._rule_match(query)

        # 2. 获取 SBERT 完整语义分数
        sbert_scores = self._sbert_match(query)

        # 3. 自适应幂次缩放（在原始区间内拉大差距）
        scaled_scores = self._adaptive_power_scale(sbert_scores, power=2.0)

        # 4. 融合决策（关键词平滑提权）
        merged = {}
        for intent, score in scaled_scores.items():
            merged[intent] = score

        for intent in rule_signals:
            sbert_score = sbert_scores.get(intent, 0)
            if sbert_score > cfg.BOOST_RANGE[0]:
                t = min((sbert_score - cfg.BOOST_RANGE[0]) / (cfg.BOOST_RANGE[1] - cfg.BOOST_RANGE[0]), 1.0)
                boosted = cfg.BOOST_TARGET[0] + (cfg.BOOST_TARGET[1] - cfg.BOOST_TARGET[0]) * t
                merged[intent] = max(merged.get(intent, 0), boosted)

        sorted_pairs = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        tasks = [t for t, s in sorted_pairs if s >= cfg.SCORE_THRESHOLDS.get(t, 0.35)][:3]
        # 使用原始 SBERT 最高分判断置信度（幂次缩放后 top 恒为 ~1.0）
        orig_sorted = sorted(sbert_scores.items(), key=lambda x: x[1], reverse=True)
        top_conf = orig_sorted[0][1] if orig_sorted else 0.0

        # ── 硬逻辑解耦规则 ──
        # 规则1：如果 howto 和 summarize 同时出现，且查询中无明确的步骤/教程关键词，则移除 howto
        if "howto" in tasks and "summarize" in tasks:
            # 检查是否包含步骤/教程类关键词
            tutorial_keywords = r"步骤|步|代码|怎么|如何|教程|配置|部署|实现|流程"
            if not re.search(tutorial_keywords, query):
                tasks.remove("howto")
                # 如果移除后列表为空，补回一个最高分的
                if not tasks:
                    tasks = [t for t, s in sorted_pairs if s >= thresholds.get(t, 0.35) and t != "howto"][:1]

        # 规则2：如果 extract 和 retrieve 同时出现，且 extract 无关键词信号，且查询中无具体数值量词，则移除 extract
        if "extract" in tasks and "retrieve" in tasks and "extract" not in rule_signals:
            # 检查是否包含数值/量词类关键词
            quantity_keywords = r"多少|参数量|token|准确率|数据量|参数|指标|数值|得分|成本|多少钱|用了多少|训练数据"
            if not re.search(quantity_keywords, query):
                tasks.remove("extract")
                if not tasks:
                    tasks = [t for t, s in sorted_pairs if s >= thresholds.get(t, 0.35) and t != "extract"][:1]

        # 规则3：纯介绍类 query（"介绍一下transformer"）排除误触的 summarize
        # 但如果 query 本身含有归纳意图（综合/几篇/整体等），保留 summarize
        if "summarize" in tasks and "retrieve" in tasks:
            intro_pattern = r"介绍|什么是|是什么|啥是"
            summarize_hint = r"综合|几篇|几份|多个|所有|整体|各种|整理|归纳|总结|概括"
            if (re.search(intro_pattern, query)
                    and not re.search(summarize_hint, query)
                    and merged.get("retrieve", 0) > merged.get("summarize", 0)):
                tasks.remove("summarize")

        # 最终 tasks 去重，保序，最多3个
        seen = set()
        final_tasks = []
        for t in tasks:
            if t not in seen:
                seen.add(t)
                final_tasks.append(t)
        tasks = final_tasks[:3]

        # 规则4：关键词匹配的意图优先于其语义混淆意图
        for intent, competitors in cfg.KEYWORD_PRIORITY.items():
            if intent in rule_signals and intent in tasks:
                for comp in competitors:
                    if comp in tasks and comp not in rule_signals:
                        i_idx = tasks.index(intent)
                        c_idx = tasks.index(comp)
                        if c_idx < i_idx:
                            tasks[c_idx], tasks[i_idx] = tasks[i_idx], tasks[c_idx]

        # 规则5：分差过滤 — 非关键词意图远低于 #1 时移除
        # 幂次缩放后 top 意图分数接近 1.0，非关键词的低分噪音意图可安全移除
        if len(tasks) >= 2:
            top_score = merged[tasks[0]]
            kept = [tasks[0]]  # 始终保留 #1
            for t in tasks[1:]:
                if t in rule_signals:
                    kept.append(t)  # 关键词意图不截断
                else:
                    relative_margin = (top_score - merged[t]) / top_score
                    if relative_margin <= 0.05:  # 差距不大时保留
                        kept.append(t)
            tasks = kept[:3]
        
        # 决定来源
        if rule_signals and sbert_scores:
            source = "rule+sbert"
        elif rule_signals:
            source = "rule"
        elif sbert_scores:
            source = "sbert"
        else:
            source = "none"

        # 4. 关键词矛盾检测：规则命中但最终没进 tasks → 规则层和语义层打架
        keyword_conflict = [s for s in rule_signals if s not in tasks]

        # 5. 如果最终无任务、最高置信度太低、或关键词矛盾，调用 LLM 兜底
        if not tasks or top_conf < 0.50 or keyword_conflict:
            tasks = self._llm_fallback(query, rule_signals, sbert_scores)
            source = "llm"

        # 6. 约束映射：确保输出属于合法组合方案
        tasks_set = set(tasks)
        if len(tasks) == 3 and tasks_set not in cfg.VALID_TRIPLE:
            reduced = False
            for valid in cfg.VALID_DUAL:
                if valid.issubset(tasks_set):
                    tasks = [t for t in tasks if t in valid]
                    reduced = True
                    break
            if not reduced:
                tasks = [tasks[0]]
        elif len(tasks) == 2 and tasks_set not in [set(c) for c in cfg.VALID_DUAL]:
            tasks = [tasks[0]]

        return {
            "user_tasks": tasks,
            "complexity": "multi_step" if len(tasks) > 1 else "single_step",
            "entities": [],
            "source": source,
        }