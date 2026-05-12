# tests/test_my_predictions.py
import sys
import os
import re
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_core.task_router import TaskRouter
from content_core.models.sbert_classifier import SBERTClassifier

MODEL_C = "BAAI/bge-base-zh-v1.5"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "a.txt")

INTENT_LABELS = ["retrieve", "compare", "summarize", "howto", "reason", "extract"]

# ── 关键词规则（与 task_router.py 一致）──
RULES_PATTERNS = [
    (re.compile(r"对比|区别|比较|vs|不同|差异|不一样|哪个更"), "compare"),
    (re.compile(r"总结|汇总|概括|归纳|概述|理一理"), "summarize"),
    (re.compile(r"提取|抽取|获取|找出|查出|参数|指标|数值|多少"), "extract"),
    (re.compile(r"找|检索|搜索|查|查找|什么是|是什么|定义|介绍"), "retrieve"),
    (re.compile(r"怎么|如何|步骤|教程|配置|部署|实现|导出|转换"), "howto"),
    (re.compile(r"为什么|原因|导致|背后|原理|造成"), "reason"),
]

BOOST_RANGE = (0.27, 0.50)
BOOST_TARGET = (0.30, 0.65)

THRESHOLDS = {
    "retrieve": 0.70, "compare": 0.70, "summarize": 0.70,
    "howto": 0.70, "reason": 0.70, "extract": 0.70,
}

# ── 我预测的测试集 ──
MY_PREDICTIONS = [
    # ══════════════════════════════════════════
    #  单意图（6 种 × 高频）
    # ══════════════════════════════════════════

    # ── retrieve ──
    ("找几篇LoRA相关的论文", ["retrieve"]),
    ("什么是RAG", ["retrieve"]),
    ("搜一下注意力机制的文章", ["retrieve"]),
    ("查查Transformer的原始论文", ["retrieve"]),
    ("帮我找一些关于模型压缩的资料", ["retrieve"]),
    ("搜索最新的Agent框架", ["retrieve"]),
    ("推荐几篇多模态学习的综述", ["retrieve"]),
    ("最近RLHF有什么新进展", ["retrieve"]),
    ("帮我查一下知识蒸馏的定义", ["retrieve"]),
    ("MoE架构是什么", ["retrieve"]),

    # ── compare ──
    ("LoRA和QLoRA有什么区别", ["compare"]),
    ("对比BERT和GPT的区别", ["compare"]),
    ("CNN和Transformer哪个好", ["compare"]),
    ("分析RAG和微调的不同", ["compare"]),
    ("DPO和PPO哪个更适合对齐", ["compare"]),
    ("SFT和RLHF有什么不一样", ["compare"]),
    ("LangChain和LlamaIndex哪个好用", ["compare"]),
    ("A和B比一下优劣", ["compare"]),
    ("PyTorch和TensorFlow选哪个", ["compare"]),
    ("GPT-4o和Claude谁更强", ["compare"]),

    # ── summarize ──
    ("总结知识图谱研究现状", ["summarize"]),
    ("归纳大模型训练的主要方法", ["summarize"]),
    ("概括多模态学习的关键技术", ["summarize"]),
    ("梳理LLM评估方法的发展脉络", ["summarize"]),
    ("用几句话总结这篇文章", ["summarize"]),
    ("帮我归纳这些资料的主要内容", ["summarize"]),
    ("汇总模型量化的研究成果", ["summarize"]),
    ("整理对比学习的主要方法", ["summarize"]),
    ("提炼这几篇论文的核心观点", ["summarize"]),
    ("概括一下RAG的几种实现范式", ["summarize"]),

    # ── howto ──
    ("怎么部署70B模型到生产环境", ["howto"]),
    ("如何用PyTorch实现Transformer", ["howto"]),
    ("教我怎么搭建RAG系统", ["howto"]),
    ("怎么在LangChain里接入自定义LLM", ["howto"]),
    ("用vLLM部署模型的步骤", ["howto"]),
    ("微调LLaMA需要什么配置", ["howto"]),
    ("在本地跑Qwen模型的教程", ["howto"]),
    ("HuggingFace的Trainer怎么用", ["howto"]),
    ("LoRA微调的具体步骤", ["howto"]),
    ("把模型导出ONNX格式的方法", ["howto"]),

    # ── reason ──
    ("为什么Transformer比RNN快", ["reason"]),
    ("模型过拟合的原因是什么", ["reason"]),
    ("解释Batch Normalization的原理", ["reason"]),
    ("大模型为什么会出现幻觉", ["reason"]),
    ("RLHF为什么能让模型对齐", ["reason"]),
    ("为什么说Mamba能替代Transformer", ["reason"]),
    ("多头注意力机制为什么有效", ["reason"]),
    ("为什么需要位置编码", ["reason"]),
    ("梯度消失是怎么产生的", ["reason"]),
    ("为什么深度学习需要normalization", ["reason"]),

    # ── extract ──
    ("BERT的参数量是多少", ["extract"]),
    ("从论文中提取实验参数和准确率", ["extract"]),
    ("找出这篇文章用了什么数据集", ["extract"]),
    ("文中提到的模型参数量是多少", ["extract"]),
    ("把训练超参提取出来", ["extract"]),
    ("看看测试集上的准确率", ["extract"]),
    ("文中对比了几种方法的性能数据", ["extract"]),
    ("提取实验部分的数值信息", ["extract"]),
    ("这篇论文用了哪些评估指标", ["extract"]),
    ("查一下训练数据集的规模", ["extract"]),

    # ══════════════════════════════════════════
    #  双意图（7 种 × 中高频）
    # ══════════════════════════════════════════

    # retrieve + summarize
    ("找几篇Agent论文，总结主要方法", ["retrieve", "summarize"]),
    ("搜一下模型压缩的综述并归纳方法", ["retrieve", "summarize"]),

    # retrieve + compare
    ("找某某某和A的论文，对比观点", ["retrieve", "compare"]),
    ("搜RAG论文并对比Naive和Advanced RAG", ["retrieve", "compare"]),

    # retrieve + extract
    ("查GPT-4论文，训练数据用了多少token", ["retrieve", "extract"]),
    ("找BERT论文，参数量是多少", ["retrieve", "extract"]),

    # retrieve + reason
    ("找RAG文章，分析为什么能降低幻觉", ["retrieve", "reason"]),
    ("搜一下Mamba的论文，为什么比Transformer快", ["retrieve", "reason"]),

    # retrieve + howto
    ("找LoRA微调教程，列出步骤", ["retrieve", "howto"]),
    ("搜vLLM部署文档，怎么配置", ["retrieve", "howto"]),

    # compare + reason
    ("对比CNN和Transformer，解释为什么Transformer更好", ["compare", "reason"]),
    ("比较DPO和PPO，说明为什么DPO更稳定", ["compare", "reason"]),

    # extract + compare
    ("BERT和GPT的参数量各是多少，哪个更大", ["extract", "compare"]),
    ("对比几种模型的训练数据量", ["extract", "compare"]),

    # ══════════════════════════════════════════
    #  三意图（4 种 × 低频）
    # ══════════════════════════════════════════

    # retrieve + summarize + compare
    ("找Agent综述，总结流派，对比优缺点", ["retrieve", "summarize", "compare"]),

    # retrieve + extract + compare
    ("找BERT和GPT论文，各自的参数量，对比", ["retrieve", "extract", "compare"]),

    # retrieve + reason + summarize
    ("查大模型幻觉原因，分析并总结现有方案", ["retrieve", "reason", "summarize"]),

    # retrieve + compare + reason
    ("找两个方法，对比差异，分析优劣原因", ["retrieve", "compare", "reason"]),
]


def match_keywords(query):
    matched = []
    for pattern, intent in RULES_PATTERNS:
        if pattern.search(query):
            matched.append(intent)
    seen = set()
    result = []
    for t in matched:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def compute_final_scores(scaled_scores, rule_signals, orig_scores=None):
    """与 task_router.py 一致的提权逻辑
    scaled_scores: 幂次缩放后的分数
    orig_scores: 原始 SBERT 分数（用于提权公式计算）
    """
    merged = {}
    for intent, score in scaled_scores.items():
        merged[intent] = score
    for intent in rule_signals:
        sbert_score = (orig_scores or scaled_scores).get(intent, 0)
        if sbert_score > BOOST_RANGE[0]:
            t = min((sbert_score - BOOST_RANGE[0]) / (BOOST_RANGE[1] - BOOST_RANGE[0]), 1.0)
            boosted = BOOST_TARGET[0] + (BOOST_TARGET[1] - BOOST_TARGET[0]) * t
            merged[intent] = max(merged.get(intent, 0), boosted)
    return merged


def decide_tasks(merged_scores, rule_signals, query):
    """与 task_router.py 一致的决策逻辑（不含 LLM 兜底）"""
    sorted_pairs = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    tasks = [t for t, s in sorted_pairs if s >= THRESHOLDS.get(t, 0.35)][:3]

    # 规则1
    if "howto" in tasks and "summarize" in tasks:
        tutorial_kw = r"步骤|步|代码|怎么|如何|教程|配置|部署|实现|流程"
        if not re.search(tutorial_kw, query):
            tasks.remove("howto")
            if not tasks:
                tasks = [t for t, s in sorted_pairs if s >= THRESHOLDS.get(t, 0.35) and t != "howto"][:1]

    # 规则2
    if "extract" in tasks and "retrieve" in tasks and "extract" not in rule_signals:
        quantity_kw = r"多少|参数量|token|准确率|数据量|参数|指标|数值|得分|成本|多少钱|用了多少|训练数据"
        if not re.search(quantity_kw, query):
            tasks.remove("extract")
            if not tasks:
                tasks = [t for t, s in sorted_pairs if s >= THRESHOLDS.get(t, 0.35) and t != "extract"][:1]

    # 规则3
    if "summarize" in tasks and "retrieve" in tasks:
        intro = r"介绍|什么是|是什么|啥是"
        summ_hint = r"综合|几篇|几份|多个|所有|整体|各种|整理|归纳|总结|概括"
        if (re.search(intro, query)
                and not re.search(summ_hint, query)
                and merged_scores.get("retrieve", 0) > merged_scores.get("summarize", 0)):
            tasks.remove("summarize")

    seen = set()
    final = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            final.append(t)
    tasks = final[:3]

    # 规则4
    keyword_priority = {"summarize": ["reason"], "reason": ["howto"], "howto": ["extract"]}
    for intent, competitors in keyword_priority.items():
        if intent in rule_signals and intent in tasks:
            for comp in competitors:
                if comp in tasks and comp not in rule_signals:
                    i_idx = tasks.index(intent)
                    c_idx = tasks.index(comp)
                    if c_idx < i_idx:
                        tasks[c_idx], tasks[i_idx] = tasks[i_idx], tasks[c_idx]
    return tasks


def adaptive_power_scale(scores_dict, power=2.0):
    """幂次缩放：拉大分差，映射到 [smin, 1.0]"""
    scores = np.array(list(scores_dict.values()))
    smin, smax = scores.min(), scores.max()
    if smax - smin < 1e-10:
        return dict(scores_dict)
    norm = (scores - smin) / (smax - smin)
    scaled = norm ** power
    target_range = 1.0 - smin
    return {k: smin + s * target_range for k, s in zip(scores_dict.keys(), scaled)}


def bar(score, length=25):
    filled = int(score * length)
    return "█" * filled + "░" * (length - filled)


# ══════════════════════════════════════
#  主流程
# ══════════════════════════════════════
print("加载 SBERT 模型 (ONNX INT8)...")
# 预加载模型，用于显示 6 维完整分数
clf = SBERTClassifier(model_name=MODEL_C)
# TaskRouter（内部也会加载 SBERT，但会被缓存命中忽略）
router = TaskRouter()
print("模型加载完成。\n")

total = len(MY_PREDICTIONS)
correct = 0
llm_used_count = 0
mistakes = []

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    _print = lambda s: print(s, file=f)
    f.write("我 vs 模型（含 LLM 兜底）— 意图预测对比测试\n")
    f.write(f"模型: {MODEL_C} (ONNX INT8)\n")
    f.write("=" * 80 + "\n")

    for query, my_intents in MY_PREDICTIONS:
        # ── 完整 route() 调用 ──
        route_result = router.route(query)
        model_tasks = route_result["user_tasks"]
        source = route_result["source"]
        is_llm = (source == "llm")
        if is_llm:
            llm_used_count += 1

        # ── 计算 6 维完整分数（用于展示） ──
        query_emb = clf.model.encode(query)
        query_norm = np.linalg.norm(query_emb)
        full_scores = {}
        for intent in INTENT_LABELS:
            anchor_embs = clf.anchor_embs[intent]
            sims = np.dot(anchor_embs, query_emb) / (
                np.linalg.norm(anchor_embs, axis=1) * query_norm + 1e-10
            )
            k = min(3, len(sims))
            top_sims = np.partition(sims, -k)[-k:]
            raw = top_sims.mean()
            neg_embs = clf.negative_embs[intent]
            neg_sims = np.dot(neg_embs, query_emb) / (
                np.linalg.norm(neg_embs, axis=1) * query_norm + 1e-10
            )
            if raw > 0.35:
                penalty = neg_sims.max() * 0.15
                adj = max(0.0, raw - penalty)
            else:
                adj = raw
            full_scores[intent] = adj

        # 幂次缩放 + 提权（与 route 内部一致，仅用于展示）
        scaled = adaptive_power_scale(full_scores, power=2.0)
        # route 内部实际使用的 sbert_scores（仅含 >=0.30 的意图）
        rule_signals = match_keywords(query)
        # 提权公式使用原始 SBERT 分数，但提权值叠加到缩放后的分数上
        boosted = compute_final_scores(scaled, rule_signals, orig_scores=full_scores)

        # ── 判断一致（单意图严格，多意图宽松） ──
        if len(my_intents) == 1:
            match = len(model_tasks) == 1 and model_tasks[0] == my_intents[0]
        else:
            match = any(m in my_intents for m in model_tasks) if model_tasks else False
        if match:
            correct += 1
        else:
            mistakes.append((query, my_intents, model_tasks, full_scores, scaled, boosted, source))

        # ── 输出 ──
        sep = "─" * 80
        _print(f"\n查询: {query}")
        _print(sep)
        _print(f"  我预测: {', '.join(my_intents)}")
        llm_tag = " [LLM 兜底]" if is_llm else ""
        _print(f"  模型决策: {', '.join(model_tasks)} (来源: {source}){llm_tag}")
        _print(f"  结果: {'✅ 一致' if match else '❌ 分歧'}")

        _print(f"\n  各意图分数 (原始 -> 缩放后 -> 提权后):")
        for intent in INTENT_LABELS:
            adj_val = full_scores[intent]
            scl_val = scaled[intent]
            bst_val = boosted.get(intent, adj_val)
            flag = ""
            if intent in model_tasks:
                flag = " ◀ 选中"
            _print(f"    {intent:12s}  {adj_val:.4f} -> {scl_val:.4f} -> {bst_val:.4f}  {bar(bst_val)}{flag}")
        _print(sep)

    # ── 汇总 ──
    accuracy = correct / total * 100
    _print("\n" + "=" * 80)
    _print("汇总")
    _print("=" * 80)
    _print(f"总用例: {total}")
    _print(f"一致: {correct}")
    _print(f"分歧: {total - correct}")
    _print(f"一致率: {accuracy:.1f}%")
    _print(f"LLM 兜底次数: {llm_used_count}/{total}")
    _print("=" * 80)

    if mistakes:
        _print(f"\n分歧详情 ({len(mistakes)} 处):")
        for q, my, model_t3, full_scores, scaled, boosted, source in mistakes:
            _print(f"\n  ❌ 查询: {q}")
            _print(f"     我预测: {my}")
            _print(f"     模型: {model_t3} (来源: {source})")
            # 模型最高分
            all_scored = sorted(boosted.items(), key=lambda x: x[1], reverse=True)
            _print(f"     模型最高: {all_scored[0][0]} ({all_scored[0][1]:.4f})")
            # 我预测的最高分
            my_scores = [(intent, boosted.get(intent, 0)) for intent in my]
            my_best = max(my_scores, key=lambda x: x[1])
            _print(f"     我预测最高: {my_best[0]} ({my_best[1]:.4f})")

print(f"\n测试完成！结果已写入 {OUTPUT_FILE}")
print(f"一致率: {accuracy:.1f}% ({correct}/{total})")
print(f"LLM 兜底次数: {llm_used_count}/{total}")
