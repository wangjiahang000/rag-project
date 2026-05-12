# tests/test_full_pipeline.py
import sys
import os
import re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_core.models.sbert_classifier import SBERTClassifier

MODEL_C = "BAAI/bge-base-zh-v1.5"

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "test_queries_v3.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "a.txt")

# ── 关键词规则（与 task_router.py 保持一致）──
RULES = [
    (re.compile(r"对比|区别|比较|vs|不同|差异|不一样|哪个更"), "compare"),
    (re.compile(r"总结|汇总|概括|归纳|概述|理一理"), "summarize"),
    (re.compile(r"提取|抽取|获取|找出|查出|参数|指标|数值|多少"), "extract"),
    (re.compile(r"找|检索|搜索|查|查找|什么是|是什么|定义|介绍"), "retrieve"),
    (re.compile(r"怎么|如何|步骤|教程|配置|部署|实现|导出|转换"), "howto"),
    (re.compile(r"为什么|原因|导致|背后|原理|造成"), "reason"),
]

# ── 差异化阈值（与 task_router.py 保持一致）──
THRESHOLDS = {
    "retrieve": 0.30,
    "compare": 0.35,
    "summarize": 0.50,
    "howto": 0.35,
    "reason": 0.35,
    "extract": 0.45,
}

BOOST_RANGE = (0.27, 0.50)
BOOST_TARGET = (0.30, 0.65)

INTENT_LABELS = ["retrieve", "compare", "summarize", "howto", "reason", "extract"]

# ── 从分类标题推断预期意图 ──
SECTION_INTENT_MAP = {
    "检索": ["retrieve"],
    "对比": ["compare"],
    "归纳/总结": ["summarize"],
    "实操/步骤": ["howto"],
    "原因": ["reason"],
    "提取": ["extract"],
    "混合意图": ["retrieve", "compare", "summarize", "howto", "reason", "extract"],  # 混合：多项
}

# ── 计算 SBERT 分数（6 个意图，top-3 均值 + 负例校准） ──
def compute_sbert_scores(clf, query):
    query_emb = clf.model.encode(query)
    query_norm = np.linalg.norm(query_emb)
    results = {}
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
        results[intent] = (raw, adj)
    return results

# ── 自适应幂次缩放（拉大分数差距） ──
def adaptive_power_scale(scores_dict, power=2.0):
    scores = np.array(list(scores_dict.values()))
    smin, smax = scores.min(), scores.max()
    if smax - smin < 1e-10:
        return dict(scores_dict)  # 全部相等，无需缩放
    norm = (scores - smin) / (smax - smin)
    scaled = norm ** power
    return {k: smin + s * (1.0 - smin) for k, s in zip(scores_dict.keys(), scaled)}

# ── 关键词匹配 ──
def match_keywords(query):
    matched = []
    for pattern, intent in RULES:
        if pattern.search(query):
            matched.append(intent)
    seen = set()
    result = []
    for t in matched:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result

# ── 关键词提权 ──
def compute_final_scores(sbert_scores, rule_signals):
    # sbert_scores: {intent: adj_score}
    merged = {}
    for intent, score in sbert_scores.items():
        merged[intent] = score

    for intent in rule_signals:
        sbert_score = sbert_scores.get(intent, 0)
        if sbert_score > BOOST_RANGE[0]:
            t = min((sbert_score - BOOST_RANGE[0]) / (BOOST_RANGE[1] - BOOST_RANGE[0]), 1.0)
            boosted = BOOST_TARGET[0] + (BOOST_TARGET[1] - BOOST_TARGET[0]) * t
            merged[intent] = max(merged.get(intent, 0), boosted)

    return merged

# ── 融合决策：阈值 + 解耦规则 + 排序（与 task_router.route 一致） ──
def decide_tasks(merged_scores, rule_signals, query):
    sorted_pairs = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    tasks = [t for t, s in sorted_pairs if s >= THRESHOLDS.get(t, 0.35)][:3]

    # 规则1：howto 和 summarize 同时出现，无步骤关键词则移除 howto
    if "howto" in tasks and "summarize" in tasks:
        tutorial_kw = r"步骤|步|代码|怎么|如何|教程|配置|部署|实现|流程"
        if not re.search(tutorial_kw, query):
            tasks.remove("howto")
            if not tasks:
                tasks = [t for t, s in sorted_pairs if s >= THRESHOLDS.get(t, 0.35) and t != "howto"][:1]

    # 规则2：extract 和 retrieve 同时出现，extract 无关键词且无数值量词则移除
    if "extract" in tasks and "retrieve" in tasks and "extract" not in rule_signals:
        quantity_kw = r"多少|参数量|token|准确率|数据量|参数|指标|数值|得分|成本|多少钱|用了多少|训练数据"
        if not re.search(quantity_kw, query):
            tasks.remove("extract")
            if not tasks:
                tasks = [t for t, s in sorted_pairs if s >= THRESHOLDS.get(t, 0.35) and t != "extract"][:1]

    # 规则3：纯介绍排除误触 summarize
    if "summarize" in tasks and "retrieve" in tasks:
        intro = r"介绍|什么是|是什么|啥是"
        summ_hint = r"综合|几篇|几份|多个|所有|整体|各种|整理|归纳|总结|概括"
        if (re.search(intro, query)
                and not re.search(summ_hint, query)
                and merged_scores.get("retrieve", 0) > merged_scores.get("summarize", 0)):
            tasks.remove("summarize")

    # 去重保序最多3个
    seen = set()
    final = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            final.append(t)
    tasks = final[:3]

    # 规则4：关键词优先于语义混淆
    keyword_priority = {
        "summarize": ["reason"],
        "reason": ["howto"],
        "howto": ["extract"],
    }
    for intent, competitors in keyword_priority.items():
        if intent in rule_signals and intent in tasks:
            for comp in competitors:
                if comp in tasks and comp not in rule_signals:
                    i_idx = tasks.index(intent)
                    c_idx = tasks.index(comp)
                    if c_idx < i_idx:
                        tasks[c_idx], tasks[i_idx] = tasks[i_idx], tasks[c_idx]

    return tasks


# ── 解析测试集 ──
def load_queries_with_expected(path):
    entries = []
    current_expected = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') and '──' in line:
                # 先从分类映射获取
                current_expected = ["other"]
                for key, intents in SECTION_INTENT_MAP.items():
                    if key in line:
                        current_expected = intents
                        break
                continue
            if line.startswith('#'):
                continue
            if line:
                # 对于未明确分类的段落（含特定领域、含专有名词等），用关键词推断预期
                if current_expected == ["other"]:
                    kw_expected = match_keywords(line)
                    if kw_expected:
                        current_expected = kw_expected
                    else:
                        current_expected = ["other"]
                entries.append((line, list(current_expected)))
    return entries

# ── 可视化条 ──
def bar(score, length=25):
    filled = int(score * length)
    return "█" * filled + "░" * (length - filled)

# ── 显示单条结果 ──
def format_results(query, expected, sbert_scores, scaled_sbert_adj, final_scores, final_scores_orig, rule_signals, decided, file):
    _print = lambda s: print(s, file=file)
    sep = "─" * 80

    _print(f"\n查询: {query}")
    _print(sep)
    _print(f"  预期: {', '.join(expected) if expected else '其他'}")
    _print(f"  关键词命中: {', '.join(rule_signals) if rule_signals else '无'}")

    _print(f"\n  ── 缩放前 ──                             ── 缩放后 ──")
    for intent in INTENT_LABELS:
        raw_val, adj_val = sbert_scores[intent]
        scl_val = scaled_sbert_adj.get(intent, adj_val)
        boosted = final_scores.get(intent, adj_val)
        flag = " ◀ 提权" if intent in rule_signals and boosted > scl_val else ""
        _print(f"  {intent}  {adj_val:.4f} -> {scl_val:.4f} -> {boosted:.4f}{flag}")
        _print(f"    缩放前: {bar(adj_val)}  {adj_val:.4f}")
        _print(f"    缩放后: {bar(scl_val)}  {scl_val:.4f}")

    _print(f"\n  最终决策: {', '.join(decided) if decided else '无'}")
    match = any(d in expected for d in decided) if expected != ["other"] else True
    status = "✅" if match else "❌"
    _print(f"  结果: {status}  (预期={expected}, 实际={decided})")
    _print(sep)


# ══════════════════════════════════════
#  主流程
# ══════════════════════════════════════
print("加载 SBERT 模型 (ONNX INT8)...")
clf = SBERTClassifier(model_name=MODEL_C)
print("模型加载完成。\n")

entries = load_queries_with_expected(QUERIES_FILE)
print(f"共加载 {len(entries)} 条测试用例\n")

correct = 0
total = 0
results_summary = []

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("全链路测试报告 — ONNX INT8 量化版\n")
    f.write(f"模型: {MODEL_C}\n")
    f.write("=" * 80 + "\n")

    for query, expected in entries:
        # SBERT 分数
        sbert_scores = compute_sbert_scores(clf, query)
        sbert_adj = {intent: v[1] for intent, v in sbert_scores.items()}

        # 自适应幂次缩放（在原始区间内拉大差距）
        scaled_adj = adaptive_power_scale(sbert_adj, power=2.0)

        # 关键词
        rule_signals = match_keywords(query)

        # 提权后综合分数（基于缩放后分数）
        final_scores = compute_final_scores(scaled_adj, rule_signals)
        # 原始未缩放的提权结果（仅用于对比显示）
        final_scores_orig = compute_final_scores(sbert_adj, rule_signals)

        # 最终决策（基于缩放后分数）
        decided = decide_tasks(final_scores, rule_signals, query)

        format_results(query, expected, sbert_scores, scaled_adj, final_scores, final_scores_orig, rule_signals, decided, file=f)

        # 统计
        if expected != ["other"]:
            total += 1
            if any(d in expected for d in decided):
                correct += 1
                results_summary.append((query, expected, decided, "✅"))
            else:
                results_summary.append((query, expected, decided, "❌"))

    # 汇总
    _print = lambda s: print(s, file=f)
    _print("\n\n" + "=" * 80)
    _print("测试汇总")
    _print("=" * 80)
    accuracy = correct / total * 100 if total > 0 else 0
    _print(f"总用例: {total}")
    _print(f"通过: {correct}")
    _print(f"失败: {total - correct}")
    _print(f"准确率: {accuracy:.1f}%")
    _print("=" * 80)
    _print("\n失败详情:")
    for query, expected, decided, status in results_summary:
        if status == "❌":
            _print(f"  ❌ 预期={expected}, 实际={decided}")
            _print(f"     查询: {query}")

print(f"测试完成，结果已写入 {OUTPUT_FILE}")
print(f"准确率: {accuracy:.1f}% ({correct}/{total})")
