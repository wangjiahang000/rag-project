"""
test_v1.py — 关键词 + SBERT 加权 vs 纯 SBERT 效果对比

测试目的：
  验证 TaskRouter 的关键词平滑提权策略是否有效改善了
  意图识别的排序结果。

运行方式：
  python tests/test_v1.py

输出说明：
  - sbert_only: 纯 SBERT 语义分数（经过自适应幂次缩放）
  - weighted:   关键词信号 + SBERT 加权融合后的分数
  - top_diff/rank_shift: 加权后排序变化
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import json
import numpy as np
from content_core.models.sbert_classifier import SBERTClassifier
import content_core.config as cfg

# -- 测试用例 --
TEST_CASES = [
    # (描述, 查询, 期望的 top-1 意图)
    ("明确对比", "LoRA和QLoRA有什么区别", "compare"),
    ("检索+介绍", "介绍一下Transformer的原理", "retrieve"),
    ("howto+实现", "怎么用PyTorch实现一个Transformer", "howto"),
    ("extract+数值", "BERT模型的参数量是多少", "extract"),
    ("reason+原因", "为什么ResNet比VGG效果好", "reason"),
    ("summarize+归纳", "总结一下RAG的最新研究进展", "summarize"),
    ("混合意图", "对比一下RAG和微调的区别，并总结各自的优缺点", "compare"),
    ("describe+检索", "什么是知识蒸馏", "retrieve"),
    ("教程类", "如何配置LoRA微调环境", "howto"),
    ("提取类", "从论文中提取训练集的准确率", "extract"),
    ("原因类", "为什么大模型会出现幻觉", "reason"),
    ("检索+介绍", "有没有关于多模态模型的最新论文", "retrieve"),
]


def power_scale(scores_dict, power=2.0):
    """自适应幂次缩放（同 TaskRouter._adaptive_power_scale）"""
    if not scores_dict:
        return {}
    scores = np.array(list(scores_dict.values()))
    smin, smax = scores.min(), scores.max()
    if smax - smin < 1e-10:
        return dict(scores_dict)
    norm = (scores - smin) / (smax - smin)
    scaled = norm ** power
    target_range = 1.0 - smin
    return dict(zip(scores_dict.keys(), smin + scaled * target_range))


def keyword_boost(sbert_scores, rule_signals, scaled_scores):
    """关键词加权融合（同 TaskRouter 的融合逻辑）"""
    merged = dict(scaled_scores)
    for intent in rule_signals:
        sbert_score = sbert_scores.get(intent, 0)
        boost_range = cfg.BOOST_RANGE
        boost_target = cfg.BOOST_TARGET
        if sbert_score > boost_range[0]:
            t = min((sbert_score - boost_range[0]) / (boost_range[1] - boost_range[0]), 1.0)
            boosted = boost_target[0] + (boost_target[1] - boost_target[0]) * t
            merged[intent] = max(merged.get(intent, 0), boosted)
    return merged


# -- 关键词规则（同 TaskRouter） --
RULES = [
    (r"对比|区别|比较|vs|不同|差异|不一样|更|哪个", "compare"),
    (r"总结|汇总|概括|归纳|概述|理一理", "summarize"),
    (r"提取|抽取|获取|找出|查出|参数|指标|数值|多少", "extract"),
    (r"找|检索|搜|搜索|查|查找|什么是|是什么|定义|介绍", "retrieve"),
    (r"怎么|如何|步骤|教程|配置|部署|实现|导出|转换", "howto"),
    (r"为什么|原因|导致|背后|原理|造成", "reason"),
]


def rule_match(query):
    tasks = []
    for pattern, task in RULES:
        if re.search(pattern, query):
            tasks.append(task)
    seen = set()
    result = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def format_score_bar(score, width=30):
    filled = int(score * width)
    return "#" * filled + "." * (width - filled)


def main():
    print("=" * 80)
    print("关键词 + SBERT 加权 vs 纯 SBERT 效果对比")
    print("=" * 80)

    classifier = SBERTClassifier()

    results = []

    for label, query, expected in TEST_CASES:
        # 纯 SBERT
        tasks, scores_list = classifier.classify(query)
        sbert_raw = dict(zip(tasks, scores_list))
        # 补全所有意图
        all_intents = ["retrieve", "compare", "summarize", "howto", "reason", "extract"]
        for intent in all_intents:
            if intent not in sbert_raw:
                sbert_raw[intent] = 0.0

        # 幂次缩放（放大差距）
        scaled = power_scale(sbert_raw, power=cfg.POWER_SCALE)

        # 关键词匹配
        signals = rule_match(query)

        # 加权融合
        weighted = keyword_boost(sbert_raw, signals, scaled)

        # 排序
        sbert_sorted = sorted(scaled.items(), key=lambda x: x[1], reverse=True)
        weighted_sorted = sorted(weighted.items(), key=lambda x: x[1], reverse=True)

        sbert_top1 = sbert_sorted[0][0] if sbert_sorted else "none"
        weighted_top1 = weighted_sorted[0][0] if weighted_sorted else "none"
        rank_shift = "[OK]" if weighted_top1 == expected else "[FAIL]"
        sbert_ok = sbert_top1 == expected

        results.append({
            "label": label,
            "query": query,
            "expected": expected,
            "sbert_top1": sbert_top1,
            "weighted_top1": weighted_top1,
            "sbert_correct": sbert_ok,
            "weighted_correct": rank_shift == "[OK]",
            "signals": signals,
        })

        print(f"\n{'-' * 80}")
        print(f"[{label}] {query}")
        print(f"  关键词命中: {signals or '无'}")
        print(f"  期望意图:   {expected}")
        print(f"  纯 SBERT:   {sbert_top1} (top-1)  {'[OK]' if sbert_ok else '[FAIL]'}")
        print(f"  加权后:     {weighted_top1} (top-1)  {rank_shift}")
        print()

        # 展示前 3 个意图的分数对比
        all_intents_sorted = sorted(
            set(sbert_sorted[0] for sbert_sorted in sbert_sorted[:3]) |
            set(s for s, _ in weighted_sorted[:3]),
        )
        print(f"  {'意图':<12} {'纯 SBERT':<20} {'加权后':<20}")
        print(f"  {'-' * 52}")
        for intent in all_intents_sorted:
            s = scaled.get(intent, 0)
            w = weighted.get(intent, 0)
            bar_s = format_score_bar(s)
            bar_w = format_score_bar(w)
            kw = " ←关键词" if intent in signals else ""
            print(f"  {intent:<12} {s:.3f} {bar_s}  {w:.3f} {bar_w}{kw}")

    # -- 汇总统计 --
    print(f"\n\n{'=' * 80}")
    print("汇总统计")
    print(f"{'=' * 80}")

    total = len(results)
    sbert_correct = sum(1 for r in results if r["sbert_correct"])
    weighted_correct = sum(1 for r in results if r["weighted_correct"])
    improved = sum(1 for r in results if r["weighted_correct"] and not r["sbert_correct"])
    regressed = sum(1 for r in results if r["sbert_correct"] and not r["weighted_correct"])

    print(f"  总测试数:          {total}")
    print(f"  纯 SBERT 正确:     {sbert_correct:>2}/{total} ({sbert_correct/total*100:.0f}%)")
    print(f"  加权后正确:        {weighted_correct:>2}/{total} ({weighted_correct/total*100:.0f}%)")
    print(f"  加权后改进:        +{improved}")
    print(f"  加权后退化:        -{regressed}")

    # -- 输出 JSON --
    output_path = os.path.join(os.path.dirname(__file__), "test_v1_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "sbert_correct": sbert_correct,
                "weighted_correct": weighted_correct,
                "improved": improved,
                "regressed": regressed,
            },
            "cases": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
