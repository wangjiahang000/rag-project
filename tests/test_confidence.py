# tests/test_confidence.py
"""分析 routing 置信度分布，评估 LLM 兜底策略"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from content_core.task_router import TaskRouter
import re

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "test_queries_v3.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "confidence_analysis.txt")


def main():
    router = TaskRouter()

    # 读所有查询
    queries = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)

    records = []
    for q in queries:
        rule_signals = router._rule_match(q)
        sbert_scores = router._sbert_match(q)

        # 模拟 route 内部的 merged 分数
        merged = {}
        for intent, score in sbert_scores.items():
            merged[intent] = score

        BOOST_RANGE = (0.27, 0.50)
        BOOST_TARGET = (0.30, 0.65)
        for intent in rule_signals:
            sbert_score = sbert_scores.get(intent, 0)
            if sbert_score > BOOST_RANGE[0]:
                t = min((sbert_score - BOOST_RANGE[0]) / (BOOST_RANGE[1] - BOOST_RANGE[0]), 1.0)
                boosted = BOOST_TARGET[0] + (BOOST_TARGET[1] - BOOST_TARGET[0]) * t
                merged[intent] = max(merged.get(intent, 0), boosted)

        thresholds = {
            "retrieve": 0.30, "compare": 0.35, "summarize": 0.50,
            "howto": 0.35, "reason": 0.35, "extract": 0.45,
        }
        sorted_pairs = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        top_conf = sorted_pairs[0][1] if sorted_pairs else 0.0
        second_conf = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0
        gap = top_conf - second_conf

        top_intent = sorted_pairs[0][0] if sorted_pairs else None
        tasks = [t for t, s in sorted_pairs if s >= thresholds.get(t, 0.35)][:3]

        # 关键词信号矛盾度：rule_signals 中有但没有出现在 tasks 中的意图
        keyword_conflict = [s for s in rule_signals if s not in tasks]

        records.append({
            "query": q, "top_conf": top_conf, "second_conf": second_conf,
            "gap": gap, "top_intent": top_intent, "tasks": tasks,
            "rule_signals": rule_signals, "keyword_conflict": keyword_conflict,
            "n_tasks": len(sorted_pairs),
        })

    total = len(records)
    lines = []
    lines.append(f"总查询: {total}")
    lines.append(f"ONNX 模式: {router.sbert.onnx_mode}")
    lines.append("")

    # ── 指标1：top_conf 分布 ──
    lines.append("=== 1. top_conf 分布 ===")
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        n = sum(1 for r in records if r["top_conf"] < threshold)
        lines.append(f"  < {threshold:.2f}: {n:3d} 条 ({n/total*100:.1f}%)")
    lines.append("")

    # ── 指标2：top-2 分差分布 ──
    lines.append("=== 2. top-1 / top-2 分差分布 ===")
    for gap_thresh in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        n = sum(1 for r in records if r["gap"] < gap_thresh)
        lines.append(f"  gap < {gap_thresh:.2f}: {n:3d} 条 ({n/total*100:.1f}%)  这个阈值会触发 LLM")
    lines.append("")

    # ── 指标3：关键词矛盾 ──
    lines.append("=== 3. 关键词信号矛盾（规则命中但未进 tasks） ===")
    conflict_total = sum(1 for r in records if r["keyword_conflict"])
    lines.append(f"  共有 {conflict_total} 条有关键词命中但被排除（{conflict_total/total*100:.1f}%）")
    lines.append("")

    # ── 分差 < 0.05 且关键词矛盾的查询 ──
    lines.append("=== 4. 低分差(<0.05) + 关键词矛盾（最应送 LLM） ===")
    candidates = [r for r in records if r["gap"] < 0.05 and r["keyword_conflict"]]
    lines.append(f"  共 {len(candidates)} 条")
    for r in candidates:
        lines.append(f"  [{r['top_intent']}({r['top_conf']:.3f}) gap={r['gap']:.3f}] {r['query'][:35]}")
        lines.append(f"    rule_signals={r['rule_signals']} keyword_conflict={r['keyword_conflict']}")
    lines.append("")

    # ── 看 LLM fallback 策略效果 ──
    lines.append("=== 5. 组合策略效果评估 ===")
    for gap_thresh in [0.03, 0.05, 0.08, 0.10]:
        # 策略：gap < thresh OR 关键词矛盾
        triggered = sum(1 for r in records
                        if r["gap"] < gap_thresh or r["keyword_conflict"])
        lines.append(f"  gap<{gap_thresh:.2f} or keyword_conflict: {triggered:3d} 条 ({triggered/total*100:.1f}%) 送 LLM")
    lines.append("")

    # ── 低置信度查询明细 ──
    lines.append("=== 6. 分差 < 0.05 的查询 ===")
    for r in records:
        if r["gap"] < 0.05:
            lines.append(f"\n  [{r['top_intent']} {r['top_conf']:.3f} gap={r['gap']:.3f}] {r['query'][:40]}")
            lines.append(f"    tasks={r['tasks']}  rules={r['rule_signals']}")
    lines.append("")

    # ── 高置信度(>0.55)但疑似误判 ──
    lines.append("=== 7. 高置信度(>0.65)但疑似误判 ===")
    suspected = [
        ("为啥", "reason"), ("干啥", "reason"),
        ("有啥不同", "compare"), ("RLHF", "reason"),
        ("提出来", "extract"), ("方法有啥不同", "compare"),
    ]
    for r in records:
        if r["top_conf"] < 0.65:
            continue
        for kw, expected in suspected:
            if kw in r["query"]:
                if r["top_intent"] != expected:
                    lines.append(f"  [{r['top_intent']:10s} ({r['top_conf']:.3f}) gap={r['gap']:.3f}] "
                                f"exp={expected:10s} → {r['query'][:35]}")
                    break

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"分析完成: {OUTPUT_FILE}")
    print(f"\n关键数字:")
    print(f"  top_conf < 0.50: {sum(1 for r in records if r['top_conf'] < 0.50)}")
    print(f"  gap < 0.05: {sum(1 for r in records if r['gap'] < 0.05)}")
    print(f"  关键词矛盾: {sum(1 for r in records if r['keyword_conflict'])}")
    print(f"  gap<0.03 + 关键词矛盾: {sum(1 for r in records if r['gap'] < 0.03 or r['keyword_conflict'])}")


if __name__ == "__main__":
    main()
