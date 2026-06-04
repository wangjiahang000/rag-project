"""评估框架：检索精度 + 意图路由准确率 + 响应延迟

使用方法：
    # 默认评估（意图路由 + 检索延迟）
    python eval/run_evaluation.py

    # 指定模式
    python eval/run_evaluation.py --mode all          # 完整评估
    python eval/run_evaluation.py --mode routing       # 仅评估路由
    python eval/run_evaluation.py --mode latency       # 仅评估延迟
    python eval/run_evaluation.py --mode ablation      # 消融实验
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ── 意图路由准确率评估 ──

ROUTING_QUERIES = [
    # (查询, 预期意图列表)
    ("什么是 Transformer 中的自注意力机制", ["retrieve"]),
    ("介绍 BERT 模型的核心思想", ["retrieve"]),
    ("RAG 技术的最新研究进展", ["retrieve"]),
    ("对比 BERT 和 GPT 的架构差异", ["compare"]),
    ("LoRA 和 QLoRA 有什么区别", ["compare"]),
    ("PyTorch 和 TensorFlow 哪个更适合做研究", ["compare"]),
    ("总结一下知识图谱的构建方法", ["summarize"]),
    ("概括 RAG 系统的主要技术路线", ["summarize"]),
    ("如何实现一个简单的 RAG 系统", ["howto"]),
    ("怎么配置 LoRA 微调环境", ["howto"]),
    ("为什么 Transformer 比 RNN 效果好", ["reason"]),
    ("为什么需要位置编码", ["reason"]),
    ("从论文中提取实验参数设置", ["extract"]),
    ("提取 BERT 模型的参数量", ["extract"]),
    ("你好", ["chitchat"]),
    ("你是谁", ["chitchat"]),
    ("谢谢", ["chitchat"]),
]


def evaluate_routing() -> Dict:
    """评估意图路由准确率"""
    from content_core.task_router import TaskRouter

    router = TaskRouter()
    correct = 0
    total = 0
    results = []

    for query, expected in ROUTING_QUERIES:
        try:
            result = router.route(query)
            actual = result["user_tasks"]
        except Exception as e:
            actual = [f"error: {e}"]

        # chitchat 特殊处理：实际可能会被 LLM 兜底或规则拦截
        if expected == ["chitchat"]:
            is_match = "chitchat" in actual or (
                not any(t in actual for t in
                        ["retrieve", "compare", "summarize", "howto", "reason", "extract"])
            )
        else:
            is_match = any(e in actual for e in expected)

        if is_match:
            correct += 1
        total += 1
        results.append((query, expected, actual, is_match))

    accuracy = correct / total * 100 if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": results,
    }


# ── 延迟评估 ──

def evaluate_latency(num_runs: int = 5) -> Dict:
    """评估端到端响应延迟（仅路由层，不依赖向量库）"""
    from content_core.task_router import TaskRouter

    router = TaskRouter()
    queries = [q for q, _ in ROUTING_QUERIES[:10]]

    latencies = []
    for _ in range(num_runs):
        for q in queries:
            start = time.perf_counter()
            router.route(q)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

    return {
        "avg_ms": (sum(latencies) / len(latencies)) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "p50_ms": sorted(latencies)[len(latencies) // 2] * 1000,
        "num_samples": len(latencies),
    }


# ── 消融实验（路由层）──

def _make_ablation_router(disable_boost=False, disable_power_scale=False):
    """创建带消融配置的路由器"""
    from content_core.task_router import TaskRouter
    import content_core.config as cfg

    router = TaskRouter()

    if disable_boost:
        cfg.BOOST_RANGE = (0.0, 0.0)
        cfg.BOOST_TARGET = (0.0, 0.0)
    if disable_power_scale:
        cfg.POWER_SCALE = 1.0

    return router


def evaluate_ablation() -> List[Dict]:
    """消融实验：逐一移除路由层组件"""
    experiments = [
        ("full",            "完整系统",          False, False),
        ("no_boost",        "移除关键词提权",     True,  False),
        ("no_power_scale",  "移除幂次缩放",       False, True),
    ]

    results = []
    for name, label, no_boost, no_power in experiments:
        router = _make_ablation_router(no_boost, no_power)

        correct = 0
        total = 0
        for query, expected in ROUTING_QUERIES:
            if expected == ["chitchat"]:
                continue  # 消融排除 chitchat
            try:
                result = router.route(query)
                actual = result["user_tasks"]
            except Exception:
                actual = []
            if any(e in actual for e in expected):
                correct += 1
            total += 1

        accuracy = correct / total * 100 if total > 0 else 0
        results.append({
            "experiment": name,
            "label": label,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        })

    return results


# ── 主入口 ──

def print_report(report: Dict):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("  评 估 报 告")
    print("=" * 60)

    for section, data in report.items():
        print(f"\n--- {section} ---")
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "details":
                    continue
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(data, list):
            for item in data:
                print(f"  [{item['experiment']}] {item['label']}: "
                      f" 准确率={item['accuracy']:.1f}% ({item['correct']}/{item['total']})")

    if "routing" in report and "details" in report["routing"]:
        print("\n--- 路由详情 ---")
        for query, expected, actual, ok in report["routing"]["details"]:
            mark = "✅" if ok else "❌"
            print(f"  {mark} 预期={expected}, 实际={actual}")
            print(f"      查询: {query}")


def main():
    parser = argparse.ArgumentParser(description="评估框架")
    parser.add_argument("--mode", default="all",
                        choices=["all", "routing", "latency", "ablation"])
    args = parser.parse_args()

    report = {}

    if args.mode in ("all", "routing"):
        print("正在评估意图路由准确率...")
        report["routing"] = evaluate_routing()

    if args.mode in ("all", "latency"):
        print("正在评估响应延迟...")
        report["latency"] = evaluate_latency()

    if args.mode in ("all", "ablation"):
        print("正在运行消融实验...")
        report["ablation"] = evaluate_ablation()

    print_report(report)

    # 输出 JSON
    output_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    json_safe = {}
    for k, v in report.items():
        if isinstance(v, dict):
            json_safe[k] = {kk: vv for kk, vv in v.items() if kk != "details"}
        else:
            json_safe[k] = v
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, ensure_ascii=False, indent=2)
    print(f"\n评估报告已保存: {output_path}")


if __name__ == "__main__":
    main()
