# tests/test_routing.py
"""TaskRouter 端到端测试（ONNX INT8 模式）"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["DEEPSEEK_API_KEY"] = "sk-xxx"

from content_core.task_router import TaskRouter

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "test_queries_v2.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "a.txt")


def main():
    router = TaskRouter()
    results = []
    results.append(f"TaskRouter 测试 — ONNX 模式: {router.sbert.onnx_mode}")
    results.append(f"{'Query':35s} {'Tasks':40s} {'Source':10s} {'Resource':10s}")
    results.append("-" * 100)

    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = router.route(line)
            except Exception:
                r = {"user_tasks": ["llm_error"], "source": "ERROR", "resource_hint": "?"}
            tasks = ", ".join(r["user_tasks"])
            results.append(
                f"{line[:33]:35s} {tasks:40s} {r['source']:10s} {r['resource_hint']:10s}"
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"结果已写入 {OUTPUT_FILE}")
    print(f"共 {len(results) - 3} 条查询")
    # 统计
    from collections import Counter
    intents = Counter()
    for line in results[3:]:
        parts = line.split()
        if parts:
            intents.update(parts[1].split(", "))
    print(f"\n意图分布: {dict(intents)}")


if __name__ == "__main__":
    main()
