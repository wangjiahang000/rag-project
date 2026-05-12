# tests/test_routing_v3.py
"""TaskRouter 200条口语化测试"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
# 从 .env 读取 API key
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
with open(env_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DEEPSEEK_API_KEY="):
            os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1]
            break

from content_core.task_router import TaskRouter

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "test_queries_v3.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "a_v3.txt")


def main():
    router = TaskRouter()
    lines = []

    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = router.route(line)
            except Exception:
                r = {"user_tasks": ["error"], "source": "ERR", "resource_hint": "?"}
            lines.append((line, r))

    # 写文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"TaskRouter 口语化测试 — ONNX: {router.sbert.onnx_mode}\n")
        f.write(f"{'Query':40s} {'Tasks':45s} {'Source':10s}\n")
        f.write("-" * 100 + "\n")
        for q, r in lines:
            tasks = ", ".join(r["user_tasks"])
            f.write(f"{q[:38]:40s} {tasks:45s} {r['source']:10s}\n")

    # 统计分析
    from collections import Counter
    intent_counts = Counter()
    sources = Counter()
    errors = 0
    for q, r in lines:
        for t in r["user_tasks"]:
            intent_counts[t] += 1
        sources[r["source"]] += 1

    print("\n========== 测试报告 ==========")
    print(f"总查询: {len(lines)}")
    print(f"来源分布: {dict(sources)}")
    print(f"意图分布:")
    for intent, count in intent_counts.most_common():
        print(f"  {intent:12s}: {count}")
    print(f"\n结果文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
