"""
test_v2.py — TaskRouter 路由结果测试

测试目的：
  验证 TaskRouter.route() 对各类查询的意图识别结果。

运行方式：
  python tests/test_v2.py

测试覆盖：
  - 已知意图组合 → 正确识别
  - 闲聊查询 → chitchat
  - 空查询 → 不崩溃
  - 混合意图 → 多意图输出
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from content_core.task_router import TaskRouter


# -- 测试用例 --
TEST_CASES = [
    # (场景, 查询, 期望包含的意图)
    ("空查询", "", []),
    ("纯空格", "   ", []),
    ("闲聊-问候", "你好", ["chitchat"]),
    ("闲聊-致谢", "谢谢", ["chitchat"]),
    ("闲聊-你是谁", "你是谁", ["chitchat"]),
    ("单意图-retrieve", "找几篇RAG相关的论文", ["retrieve"]),
    ("单意图-retrieve2", "什么是Transformer", ["retrieve"]),
    ("单意图-compare", "LoRA和QLoRA有什么不同", ["compare"]),
    ("单意图-compare2", "对比一下RAG和微调", ["compare"]),
    ("单意图-summarize", "总结一下RAG的最新进展", ["summarize"]),
    ("单意图-howto", "怎么用PyTorch训练一个模型", ["howto"]),
    ("单意图-reason", "为什么大模型会出现幻觉", ["reason"]),
    ("单意图-extract", "BERT模型的参数量是多少", ["extract"]),
    ("双意图-retrieve+summarize", "搜索RAG相关的论文并总结", ["retrieve", "summarize"]),
    ("双意图-compare+reason", "对比RAG和微调，分析原因", ["compare", "reason"]),
    ("双意图-extract+compare", "提取LoRA和QLoRA的参数对比", ["extract", "compare"]),
    ("混合长查询", "对比一下Transformer和BERT的区别，并总结各自的优缺点", ["compare"]),
]


def main():
    print("=" * 80)
    print("TaskRouter 路由结果测试")
    print("=" * 80)

    router = TaskRouter()

    results = []
    passed = 0
    failed = 0

    for scenario, query, expected in TEST_CASES:
        try:
            result = router.route(query)
        except Exception as e:
            print(f"\n[崩溃] {scenario}: {e}")
            failed += 1
            continue

        tasks = result.get("user_tasks", [])
        source = result.get("source", "?")

        # 检查是否包含期望意图
        if not expected:
            ok = tasks == []
        else:
            ok = all(e in tasks for e in expected)

        # 附加检查：chitchat 查询不应包含非闲聊意图
        if "chitchat" in expected and any(t != "chitchat" for t in tasks):
            ok = False

        status = "[OK]" if ok else "[FAIL]"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} [{scenario}]")
        print(f"  查询:   {query[:50]}{'...' if len(query) > 50 else ''}")
        print(f"  期望:   {expected or '空列表'}")
        print(f"  结果:   {tasks}")
        print(f"  来源:   {source}")
        print(f"  复杂度: {result.get('complexity', '?')}")

        results.append({
            "scenario": scenario,
            "query": query,
            "expected": expected,
            "result": tasks,
            "source": source,
            "passed": ok,
        })

    # -- 汇总 --
    total = passed + failed
    print(f"\n\n{'=' * 80}")
    print(f"汇总: {passed}/{total} 通过, {failed}/{total} 失败")
    print(f"{'=' * 80}")

    for r in results:
        if not r["passed"]:
            print(f"  [FAIL] [{r['scenario']}] 期望 {r['expected']}, 得到 {r['result']}")

    # -- 输出 JSON --
    output_path = os.path.join(os.path.dirname(__file__), "test_v2_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"passed": passed, "failed": failed, "total": total},
            "cases": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
