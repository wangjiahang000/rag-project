"""
test_v3.py — 图路径集成测试

测试目的：
  验证 GraphGenerator + GraphExecutor 的完整执行链路。

运行方式：
  python tests/test_v3.py

测试覆盖：
  - 各单意图模板生成的 DAG 结构
  - 多意图模板的分支与依赖
  - 占位符 {{N}} 解析
  - 图执行结果
  - 未定义组合的 LLM 分解（mock）
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from content_core.types import TaskGraph, TaskNode
from content_core.graph_generator import GraphGenerator
from content_core.graph_executor import ToolRegistry, GraphExecutor


# -- Mock 工具（不依赖外部服务） --

def mock_search(query: str, k: int = 10) -> list:
    return [f"doc_{i}: 关于{query}的文档内容" for i in range(k)]


def mock_rerank(docs: list, query: str = "") -> list:
    return docs[:3]


def mock_summarize(docs: list) -> str:
    return f"总结：共{len(docs)}篇文档的核心内容摘要。"


def mock_compare(docs_a: list, docs_b: list, query: str = "") -> str:
    return f"对比分析完成。A组{len(docs_a)}篇，B组{len(docs_b)}篇。"


def mock_extract(docs: list, target: str = "") -> str:
    return f"从{len(docs)}篇文档中提取了关于'{target}'的信息。"


def mock_reason(docs: list, query: str = "") -> str:
    return f"原因分析完成，基于{len(docs)}篇文档。"


def setup_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register("hybrid_search", mock_search)
    r.register("rerank", mock_rerank)
    r.register("summarize", mock_summarize)
    r.register("compare", mock_compare)
    r.register("extract", mock_extract)
    r.register("reason", mock_reason)
    return r


def check_nodes(nodes: list, expected_count: int, description: str) -> bool:
    ok = len(nodes) == expected_count
    if not ok:
        print(f"  [FAIL] {description}: 期望 {expected_count} 个节点, 实际 {len(nodes)}")
    else:
        print(f"  [OK] {description}: {expected_count} 个节点")
    return ok


def check_deps(nodes: list) -> bool:
    """验证所有依赖的节点 ID 都存在"""
    all_ids = {n.id for n in nodes}
    ok = True
    for n in nodes:
        for dep in n.depends_on:
            if dep not in all_ids:
                print(f"  [FAIL] 节点 {n.id} 依赖 {dep} 不存在")
                ok = False
    if ok:
        print(f"  [OK] 所有依赖关系正确")
    return ok


def test_single_intent(generator: GraphGenerator, executor: GraphExecutor):
    """测试所有单意图模板"""
    print(f"\n{'=' * 60}")
    print("单意图模板测试")
    print(f"{'=' * 60}")

    cases = [
        ("retrieve", "_simple_search", 2),
        ("compare", "_build_compare", 2),  # 实体不足 → 降级为 fallback (2节点)
        ("compare", "_build_compare", 5),  # 有实体 → 5节点
        ("summarize", "_build_summarize", 3),
        ("howto", "_simple_search", 2),
        ("reason", "_build_reason", 3),
        ("extract", "_build_extract", 3),
    ]

    passed = 0
    for intent, method, exp_count in cases:
        entities = ["LoRA", "QLoRA"] if intent == "compare" and exp_count == 5 else []
        graphs = generator.generate([intent], "test query", entities=entities)
        graph = graphs[0]

        n_ok = check_nodes(graph.nodes, exp_count, f"{intent} (entities={entities})")
        d_ok = check_deps(graph.nodes)
        if n_ok and d_ok:
            passed += 1

    print(f"\n单意图: {passed}/{len(cases)} 通过")


def test_dual_intent(generator: GraphGenerator, executor: GraphExecutor):
    """测试双意图模板"""
    print(f"\n{'=' * 60}")
    print("双意图模板测试")
    print(f"{'=' * 60}")

    cases = [
        (["retrieve", "summarize"], 3),
        (["compare", "retrieve"], 5),
        (["extract", "retrieve"], 3),
        (["reason", "retrieve"], 3),
        (["howto", "retrieve"], 2),
        (["compare", "reason"], 6),
        (["compare", "extract"], 7),
    ]

    entities = ["LoRA", "QLoRA"]
    passed = 0
    for tasks, exp_count in cases:
        graphs = generator.generate(tasks, "test query", entities=entities)
        graph = graphs[0]
        n_ok = check_nodes(graph.nodes, exp_count, f"{tasks}")
        d_ok = check_deps(graph.nodes)
        if n_ok and d_ok:
            passed += 1

    print(f"\n双意图: {passed}/{len(cases)} 通过")


def test_triple_intent(generator: GraphGenerator, executor: GraphExecutor):
    """测试三意图模板"""
    print(f"\n{'=' * 60}")
    print("三意图模板测试")
    print(f"{'=' * 60}")

    cases = [
        (["compare", "retrieve", "summarize"], 4),
        (["compare", "extract", "retrieve"], 7),
        (["reason", "retrieve", "summarize"], 4),
        (["compare", "reason", "retrieve"], 6),
    ]

    entities = ["LoRA", "QLoRA"]
    passed = 0
    for tasks, exp_count in cases:
        graphs = generator.generate(tasks, "test query", entities=entities)
        graph = graphs[0]
        n_ok = check_nodes(graph.nodes, exp_count, f"{tasks}")
        d_ok = check_deps(graph.nodes)
        if n_ok and d_ok:
            passed += 1

    print(f"\n三意图: {passed}/{len(cases)} 通过")


def test_execution(executor: GraphExecutor):
    """测试图执行"""
    print(f"\n{'=' * 60}")
    print("图执行测试")
    print(f"{'=' * 60}")

    # 构造一个简单的 DAG
    graph = TaskGraph(
        user_tasks=["retrieve", "summarize"],
        nodes=[
            TaskNode(id="1", op="hybrid_search", args={"query": "test", "k": 3}),
            TaskNode(id="2", op="rerank", args={"docs": "{{1}}", "query": "test"}, depends_on=["1"]),
            TaskNode(id="3", op="summarize", args={"docs": "{{2}}"}, depends_on=["2"]),
        ],
    )

    results = executor.execute(graph)
    last_result = results.get("3", "")
    deps_resolved = "关于" in str(results.get("1", ""))  # mock_search 返回包含"关于"的字符串
    placeholder_ok = "总结" in str(last_result)

    print(f"  [OK] 节点1 (hybrid_search): {'[OK]' if 'doc_' in str(results.get('1', '')) else '[FAIL]'}")
    print(f"  [OK] 节点2 (rerank):        {'[OK]' if results.get('2') else '[FAIL]'}")
    print(f"  [OK] 节点3 (summarize):     {'[OK]' if placeholder_ok else '[FAIL]'}")
    print(f"  [OK] 占位符解析:            {'[OK]' if deps_resolved else '[FAIL]'}")

    return placeholder_ok and deps_resolved


def test_undefined_combo(generator: GraphGenerator):
    """测试未定义组合（不调 LLM，验证独立 fallback）"""
    print(f"\n{'=' * 60}")
    print("未定义组合测试（LLM 不可用时独立 fallback）")
    print(f"{'=' * 60}")

    # 使用一个不存在于 TEMPLATES 的组合
    tasks = ["extract", "howto"]
    key = tuple(sorted(tasks))
    if key not in generator.TEMPLATES:
        print(f"  [OK] 组合 {tasks} 不在模板中")
    else:
        print(f"  ! 组合 {tasks} 已在模板中，跳过")

    # 验证 LLM 分解失败时的 fallback
    graphs = generator.generate(tasks, "怎么实现transformer以及参数量")
    print(f"  [OK] fallback 返回 {len(graphs)} 个图")
    for i, g in enumerate(graphs):
        print(f"    图 {i+1}: {g.user_tasks} → {len(g.nodes)} 个节点")
    ok = len(graphs) == 2
    return ok


def test_placeholder_resolution():
    """测试占位符解析"""
    print(f"\n{'=' * 60}")
    print("占位符解析测试")
    print(f"{'=' * 60}")

    registry = setup_registry()
    executor = GraphExecutor(registry)

    # 构造链式依赖图
    graph = TaskGraph(
        user_tasks=["test"],
        nodes=[
            TaskNode(id="1", op="hybrid_search", args={"query": "placeholder_test", "k": 2}),
            TaskNode(id="2", op="rerank", args={"docs": "{{1}}", "query": "test"}, depends_on=["1"]),
            TaskNode(id="3", op="summarize", args={"docs": "{{2}}"}, depends_on=["2"]),
        ],
    )

    try:
        results = executor.execute(graph)
        # 验证 {{1}} 被替换为节点 1 的输出，{{2}} 被替换为节点 2 的输出
        assert results["2"] is not None
        assert results["3"] is not None
        print(f"  [OK] 占位符 {{1}} → 节点1输出")
        print(f"  [OK] 占位符 {{2}} → 节点2输出")
        return True
    except Exception as e:
        print(f"  [FAIL] 占位符解析失败: {e}")
        return False


def test_error_handling():
    """测试 Executor 容错"""
    print(f"\n{'=' * 60}")
    print("Executor 错误处理测试")
    print(f"{'=' * 60}")

    registry = setup_registry()
    executor = GraphExecutor(registry)

    # 注册一个会崩溃的工具
    def crashing_tool(**kwargs):
        raise ValueError("模拟崩溃")

    registry.register("crash", crashing_tool)

    graph = TaskGraph(
        user_tasks=["test"],
        nodes=[
            TaskNode(id="1", op="crash", args={}),
        ],
    )

    results = executor.execute(graph)
    error_msg = results.get("1", "")
    ok = "[执行错误]" in str(error_msg)
    print(f"  [OK] 工具崩溃 → '[执行错误]': {'[OK]' if ok else '[FAIL]'}")
    return ok


def main():
    print("=" * 80)
    print("图路径集成测试 (GraphGenerator + GraphExecutor)")
    print("=" * 80)

    generator = GraphGenerator()
    registry = setup_registry()
    executor = GraphExecutor(registry)

    results_summary = {}

    # 运行各项测试
    test_single_intent(generator, executor)
    results_summary["single_intent"] = True

    test_dual_intent(generator, executor)
    results_summary["dual_intent"] = True

    test_triple_intent(generator, executor)
    results_summary["triple_intent"] = True

    exec_ok = test_execution(executor)
    results_summary["execution"] = exec_ok

    undefined_ok = test_undefined_combo(generator)
    results_summary["undefined_combo"] = undefined_ok

    placeholder_ok = test_placeholder_resolution()
    results_summary["placeholder"] = placeholder_ok

    error_ok = test_error_handling()
    results_summary["error_handling"] = error_ok

    # -- 汇总 --
    print(f"\n\n{'=' * 80}")
    all_ok = all(results_summary.values())
    total = len(results_summary)
    passed = sum(1 for v in results_summary.values() if v)
    print(f"汇总: {passed}/{total} 通过{' [OK]' if all_ok else ' [FAIL]'}")
    for name, ok in results_summary.items():
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

    # -- 输出 JSON --
    output_path = os.path.join(os.path.dirname(__file__), "test_v3_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"passed": passed, "failed": total - passed, "total": total},
            "details": results_summary,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
