# GraphGenerator — 模板化任务图生成

## 概述

`GraphGenerator` 将意图组合映射为可执行的 DAG 任务图（TaskGraph）。核心设计原则：**已知组合走模板（零延迟），未知组合 LLM 分解后走模板**。

## 架构

```
    [意图组合] 如 ["compare", "reason"]
          │
          ▼
┌───────────────────────────┐
│ TEMPLATES 查找              │  已知 26 种合法组合
│ key = sorted(tasks)        │  O(1) 查表
└─────────┬─────────────────┘
          │
     ┌────┴────┐
     ▼         ▼
   存在        不存在
     │         │
     ▼         ▼
  走模板方法   LLM 分解 (_llm_decompose)
     │          │ 拆成多个合法子问题
     │          │ 各子问题再走模板
     │          ▼
     │     ┌──────────┐
     │     │ 合法子图列表 │
     │     └──────────┘
     │          │
     └────┬─────┘
          ▼
    List[TaskGraph]
          │
          ▼
    GraphExecutor 执行
```

## 文件位置

`content_core/graph_generator.py`，约 350 行。

---

## 详细设计

### 1. 模板映射表

```python
TEMPLATES = {
    # ── 单意图 ──
    ("retrieve",): "_simple_search",
    ("compare",): "_build_compare",
    ("summarize",): "_build_summarize",
    ("howto",): "_simple_search",
    ("reason",): "_build_reason",
    ("extract",): "_build_extract",

    # ── 双意图 ──
    ("retrieve", "summarize"): "_build_summarize",
    ("compare", "retrieve"): "_build_compare",
    ("extract", "retrieve"): "_build_extract",
    ("reason", "retrieve"): "_build_reason",
    ("howto", "retrieve"): "_simple_search",
    ("compare", "reason"): "_build_compare_reason",
    ("compare", "extract"): "_build_extract_compare",

    # ── 三意图 ──
    ("compare", "retrieve", "summarize"): "_build_summarize_compare",
    ("compare", "extract", "retrieve"): "_build_extract_compare",
    ("reason", "retrieve", "summarize"): "_build_reason_summarize",
    ("compare", "reason", "retrieve"): "_build_compare_reason",
}
```

`retrieve` 被其他意图吸收（作为 hybrid_search 的基础步骤），不单独出现。

### 2. 基础工具链模板

#### `_simple_search(query, k)`

最简单的链路，用于 retrieve 或单意图降级：

```
hybrid_search → rerank
```

#### `_search_rerank_then(tool, query, k, extra_args)`

通用三节点模板，追加任意工具到搜索链路后：

```
hybrid_search → rerank → {tool}
```

用于 summarize、reason、extract 的单意图场景。

#### `_compare_chain(query, entities)`

核心对比链路：

```
hybrid_search(e0) ─→ rerank ─→ compare
hybrid_search(e1) ─→ rerank ─→ compare
```

每个实体独立搜索 → 各自重排序 → 合并对比。

### 3. 实体处理

#### `_ensure_entities(query, entities)`

确保至少有 2 个实体用于对比。逻辑：

1. 如果已有 ≥2 个 tech_dict 实体 → 直接返回
2. 不足时调用 `tech_extract_entities(query)` 从查询中抽取所有实体
3. 返回实体列表（可能含 UNKNOWN 类型）

#### `_extract_context(query, exclude)`

从查询中提取 UNKNOWN 实体作为搜索的领域上下文，排除已作为搜索词的实体。

例如 `"对比一下cnn与rnn在计算机视觉方面表现"` → context = `"计算机 视觉"` → 搜索变成 `"cnn 计算机 视觉" / "rnn 计算机 视觉"`

#### `_is_person_name(word)`

启发式人名判断：

```python
bool(re.match(r'^[A-Z][a-z]+$', word))
```

严格的首字母大写 + 后续全小写的纯字母模式，与 tech_dict 配合：在 tech_dict 中的词（transformer、attention 等）即使匹配此模式也被视为技术词。

#### `_search_query(entity, context)`

构造搜索词：

| 实体类型 | 行为 | 示例 |
|---------|------|------|
| tech_dict 命中 | `"{entity} {context}"` | `"cnn 计算机 视觉"` |
| 人名 | `"{entity} 论文"` | `"Yann 论文"` |
| 其他 | `entity` | `"angle"` |

#### `_build_entity_node(node_id, entity, context, k)`

为单个实体构建搜索节点：

1. 如果是人名 → 调用 `metadata_lookup(entity)` 查元数据索引
2. 索引命中 → 创建 `metadata_search` 节点（直接使用索引结果）
3. 索引未命中 → `hybrid_search` 节点

### 4. 组合构建器

#### 4.1 双意图

| 方法 | 意图组合 | 生成链路 |
|------|---------|---------|
| `_build_compare_reason` | compare + reason | compare 链路 + 追加 reason |
| `_build_extract_compare` | compare + extract | search(e0)→rerank→extract(e0), search(e1)→rerank→extract(e1), compare |

#### 4.2 三意图

| 方法 | 意图组合 | 生成链路 |
|------|---------|---------|
| `_build_summarize_compare` | summarize + retrieve + compare | summarize 链路 + 追加 compare |
| `_build_reason_summarize` | reason + retrieve + summarize | reason 链路 + 追加 summarize |

### 5. LLM 问题分解

#### `_llm_decompose(query, user_tasks)`

当意图组合不在 TEMPLATES 中时调用。将 undefined 组合拆解为多个合法的子问题。

**prompt 设计**：包含合法组合列表、原始问题、原始意图，要求返回 JSON 数组。

**验证**：检查每个子问题的 `user_tasks` 在 `TEMPLATES` 中，丢弃重复和无效组合。

**错误处理**：

| 场景 | 行为 |
|------|------|
| LLM 全部重试失败 | 各意图独立成单意图图 |
| 返回非法组合 | 非法子图丢弃 |
| 返回空数组 | 降级到单意图回退 |
| 返回未拆分（与输入相同组合） | 丢弃（无意义） |

### 6. 主入口

#### `generate(user_tasks, query, entities, complexity)`

```python
def generate(self, user_tasks, query, entities=None, complexity="single_step"):
    key = tuple(sorted(user_tasks))

    # 1. 已知模板 → 单图
    if key in TEMPLATES:
        method = getattr(self, TEMPLATES[key])
        nodes = method(query, entities)
        return [TaskGraph(user_tasks=user_tasks, nodes=nodes)]

    # 2. 未知组合 → LLM 分解
    llm_graphs = self._llm_decompose(query, user_tasks)
    if llm_graphs:
        return llm_graphs

    # 3. LLM 失败 → 各意图独立
    return [单个意图的 TaskGraph 列表]
```

---

## 关键设计决策

1. **离线模板优先** — 26 种合法组合 O(1) 查表，已知请求零延迟
2. **LLM 只做分解，不做规划** — 符合"模板驱动"设计，LLM 只说"这是个什么问题"而非"怎么解决"
3. **DAG 按序执行** — 每个 `TaskNode` 有 `depends_on` 列表，`GraphExecutor` 按拓扑顺序执行
4. **实体类别敏感** — 人名走元数据索引（如果实现），技术词走全文搜索，搜索词构造不同
5. **领域上下文自动提取** — UNKNOWN 实体作为搜索上下文，提升检索精度

---

## 搜索链路示例

**输入：** `"对比一下cnn与rnn在计算机视觉方面表现"`

**路由结果：** `["compare", "retrieve"]` → key = `("compare", "retrieve")` → TEMPLATES 命中 `_build_compare`

**生成图：**

| ID | OP | Args |
|:---:|:---:|:---:|
| 1 | `hybrid_search` | `query: "cnn 计算机 视觉"` |
| 2 | `hybrid_search` | `query: "rnn 计算机 视觉"` |
| 3 | `rerank` | `docs: {{1}}`, `query: 原查询` |
| 4 | `rerank` | `docs: {{2}}`, `query: 原查询` |
| 5 | `compare` | `docs_a: {{3}}`, `docs_b: {{4}}` |
