# TaskRouter 三层路由架构

```mermaid
flowchart TD
    Q[用户查询] --> A{闲聊检测}
    A -->|是| C[返回 chitchat]
    A -->|否| B1[第一层: 关键词规则]

    B1 --> R[规则信号 rule_signals<br/>关键词匹配, 提供意图信号]
    R --> B2[第二层: SBERT 语义分类]

    B2 --> S[语义分数 sbert_scores<br/>锚点对比 + 负例校准<br/>6 个意图独立打分]

    S --> F[融合决策]
    R --> F

    F --> D{硬逻辑解耦规则}
    D --> T[排序 → 阈值筛选 → 最多 3 个 tasks]

    T --> E{LLM 兜底条件}
    E -->|not tasks| L[第三层: LLM 兜底<br/>DeepSeek Chat]
    E -->|top_conf < 0.60| L
    E -->|keyword_conflict| L
    E -->|否| O1[sbert 或 rule+sbert]

    L --> O2[llm 来源]

    O1 --> OUT[最终路由结果: tasks + resource_hint + sources]
    O2 --> OUT
```

## 三层职责

| 层 | 方法 | 职责 | 输出 |
|----|------|------|------|
| 第一层 | `_rule_match()` | 关键词正则匹配，提供意图信号（不做最终决策） | `rule_signals: List[str]` |
| 第二层 | `_sbert_match()` | 锚点语义对比 + 负例校准，6 个意图独立打分 | `sbert_scores: Dict[str, float]` |
| 融合 | `route()` | 规则信号平滑提权 + 硬逻辑解耦 + 差异化阈值筛选 | `tasks: List[str]` |
| 第三层 | `_llm_fallback()` | DeepSeek Chat 兜底裁决（仅在不确定时调用） | `tasks: List[str]` |

## 融合决策流程

```
规则信号 → 关键词提权(0.27-0.50 → 0.30-0.65)
       → 差异化阈值过滤(retrieve:0.30, summarize:0.50, ...)
       → 硬逻辑解耦(howto/summarize互斥, extract/retrieve消歧)
       → 关键词优先级排序(reason不被howto反超)
       → LLM触发判断
```

## LLM 触发条件（任一）

```python
if not tasks         # 无意图过阈值
   or top_conf < 0.60  # 最高分太低
   or keyword_conflict: # 规则和语义打架
```

## 来源标记

| source | 含义 | 占比 |
|--------|------|------|
| sbert | 仅靠语义分类 | ~50% |
| rule+sbert | 规则+语义融合 | ~45% |
| llm | LLM 兜底裁决 | ~5% |
