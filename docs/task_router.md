# TaskRouter — 三级意图路由

## 概述

`TaskRouter` 是意图识别的核心组件，将用户自然语言查询映射为结构化意图标签（retrieve, compare, summarize, howto, reason, extract）。采用三级串联架构：**关键词规则提供信号 → SBERT 语义裁决 → LLM 兜底**，保证低延迟的同时提供高准确率。

## 架构

```
用户查询
    │
    ▼
┌─────────────────────────────┐
│ 1. 关键词规则匹配 (_rule_match) │  关键词命中
│    6 条正则，每条映射一个意图    │  作为信号分数
└──────────┬──────────────────┘
           │ 规则信号列表
           ▼
┌─────────────────────────────┐
│ 2. SBERT 语义分类            │  6 个意图独立打分
│    (_sbert_match)            │  正例 top-3 均值
│                               │  负例对比校准
└──────────┬──────────────────┘
           │ 6 维分数向量
           ▼
┌─────────────────────────────┐
│ 3. 分数融合与决策             │  幂次缩放拉大差距
│    · 自适应幂次缩放            │  关键词平滑提权
│    · 关键词平滑提权            │  硬逻辑解耦规则
│    · 5 条硬逻辑解耦规则        │  约束到合法组合
└──────────┬──────────────────┘
           │ 意图列表
           ▼
┌─────────────────────────────┐
│ 4. LLM 兜底（条件触发）        │  冲突/低置信/空结果
│    (LLM_fallback)            │  时调用 DeepSeek
└──────────┬──────────────────┘
           │ 最终意图
           ▼
       ["compare", "reason"]
```

## 文件位置

`content_core/task_router.py`，约 290 行。

---

## 详细设计

### 1. 关键词规则匹配

6 条正则表达式，覆盖所有意图。**不作为决策依据，只作为信号**，后续与 SBERT 分数融合。

```python
RULES = [
    (r"对比|区别|比较|vs|不同|差异|不一样|更|哪个", "compare"),
    (r"总结|汇总|概括|归纳|概述|理一理", "summarize"),
    (r"提取|抽取|获取|找出|查出|参数|指标|数值|多少", "extract"),
    (r"找|检索|搜|搜索|查|查找|什么是|是什么|定义|介绍", "retrieve"),
    (r"怎么|如何|步骤|教程|配置|部署|实现|导出|转换", "howto"),
    (r"为什么|原因|导致|背后|原理|造成", "reason"),
]
```

`extract` 排在 `retrieve` 之前避免"找出/查出"被 `retrieve` 的"找"截胡。

### 2. SBERT 语义分类

#### 2.1 模型

默认使用 `BAAI/bge-base-zh-v1.5`（BGE 中文 base 模型），支持 ONNX INT8 量化推理：

- ONNX 模式：`_ONNXEncoder` 用 onnxruntime CPU 推理，CLS token pooling + L2 normalize
- 原始模式：SentenceTransformer.encode()
- 自动检测：`bge_base_zh_int8.onnx` 存在则自动启用 ONNX 模式

#### 2.2 锚点分类机制

非传统分类器——无 softmax 层，而是预写正例锚点句和负例锚点句，通过语义相似度打分。

**正例锚点**：每个意图 11~23 条语义集中的句子。

```
retrieve: "帮我查一下关于这个主题的资料", "搜索一下相关的研究文献" ...
compare:  "比较一下A和B有什么区别", "对比这两种方法的不同之处" ...
summarize:"总结一下这个领域的研究进展", "概括这些文献的核心观点" ...
howto:    "如何实现这个功能", "怎么配置和部署这个系统" ...
reason:   "为什么会出现这种现象", "背后的原理和机制是什么" ...
extract:  "从这段话中提取具体的数值信息", "从论文里找出实验参数和指标" ...
```

**负例锚点**：每个意图 6 条，语义上接近但不属于该意图的句子。仅在正分 >0.65 时启用，以 0.30 比例扣分。

**打分公式**：

```
query_embedding = model.encode(query)
for each intent:
    pos_sims = cosine_similarity(query, positive_anchors)
    pos_score = top-3(pos_sims).mean()
    
    if pos_score > 0.65:
        neg_penalty = max(cosine_similarity(query, negative_anchors)) * 0.30
        score = max(0, pos_score - neg_penalty)
    else:
        score = pos_score
```

每个意图独立打分，互不影响，天然支持多意图输出。

### 3. 分数融合与决策

#### 3.1 自适应幂次缩放

SBERT 原始分数差距偏小，用幂次放大：

```python
norm = (scores - smin) / (smax - smin)       # 归一化到 [0, 1]
scaled = norm ** power                         # 幂次缩放（power=2.0）
result = smin + scaled * (1.0 - smin)          # 映射回 [smin, 1.0]
```

效果：top 意图拉大到 1.0，低分意图差距放大。

#### 3.2 关键词平滑提权

规则命中的意图，根据 SBERT 分数线性提权：

| SBERT 原始分 | 提权后 | 含义 |
|:---:|:---:|:---:|
| 0.27 | 0.30 | 刚过门槛，微抬 |
| 0.35 | 0.42 | 适度提权 |
| 0.50 | 0.65 | 满额提权 |

≤0.27 不触发，防止字面匹配误判。

#### 3.3 硬逻辑解耦规则

5 条解耦规则，处理语义混淆场景：

| 规则 | 条件 | 行为 |
|------|------|------|
| howto + summarize | 无教程关键词 | 移除 howto |
| extract + retrieve | 无关键词 + 无数值量词 | 移除 extract |
| retrieve + summarize | 介绍类 + 无归纳词 + retrieve 分更高 | 移除 summarize |
| 关键词优先级 | 关键词意图被混淆意图反超 | 交换顺序 |
| 分差过滤 | 非关键词意图远低于 #1 | 移除噪音意图 |

#### 3.4 合法组合约束

输出必须属于预定义的合法组合，非法组合降级：

- 单意图：`{retrieve, compare, summarize, howto, reason, extract}`
- 双意图：`{retrieve+summarize, retrieve+compare, retrieve+extract, retrieve+reason, retrieve+howto, compare+reason, extract+compare}`
- 三意图：`{retrieve+summarize+compare, retrieve+extract+compare, retrieve+reason+summarize, retrieve+compare+reason}`

超出组合 → 先尝试匹配双意图子集 → 否则只保留 #1。

### 4. LLM 兜底

**触发条件**（任一满足）：

1. 最终 tasks 为空
2. 最高置信度 ≤ 0.70
3. 关键词命中但未进 tasks（规则与语义冲突）

**实现**：调 DeepSeek Chat，temperature=0，带规则和语义分数作为上下文。最多重试 3 次，全部失败返回 `["retrieve"]`。

### 5. 闲聊检测

独立于路由管道，在最终 tasks 为空时触发。匹配预设的正则模式：

```python
["你是谁", "你叫什么", "你好|嗨|hello|hi", "谢谢|多谢|感谢", ...]
```

命中返回 `["chitchat"]`，下游不走搜索链路。

---

## 配置参数

定义在 `content_core/config.py`：

| 参数 | 值 | 说明 |
|------|:---:|------|
| `POWER_SCALE` | 2.0 | 幂次缩放指数 |
| `BOOST_RANGE` | (0.27, 0.50) | 关键词提权的 SBERT 区间 |
| `BOOST_TARGET` | (0.30, 0.65) | 提权后的目标区间 |
| `SCORE_THRESHOLDS` | 0.70 (所有意图) | 意图采纳阈值 |
| `LLM_FALLBACK_THRESHOLD` | 0.70 | LLM 兜底触发阈值 |
| `RELATIVE_MARGIN` | 0.05 | 分差过滤比例 |
| `LLM_RETRY_COUNT` | 2 | LLM 重试次数 |
| `LLM_TIMEOUT` | 30s | LLM 超时 |

## 关键设计决策

1. **关键词不作为决策依据，只作为信号** — 防止条件规则误判，语义分数起主导作用
2. **正负例锚点代替传统分类** — 无需标注数据，零样本迁移，支持多意图自然输出
3. **幂次缩放拉大差距** — 区分模棱两可和确定性意图
4. **LLM 只在冲突时介入** — 正常请求零延迟，保持低响应时间

---

## 数据流总结

```
用户输入 → _rule_match(信号) + _sbert_match(分数)
         → _adaptive_power_scale(拉大差距)
         → 关键词提权融合
         → 硬逻辑解耦(5条规则)
         → 合法组合约束
         → [条件] LLM 兜底
         → 输出: {"user_tasks": [...], "complexity": "...", "source": "..."}
```
