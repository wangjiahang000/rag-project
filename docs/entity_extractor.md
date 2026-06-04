# EntityExtractor — 实体抽取引擎

## 概述

`EntityExtractor` 是基于 jieba + 技术词典 + 质量分闭环的实体抽取引擎。支持三级匹配（技术词典 → 正则 → jieba 未登录词）、自动重载、候选词发现与质量评分。

## 架构

```
输入文本
    │
    ▼
┌─────────────────┐
│ _check_reload    │  检查文件 mtime，变更则重载
└─────────┬───────┘
          ▼
┌─────────────────┐
│ jieba.posseg    │  分词 + 词性标注
└─────────┬───────┘
          ▼
┌─────────────────────────────┐
│ _merge_entities 三级匹配      │
│                              │
│  第 1 遍：技术词典 (TECH)     │  字母边界感知正则
│  ~~~~~~~~~~~~~~~~~~~~~~~~    │  最长词优先
│                              │
│  第 2 遍：正则 (REGEX)       │  版本号/URL/邮箱
│  ~~~~~~~~~~~~~~~~~~~~~~~~    │
│                              │
│  第 3 遍：jieba 未覆盖 (UNKNOWN) │  过滤黑名单/词性/长度
│  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~    │  异步候选词统计
│                                  │  质量分计算
└─────────┬─────────────────────┘
          ▼
    实体列表: [{"text": "cnn", "type": "TECH", "start": 0, "end": 3}, ...]
```

## 文件位置

`content_core/tools/process/entity_extractor.py`，约 420 行。

---

## 详细设计

### 1. 持久化文件

| 文件 | 路径 | 格式 | 用途 |
|------|------|------|------|
| 技术词典 | `memory/tech_dict.txt` | `词 词频` | 453 个技术实体，人工维护 |
| 候选词 | `memory/candidates.txt` | TSV | 自动发现，质量分≥0.55 可准入 |
| 黑名单 | `memory/blacklist.txt` | 每行一词 | 297 个停用词，jieda.del_word + 输出过滤 |

### 2. 初始化

#### `__init__(memory_dir)`

```
1. _init_jieba()      → load_userdict + del_word
2. _load_all()        → 读 tech_dict/blacklist 到内存
3. _build_bg_freq()   → 从 tech_dict 构建单字/二元频次表
```

#### `_init_jieba()`

- 调用 `jieba.load_userdict(tech_dict.txt)` 注册技术词
- 调用 `jieba.del_word()` 处理黑名单词
- 保存 `jieba` 和 `jieba.posseg` 引用

#### `_build_bg_freq()`

从 tech_dict 构建字符级频率统计，用于 PMI 计算：

```python
for word in tech_dict:
    for ch in word:
        _char_freq[ch] += freq
    for bigram in word:
        _bigram_freq[bigram] += freq
```

英文数字字符加拉普拉斯平滑（`+1`），避免 log(0)。

### 3. 自动重载

#### `_check_reload()`

每次 `extract()` 调用时检查：

1. 比较 `tech_dict.txt / candidates.txt / blacklist.txt` 的 mtime
2. 任一文件变更 → 重新初始化 jieba + 重载所有词典 + 重建背景频率表
3. 变更检测精准到文件级别，非轮询

### 4. 实体匹配逻辑

#### `extract(text)` → `_merge_entities(words, text)`

三级匹配：

#### 第 1 遍：技术词典（TECH）

```
sorted_dict = sorted(tech_dict.keys(), key=len, reverse=True)
for term in sorted_dict:
    pattern = re.compile(
        r'(?<![a-z])' + re.escape(term) + r'(?![a-z])',
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        # 未被覆盖 → 标记为 TECH
```

**设计要点**：

- **字母边界正则** `(?<![a-z])` + `(?![a-z])` 而非 `\b`：
  - `\b` 在 Python re 中会将中文字符视为 `\w`，导致 CJK-英文混合文本无法正确匹配
  - 字母边界正则只检查前后是否紧邻英文字母，CJK 字符不干扰
  - 解决了 `"ann"` 误配 `"Yann"` 的子串吞没问题
- **最长词优先**：按词长降序匹配，避免短词抢占长词
- **大小写不敏感**：`re.IGNORECASE`，同时维护 `cand_key` 的原始大小写
- **覆盖标记**：匹配的区域标记 `covered = True`，后续跳过

#### 第 2 遍：正则（REGEX）

```python
patterns = [
    (r'\b[a-zA-Z]+\d+(?:\.\d+)+\b', 'REGEX'),  # 版本号 v1.2.3
    (r'https?://\S+', 'REGEX'),                  # URL
    (r'[\w.+-]+@[\w.-]+\.\w+', 'REGEX'),        # 邮箱
]
```

#### 第 3 遍：jieba 未覆盖（UNKNOWN）

处理 jieba 分词结果中未被前两遍覆盖的词：

```python
for word_obj in words:
    word = word_obj.word
    flag = word_obj.flag

    # 过滤条件（任一满足则跳过）：
    #   1. 在黑名单中
    #   2. 长度 < 2
    #   3. 纯数字/空格/符号
    #   4. 已被前面覆盖
    #   5. 词性不在白名单中

    valid_pos = ('n', 'v', 'a', 'eng', 'x', 'nz', 'vn', 'an', 'ns', 'nt', 'nrt', 'nr')
```

**词性白名单**：名词、动词、形容词、英文、未知、专有名词等。

### 5. 候选词发现

#### `_update_candidates(word, count)`

当 UNKNOWN 实体累计出现 ≥3 次，计算质量分并写入 `candidates.txt`：

**写入格式**：`词\t出现次数\t质量分\t最近命中日期`

**准入线**：质量分 ≥ 0.55

**文件维护**：保留历史候选词记录，质量分随出现次数递增后可准入。

### 6. 质量分计算

#### `_compute_quality_score(word, count)`

```
综合分 = 0.3 × 频次分 + 0.4 × 凝固度 + 0.6 × 技术相关性
```

#### 6.1 频次分 (weight=0.3)

```
score_freq = min(count / 10, 1.0)
```

出现 10 次以上满分。

#### 6.2 凝固度/PMI (weight=0.4)

`_compute_pmi(word)`

计算相邻字符对的最小 PMI，衡量词内部的语素粘合程度：

```
for each bigram (ch1, ch2) in word:
    P(ch1) = _char_freq[ch1] / total_chars
    P(ch2) = _char_freq[ch2] / total_chars
    P(bg)  = _bigram_freq[bigram] / total_bigrams
    PMI = log(P(bg) / (P(ch1) * P(ch2)))

score_pmi = min(min(PMI_values) / 8.0, 1.0)
```

**背景频率**来自 tech_dict 的字符级统计。PMI 满分阈值 8.0。

**特点**：英文人名（如 Kaiming）PMI 可能为负，因为 tech_dict 不包含英文人名的 n-gram 模式，这是已知限制。

#### 6.3 技术相关性 (weight=0.3)

`_compute_tech_similarity(word)`

与 tech_dict 中最长公共子串的比例：

```
score_tech = max(LCS(word, term) for term in tech_dict) / len(word)
```

小于 2 或自身长度 < 2 则返回 0。

### 7. 全局接口

提供延迟初始化的全局单例：

```python
_global_extractor: Optional[EntityExtractor] = None

def get_extractor(memory_dir="memory") -> EntityExtractor:
    if _global_extractor is None:
        _global_extractor = EntityExtractor(memory_dir)
    return _global_extractor

def extract_entities(text: str) -> List[str]:
    return get_extractor().extract_entities(text)
```

---

## 质量分解读

| 维度 | 权重 | 公式 | 满分条件 |
|:----:|:----:|:----|:--------:|
| 频次分 | 0.3 | `min(count/10, 1.0)` | 出现 ≥10 次 |
| 凝聚分 | 0.4 | `min(min_PMI/8.0, 1.0)` | PMI ≥ 8.0 |
| 技术分 | 0.3 | `max(LCS/len(word))` | 完全包含于某技术词 |

准入阈值：**综合分 ≥ 0.55** + 累计出现 ≥ 3 次。

---

## 关键设计决策

1. **三级匹配替代单模型** — 无需 GPU，无网络依赖，离线可用，可调试
2. **字母边界正则替代 `\b`** — 解决中英文混合文本的边界检测问题
3. **技术词典字符频率做 PMI 背景** — 轻量、领域相关，无需大型语料
4. **文件 mtime 自动重载** — 热更新词典，无需重启服务
5. **候选词闭环** — 自动发现 → 质量打分 → 人工审查 → 录入词典
6. **大小写不敏感** — tech_dict 统一小写存储，匹配时 re.IGNORECASE
