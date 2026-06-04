# MyRAG — 学术论文智能问答系统

基于 RAG（Retrieval-Augmented Generation）的学术论文智能问答系统。支持上传 PDF/TXT 文档，通过意图路由 + 混合检索 + LLM 生成实现带引用的学术问答。

---

## 目录

1. [技术栈](#1-技术栈)
2. [系统架构](#2-系统架构)
3. [请求流水线](#3-请求流水线)
4. [核心模块详解](#4-核心模块详解)
5. [数据流图](#5-数据流图)
6. [部署指南](#6-部署指南)
7. [API 文档](#7-api-文档)
8. [评估指标](#8-评估指标)

---

## 1. 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **后端框架** | Python 3.12 + FastAPI | 异步 Web 框架 |
| **API 网关** | Uvicorn | ASGI 服务器 |
| **向量数据库** | ChromaDB | HNSW 近似最近邻搜索 |
| **嵌入模型** | BAAI/bge-small-zh-v1.5 | 中文嵌入（384 维） |
| **语义分类** | BAAI/bge-base-zh-v1.5 | SBERT 零样本分类 |
| **稀疏检索** | BM25Okapi (rank_bm25) | 关键词检索，jieba 分词 |
| **重排序** | BAAI/bge-reranker-v2-m3 (可选) | Cross-Encoder 重排序 |
| **LLM** | DeepSeek Chat (via litellm) | 意图兜底 + 答案生成 + 工具调用 |
| **前端** | React 18 + Vite 5 | SPA 单页应用 |
| **Markdown** | react-markdown + remark-gfm | 流式 Markdown 渲染 |
| **PDF 解析** | pdfplumber | PDF 文本与表格提取 |
| **ONNX 推理** | onnxruntime (可选) | INT8 量化加速 |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        前端 React SPA                               │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────────┐  │
│  │ 问答界面  │  │ 用户画像  │  │ 系统监控   │  │ 会话管理          │  │
│  │ Chat.jsx │  │Profile.jsx│  │ Stats.jsx │  │ localStorage UUID │  │
│  └────┬─────┘  └──────────┘  └───────────┘  └───────────────────┘  │
│       │ SSE 流式 / JSON                                           │
└───────┼─────────────────────────────────────────────────────────────┘
        │ HTTP
┌───────┼─────────────────────────────────────────────────────────────┐
│       ▼                     后端 FastAPI                            │
│  ┌──────────┐  ┌─────────────────┐  ┌───────────────────────────┐  │
│  │ 中间件栈  │  │   /chat          │  │  /chat/stream (SSE)      │  │
│  │ CORS     │  │  非流式问答       │  │  Server-Sent Events     │  │
│  │ Metrics  │  │  缓存 + 画像     │  │  逐 token 推送          │  │
│  │ Timing   │  │  会话保存        │  │  最后追加参考文献       │  │
│  └──────────┘  └────────┬────────┘  └───────────┬───────────────┘  │
│                         │                       │                  │
│                         ▼                       ▼                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  核心 RAG 引擎                                │  │
│  │                                                              │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐                │  │
│  │  │ 查询缓存  │   │ 指代消解  │   │ 意图路由  │                │  │
│  │  │ LRU 128  │   │ LLM 还原  │   │ 规则→SBERT→LLM            │  │
│  │  │ SHA256   │   │ 代词→名词 │   │ 6 意图融合决策            │  │
│  │  └──────────┘   └──────────┘   └─────┬────┘                │  │
│  │                                      │                      │  │
│  │                                      ▼                      │  │
│  │  ┌──────────────────────────────────────────┐               │  │
│  │  │  DAG 图生成 + 并行执行                    │               │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │               │  │
│  │  │  │ 模板匹配  │→ │ 拓扑分层  │→ │ 并行执行│ │               │  │
│  │  │  │ LLM 分解  │  │ 依赖排序  │  │ 4线程池 │ │               │  │
│  │  │  └──────────┘  └──────────┘  └────────┘ │               │  │
│  │  └──────────────────────────────────────────┘               │  │
│  │                                      │                      │  │
│  │                                      ▼                      │  │
│  │  ┌──────────────────────────────────────────┐               │  │
│  │  │  增强检索                                │               │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │               │  │
│  │  │  │ 混合搜索  │→ │ 实体提权  │→ │ 时间衰减│ │               │  │
│  │  │  │ 0.7+0.3  │  │ ×1.2 max │  │ e^{-λt}│ │               │  │
│  │  │  └──────────┘  └──────────┘  └────────┘ │               │  │
│  │  └──────────────────────────────────────────┘               │  │
│  │                                      │                      │  │
│  │                                      ▼                      │  │
│  │  ┌──────────────────────────────┐                          │  │
│  │  │  LLM 生成（带引用）           │                          │  │
│  │  │  上下文注入 + 历史 + 引用格式  │                          │  │
│  │  │  Semaphore(5) 限流           │                          │  │
│  │  │  指数退避重试                │                          │  │
│  │  └──────────────┬───────────────┘                          │  │
│  │                 ▼                                          │  │
│  │  ┌──────────────────────────────┐                          │  │
│  │  │  保存会话 + 写缓存 + 打指标   │                          │  │
│  │  └──────────────────────────────┘                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─────────────┐  ┌──────────┐  ┌────────────┐               │
│  │ ChromaDB    │  │ BM25     │  │ 会话管理器  │               │
│  │ 向量索引     │  │ 关键词索引 │  │ 1000上限    │               │
│  │ HNSW        │  │ 内存缓存  │  │ 30min TTL  │               │
│  └─────────────┘  └──────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 请求流水线

### 3.1 标准问答流程

```
用户输入
    │
    ▼
┌─────────────────┐
│  1. 查询缓存     │ ←─── LRU OrderedDict，SHA256 归一化 key
│     L1: 内存     │      容量 128，TTL 3600s
└────────┬────────┘
    │ 未命中
    ▼
┌─────────────────┐
│  2. 会话管理     │ ←─── SessionManager.get_or_create(session_id)
│     获取历史     │      最近 6 轮对话
└────────┬────────┘
    │
    ▼
┌─────────────────┐
│  3. 指代消解     │ ←─── 仅含 "它/这/那/该/其/它们/这些/那些"
│     (条件触发)   │      LLM 将代词还原为历史具体名词
└────────┬────────┘
    │
    ▼
┌─────────────────┐
│  4. 三层意图路由  │
│                  │
│  ┌────────────┐  │
│  │ ① 关键词规则 │─── 正则匹配 6 类意图关键词
│  └──────┬─────┘  │
│         ▼        │
│  ┌────────────┐  │
│  │ ② SBERT语义 │─── 锚点零样本分类，余弦相似度
│  └──────┬─────┘  │     幂次缩放 + 差异化阈值
│         ▼        │
│  ┌────────────┐  │
│  │ ③ 融合决策  │─── 关键词平滑提权 + 硬逻辑解耦
│  └──────┬─────┘  │     白名单校验 + LLM 兜底
└────────┬────────┘
    │  user_tasks, entities
    ▼
┌─────────────────┐
│  5. DAG 图生成   │ ←─── 意图组合查模板表
│     模板/LLM     │      未定义组合由 LLM 拆解
└────────┬────────┘
    │  TaskGraph
    ▼
┌─────────────────┐
│  6. DAG 并行执行  │ ←─── 拓扑分层排序
│                  │      ThreadPoolExecutor(4)
│  ┌────────────┐  │      同层无依赖节点并行
│  │ hybrid_search│ │
│  │ rerank       │ │
│  │ compare     │ │
│  │ summarize   │ │
│  │ extract     │ │
│  │ reason      │ │
│  └────────────┘  │
└────────┬────────┘
    │ 检索结果
    ▼
┌─────────────────┐
│  7. 增强检索     │ ←─── 实体软加权 (×1.2)
│     上下文构建    │      时间指数衰减 (λ=0.1)
│                  │      结构化上下文 + 引用编号
└────────┬────────┘
    │ context, citations
    ▼
┌─────────────────┐
│  8. LLM 生成     │ ←─── 带引用格式的学术回答
│     限流 + 重试   │      Semaphore(5)，指数退避
└────────┬────────┘
    │ answer
    ▼
┌─────────────────┐
│  9. 保存会话     │ ←─── 记录 Turn + 更新画像
│     写缓存       │      写入 LRU Cache
│     打指标       │      inc("llm_calls") 等
└────────┬────────┘
    │
    ▼
 返回 ChatResponse (JSON / SSE Stream)
```

### 3.2 SSE 流式输出格式

```
data: {"type": "token", "data": "基于"}       ← 逐 token 输出
data: {"type": "token", "data": "检索"}       
data: {"type": "token", "data": "增强"}       
...
data: {"type": "references", "data": "\n\n---\n\n参考文献：\n  [1] ..."}  ← 末尾附加参考文献
data: {"type": "done", "tasks": [...], "source": "sbert", "plan": [...], "citations": [...]}  ← 结束元数据
```

---

## 4. 核心模块详解

### 4.1 三层意图路由 (`content_core/task_router.py`)

| 层 | 方法 | 延迟 | 准确率 | 说明 |
|----|------|------|--------|------|
| L1 | 关键词正则 | <1ms | 中 | 6 类意图 (retrieve/compare/summarize/howto/reason/extract)，正则匹配 |
| L2 | SBERT 锚点分类 | ~50ms | 高 | BGE 零样本多标签，锚点嵌入持久化到 pickle，启动免计算 |
| L3 | DeepSeek LLM | ~1-3s | 最高 | 规则/语义冲突或无意图时调用 |

**融合决策流程**：
1. 关键词匹配结果对各意图做平滑提权（SBERT 0.27→0.30, 0.50→0.65）
2. 幂次缩放放大差异（默认指数 2.0）
3. 差异化阈值过滤（retrieve 0.65, 其余 0.70）
4. 硬逻辑解耦：suppress howto + summarize 同现（无教程关键词）、demote extract（无数量词）、suppress 非首轮 summarize
5. 关键词优先级排序：extract > compare > summarize > howto > reason
6. 合法组合白名单校验（单/双/三组合枚举）
7. 全空或置信度过低时 → LLM fallback

### 4.2 DAG 执行引擎 (`content_core/graph_generator.py` + `graph_executor.py`)

**模板覆盖**：5 种单意图 + 7 种双意图 + 4 种三意图组合 = 16 个离线模板

| 意图组合 | 模板 DAG |
|---------|---------|
| retrieve | `hybrid_search → rerank` |
| compare | `entity_extract → para_search[A] + para_search[B] → dual_rerank → compare` |
| summarize | `hybrid_search → rerank → summarize` |
| retrieve+summarize | `hybrid_search → rerank → summarize` |
| retrieve+compare | `entity_extract → para_search[A/B] → rerank → compare` |
| ... | ... |

**并行执行**：
- 拓扑分层：找出无入度节点组成 layer 0，移除后重复
- 每层节点通过 `ThreadPoolExecutor(max_workers=4)` 并发执行
- `_resolve_args()` 解析 `{{node_id}}` 引用，从上层结果注入

### 4.3 混合检索 (`content_core/data/vector_store.py`)

```
Score = 0.7 × ChromaDB (余弦相似度) + 0.3 × BM25 (归一化)
```

| 组件 | 细节 |
|------|------|
| **向量检索** | ChromaDB + BAAI/bge-small-zh-v1.5，384 维，余弦距离 |
| **稀疏检索** | BM25Okapi + jieba 分词，`_bm25_cache` 内存驻留 |
| **缓存机制** | 全局 `dict[str, tuple[mtime, size, BM25Okapi, list]]`，文件 mtime 变化时自动重载 |
| **增量索引** | 比对 ChromaDB metadata source 字段，跳过已索引文件 |
| **批处理** | ChromaDB add 每 5000 条 commit 一次 |

### 4.4 增强检索 (`content_core/retrieval/enhanced_search.py`)

```
最终得分 = 原始得分 × 实体提升因子 × 时间衰减因子

实体提升因子 = 1 + 0.2 × (匹配实体数 / 查询实体数)
时间衰减因子 = exp(-0.1 × |查询年份 - 文档年份|)
```

- 实体通过 `entity_extractor.py` 提取（jieba + 技术词典）
- 超过 20 年的文档被硬裁剪

### 4.5 LRU 查询缓存 (`backend/cache.py`)

```
层    实现       容量  TTL
────────────────────────────
L1    OrderedDict  128  3600s
L2    (预留 Redis)  —    —
```

- Key：`sha256(normalized_question)` 归一化
- Value：`json.dumps(ChatResponse)`
- LRU：`move_to_end` + `popitem(last=False)`
- 命中时 <10ms 返回，跳过完整流水线

### 4.6 会话与用户画像 (`backend/session.py`)

```
Session
├── history: list[Turn]          # 对话历史
├── interest_tags: dict[str, float]  # 兴趣标签 + 衰减权重
├── intent_history: dict[str, int]   # 意图频次统计
├── favorite_papers: list[str]       # 收藏论文
├── total_queries: int
└── last_active: timestamp
```

- `interest_tags` 更新：新实体 +1，每轮全体 ×0.98 衰减
- 画像可访问 `/profile/{session_id}` 查看
- 指代消解：`REFERENCE_PATTERNS` 正则匹配 "它/这/那/该/其/它们/这些/那些/上述/以上/这个/那个"

### 4.7 LLM 集成

| 参数 | 值 |
|------|------|
| 库 | litellm |
| 模型 | deepseek/deepseek-chat |
| 并发限制 | `threading.Semaphore(5)` + `asyncio.Semaphore(5)` |
| 重试 | 指数退避 `RETRY_DELAY × 2^attempt`，默认 3 次 |
| 超时 | 60s |
| 降级 | 全部失败返回"没有检索到相关文献" |
| 流式 | `litellm.acompletion(stream=True)` → SSE |

### 4.8 前端 (`frontend/`)

| 特性 | 实现 |
|------|------|
| 流式渲染 | fetch + ReadableStream + async generator → 逐 token setState |
| Markdown | react-markdown + remark-gfm |
| 会话 ID | 浏览器 localStorage 存储 UUID，支持重置 |
| 暗色主题 | CSS 变量驱动，响应式布局 |
| 三 tab | 问答 / 画像 / 监控 |

---

## 5. 数据流图

### 5.1 用户请求数据流

```
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 用户  │───▶│ 前端 SPA  │───▶│ FastAPI  │───▶│ 缓存检查  │
└──────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                  │ 未命中
                                                  ▼
                                          ┌──────────────┐
                                          │ 会话管理+消解  │
                                          └──────┬───────┘
                                                  ▼
                                          ┌──────────────┐
                                          │   意图路由    │
                                          └──────┬───────┘
                                                  ▼
                                          ┌──────────────┐
                                          │ DAG 生成+执行 │
                                          └──────┬───────┘
                                                  ▼
                                          ┌──────────────┐
                                          │   增强检索    │
                                          └──────┬───────┘
                                                  ▼
                                          ┌──────────────┐
                                          │   LLM 生成   │
                                          └──────┬───────┘
                                                  │
                    ┌──────────────────────────────┼──────┐
                    │                              │      │
                    ▼                              ▼      ▼
              ┌──────────┐              ┌────────────┐ ┌──────┐
              │ 写缓存    │              │ 保存会话+画像 │ │打指标│
              └──────────┘              └────────────┘ └──────┘
                    │
                    ▼
              ┌──────────┐
              │ 返回前端   │
              └──────────┘
```

### 5.2 索引构建数据流

```
PDF 文件 (data/)
    │
    ▼
pdfplumber 提取 + 噪声清洗
    │
    ▼
TXT 文件 (last/storage/papers/txt/)
    │
    ▼
chunker.py 结构化分块（按章节→段落→句子）
    │
    ▼
ChromaDB 向量索引 + BM25 pickle
    │
    ▼
chroma_data/ (持久化存储)
```

### 5.3 意图路由数据流

```
用户Query
    │
    ├──→ 关键词正则匹配 ──→ 6 意图分数 (0/1)
    │
    ├──→ SBERT 锚点分类 ──→ 6 意图分数 (0~1)
    │       │
    │       └──→ 加载锚点嵌入 (pickle 缓存)
    │
    ▼
融合决策引擎
    ├── 关键词平滑提权
    ├── 幂次缩放
    ├── 差异化阈值
    ├── 硬逻辑解耦
    ├── 关键字优先级排序
    └── 合法组合白名单校验
    │
    ├──→ 输出意图列表 + source 标签
    │
    └──→ 置信度过低 → LLM fallback → 意图列表
```

---

## 6. 部署指南

### 6.1 环境要求

- Python 3.12+
- Node.js 20+ (前端构建)
- 8GB+ RAM (建议)

### 6.2 安装步骤

```bash
# 1. 克隆项目
cd my_rag_project

# 2. 创建虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 配置环境变量
# 编辑 .env 文件，设置 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL

# 5. 下载嵌入模型
python scripts/download_model.py

# 6. (可选) 从 PDF 提取文本并索引
python scripts/reindex.py --extract-pdfs --clear

# 7. 安装前端依赖并构建
cd frontend
npm install
npm run build
cd ..

# 8. 启动服务
python run.py
# 或: uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 6.3 前端开发模式

```bash
# 终端 1: 启动后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 终端 2: 启动 Vite 开发服务器（热重载）
cd frontend
npm run dev
# → http://localhost:5173 (自动代理 API 到 8000)
```

### 6.4 索引管理

```bash
# 全量重建
python scripts/reindex.py --clear

# 增量索引（仅新文件）
python scripts/reindex.py --incremental

# 预览分块效果
python scripts/reindex.py --dry-run

# 从 PDF 提取 + 重建
python scripts/reindex.py --extract-pdfs --clear

# 指定分块策略
python scripts/reindex.py --strategy recursive  # 递归字符分块
python scripts/reindex.py --strategy structure  # 结构化分块（默认）
```

---

## 7. API 文档

| 端点 | 方法 | 说明 | Body/Params |
|------|------|------|-------------|
| `/health` | GET | 健康检查 | — |
| `/chat` | POST | 非流式问答 | `{"question": "...", "session_id": "..."}` |
| `/chat/stream` | POST | SSE 流式问答 | 同上 |
| `/profile/{session_id}` | GET | 获取用户画像 | — |
| `/profile/{session_id}` | DELETE | 清除用户数据 | — |
| `/profile/{session_id}/interests` | GET | 获取兴趣标签 | — |
| `/stats` | GET | 全局会话统计 | — |
| `/metrics` | GET | 进程内性能指标 | — |
| `/metrics/reset` | POST | 重置指标计数器 | — |

### ChatResponse 结构

```json
{
  "user_tasks": ["retrieve", "summarize"],
  "plan": [
    {"id": "1", "op": "hybrid_search", "args": {"query": "..."}},
    {"id": "2", "op": "rerank", "args": {"input": "{{1}}"}},
    {"id": "3", "op": "summarize", "args": {"input": "{{2}}"}}
  ],
  "answer": "根据检索到的文献... [1][2]  \n\n---\n\n参考文献：\n  [1] 2301.12345 (2023)\n  [2] 2305.67890 (2023)",
  "citations": [
    {"index": 1, "source": "2301.12345", "title": "...", "year": 2023},
    {"index": 2, "source": "2305.67890", "title": "...", "year": 2023}
  ],
  "source": "sbert"
}
```

---

## 8. 评估指标

### 8.1 评估命令

```bash
# 完整评估
python eval/run_evaluation.py --mode all

# 单项评估
python eval/run_evaluation.py --mode routing    # 意图路由准确率
python eval/run_evaluation.py --mode latency    # 响应延迟
python eval/run_evaluation.py --mode ablation   # 消融实验
```

### 8.2 关键指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 路由准确率 | >90% | 6 类意图分类正确率 |
| 端到端延迟 | <10s | 完整流水线（不含 LLM 生成） |
| 缓存命中率 | >30% | 重复查询走缓存 <10ms |
| DAG 并行加速比 | >1.5x | 相对串行执行的加速 |
| 检索召回率 | >85% | Top-10 命中相关文档 |

### 8.3 监控端点

访问 `/metrics` 获取运行时指标：

```json
{
  "uptime_seconds": 3600,
  "counters": {
    "http_requests_total:method=GET,path=/health,status=200": 100,
    "cache_hit:endpoint=chat": 30,
    "cache_miss:endpoint=chat": 70,
    "llm_calls:endpoint=chat": 50,
    "routing_source:sbert": 40,
    "routing_source:keyword": 20
  },
  "latencies": {
    "http_request_duration_ms:path=/chat": {
      "count": 50,
      "avg_ms": 3500,
      "max_ms": 8500,
      "p50_ms": 2800
    }
  }
}
```

---

## 版本信息

- **当前版本**: 2.0
- **更新日期**: 2026-05-29
- **许可证**: 内部项目
