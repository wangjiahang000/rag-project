# 项目文件结构

```
my_rag_project/
│
├── .env                              # 环境变量（API Key 等）
├── .gitignore
├── requirements.txt                  # Python 依赖
├── README.txt
│
├── project_structure.md              # 本文件 — 项目结构
├── project_overview.md               # 项目说明文档
├── optimization_plan.md              # 优化方案文档
├── routing_architecture.md           # 路由架构设计
├── structure_model_diagram.md        # 结构模型图
├── remaining_plan.md                 # 剩余计划
│
├── chroma_data/                      # 向量数据库持久化
│   ├── chroma.sqlite3                # ChromaDB SQLite 元数据
│   ├── bm25.pkl                      # BM25 索引 pickle
│   ├── docs.pkl                      # 文档列表 pickle
│   └── 518ce314-.../                 # ChromaDB HNSW 索引文件
│
├── memory/                           # 实体提取器状态
│   ├── tech_dict.txt                 # 技术术语词典（~300+ 条）
│   ├── blacklist.txt                 # 停用词过滤表（~300 条）
│   └── candidates.txt                # 自动发现候选词（质量评分）
│
├── models/                           # 本地模型
│   └── bge-small-zh-v1.5/            # BGE 小模型中文嵌入
│
├── backend/                          # FastAPI 后端服务
│   ├── __init__.py
│   ├── main.py                       # 应用入口 + 中间件 + SPA 挂载
│   ├── schemas.py                    # Pydantic 数据模型
│   ├── cache.py                      # LRU 查询缓存（L1 内存）
│   ├── session.py                    # 会话管理 + 指代消解 + 用户画像
│   ├── metrics.py                    # 进程内指标收集器
│   ├── dependencies.py               # 依赖注入（单例管理）
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py                 # GET /health
│   │   ├── chat.py                   # POST /chat, POST /chat/stream
│   │   └── profile.py               # 用户画像 API
│   └── routes/                       # 旧版路由（已删除）
│
├── content_core/                     # 核心 RAG 引擎
│   ├── __init__.py
│   ├── config.py                     # 全局参数常量
│   ├── types.py                      # Pydantic 类型定义
│   ├── task_router.py                # 三层意图路由
│   ├── graph_generator.py            # DAG 图生成（模板 + LLM）
│   ├── graph_executor.py             # DAG 并行执行引擎
│   ├── data/
│   │   ├── vector_store.py           # 混合检索（ChromaDB + BM25）
│   │   └── chunker.py               # 结构化分块器
│   ├── models/
│   │   ├── sbert_classifier.py       # SBERT 锚点分类器
│   │   └── export_onnx.py            # ONNX INT8 导出
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── enhanced_search.py       # 实体提权 + 时间衰减
│   ├── generation/
│   │   ├── __init__.py
│   │   └── context_builder.py       # 上下文构建器
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── reranker.py              # Cross-Encoder / BM25 重排序
│   └── tools/
│       ├── search/
│       │   └── hybrid_search.py     # 混合搜索工具适配器
│       └── process/
│           ├── entity_extractor.py  # 实体提取（jieba + 词典）
│           ├── rerank.py            # 重排序工具
│           ├── compare.py           # LLM 文档对比
│           ├── summarize.py         # LLM 摘要生成
│           ├── extract.py           # LLM 信息抽取
│           └── reason.py            # LLM 因果推理
│
├── eval/                            # 评估框架
│   └── run_evaluation.py            # 路由/延迟/消融评估
│
├── scripts/                         # 工具脚本
│   ├── download_model.py            # 下载 BGE 嵌入模型
│   └── reindex.py                   # 索引管理（全量/增量/预览）
│
├── frontend/                        # React 前端
│   ├── package.json                 # 依赖声明
│   ├── vite.config.js               # Vite 配置 + 开发代理
│   ├── index.html                   # HTML 入口
│   ├── dist/                        # 构建产物
│   │   ├── index.html
│   │   └── assets/
│   │       ├── index-*.css
│   │       └── index-*.js
│   └── src/
│       ├── main.jsx                 # React 入口
│       ├── App.jsx                  # 主组件（侧栏 + 路由）
│       ├── App.css                  # 暗色主题全局样式
│       ├── api.js                   # API 客户端
│       └── components/
│           ├── Chat.jsx             # 聊天界面
│           ├── Profile.jsx          # 用户画像面板
│           └── Stats.jsx            # 监控面板
│
├── data/                            # PDF 论文源文件
├── docs/                            # 文档
├── last/storage/papers/txt/         # PDF 提取后的 TXT 文件
│
└── venv/                            # Python 虚拟环境（Windows）
```
