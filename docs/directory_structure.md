# 项目目录结构

```

my_rag_project/
│
├── backend/                          # FastAPI 后端
│   ├── main.py                       # 主入口，注册工具、初始化路由/执行器、暴露 /chat 和 /health 接口
│   ├── routes/
│   │   ├── arxiv.py                  # arXiv 论文查询路由
│   │   ├── chat.py                   # 对话路由
│   │   └── upload.py                 # 文件上传路由
│
├── content_core/                     # 核心逻辑层
│   ├── config.py                     # 路由配置常量（阈值、权重、合法组合、LLM 参数）
│   ├── types.py                      # 数据模型定义（TaskNode, TaskGraph）
│   ├── task_router.py                # 三级意图路由：关键词 → SBERT 语义 → LLM 兜底
│   ├── graph_generator.py            # 模板化任务图生成，意图组合 → 可执行 TaskGraph
│   ├── graph_executor.py             # DAG 执行引擎，按拓扑顺序执行 TaskNode
│   ├── data/
│   │   └── vector_store.py           # ChromaDB 向量存储，支持 dense + sparse 混合检索
│   ├── models/
│   │   ├── sbert_classifier.py       # SBERT 多标签意图分类器（ONNX INT8 推理）
│   │   └── export_onnx.py            # SentenceTransformer → ONNX INT8 导出脚本
│   └── tools/
│       ├── process/                  # 文档加工工具
│       │   ├── entity_extractor.py   # 实体抽取引擎（jieba + 技术词典 + 质量分闭环）
│       │   ├── compare.py            # LLM 对比：从核心观点/方法/结论分析两组文档异同
│       │   ├── extract.py            # LLM 信息提取：从文档中提取关于目标的具体信息
│       │   ├── reason.py             # LLM 原因分析：从文档中分析原因/原理
│       │   ├── rerank.py             # LLM 重排序：基于查询对文档重排
│       │   └── summarize.py          # LLM 总结：3-5 句核心内容
│       └── search/
│           └── hybrid_search.py      # 混合检索入口，委托 VectorStore.hybrid_search
│
├── data/                             # 数据层
│   └── metadata_index.py             # 元数据索引桩函数（待实现）
│
├── memory/                           # 实体抽取持久化数据
│   ├── tech_dict.txt                 # 技术词典（453 个技术实体，含词频）
│   ├── blacklist.txt                 # 停用词黑名单（297 个非实体词）
│   └── candidates.txt                # 候选词发现记录（质量分≥0.55 可准入）
│
├── tests/                            # 测试文件
│   ├── test_v1.py                    # SBERT 关键词加权对比测试
│   ├── test_v2.py                    # TaskRouter 路由结果测试
│   ├── test_v3.py                    # 完整 pipeline 链路测试
│   ├── test_v4.py                    # 实体抽取质量报告测试
│   ├── test_routing.py               # 路由规则测试
│   ├── test_routing_v3.py            # 路由规则 v3 测试
│   ├── test_confidence.py            # 置信度测试
│   ├── test_full_pipeline.py         # 完整 pipeline 测试
│   ├── test_my_predictions.py        # 预测测试
│   ├── test_rag.py                   # RAG 测试
│   └── test_task_router.py           # TaskRouter 单元测试
│
├── last/                             # 旧版（arXiV 论文下载器遗留代码）
│   ├── Scripts/                      # 数据导入/迁移脚本
│   ├── core/                         # 旧版核心（loader, searcher, mysql_client 等）
│   └── models/                       # 旧版模型（sbert 训练、SQL 迁移）
│
├── frontend/                         # 前端（待实现）
├── checkpoints/                      # 模型检查点
├── chroma_data/                      # ChromaDB 持久化数据
│
├── docs/                             # 本文档
├── run.py                            # 项目启动入口
├── requirements.txt                  # Python 依赖
└── venv/                             # Python 虚拟环境
```

## 文件用途索引

| 文件 | 用途 |
|------|------|
| `backend/main.py` | FastAPI 应用入口，注册 6 个工具（hybrid_search/rerank/compare/summarize/extract/reason），暴露 `/chat` API |
| `content_core/config.py` | 路由阈值、权重、合法意图组合、LLM 重试参数集中管理 |
| `content_core/types.py` | `TaskNode` 和 `TaskGraph` 两个 Pydantic 模型定义 |
| `content_core/task_router.py` | 三级意图路由：关键词规则 → SBERT 语义分类 → LLM 兜底 |
| `content_core/graph_generator.py` | 模板化任务图生成，按意图组合映射为可执行节点序列 |
| `content_core/graph_executor.py` | DAG 执行引擎，按拓扑顺序执行 TaskNode，`{{id}}` 模板引用解析 |
| `content_core/data/vector_store.py` | ChromaDB 向量存储，稠密+稀疏混合检索 |
| `content_core/models/sbert_classifier.py` | 基于 BGE 的 6 类意图分类器，ONNX INT8 推理，正负例锚点打分 |
| `content_core/tools/process/entity_extractor.py` | jieba + 技术词典实体抽取，含质量分（频次/凝固度/技术相关性）和候选词发现 |
| `content_core/tools/process/compare.py` | LLM 对比工具，从观点/方法/结论三方面分析 |
| `content_core/tools/process/extract.py` | LLM 信息提取工具 |
| `content_core/tools/process/reason.py` | LLM 原因分析工具 |
| `content_core/tools/process/rerank.py` | LLM 重排序工具 |
| `content_core/tools/process/summarize.py` | LLM 文本总结工具 |
| `content_core/tools/search/hybrid_search.py` | 混合检索入口，委托 VectorStore |
| `data/metadata_index.py` | 元数据索引桩函数（人名→论文），待实现 |
| `memory/tech_dict.txt` | 453 个技术实体词典 |
| `memory/blacklist.txt` | 297 个停用词 |
| `memory/candidates.txt` | 候选词发现记录 |
| `run.py` | 项目启动入口 |
