"""
TaskRouter 路由配置常量

所有魔术数字集中管理，避免散落在业务逻辑中。
修改参数时只需改此文件，无需翻阅路由代码。
"""

# ── 自适应幂次缩放 ──
# 幂次缩放指数，越大则高分和低分之间的差距越明显
# 推荐范围 [1.0, 4.0]，2.0 在拉开差距的同时保留一定区分度
POWER_SCALE = 2.0

# ── 关键词平滑提权参数 ──
# 规则命中的意图，根据 SBERT 分数线性映射到目标区间
# SBERT 0.27 → 提到 0.30（刚过门槛，微抬）
# SBERT 0.35 → 提到 0.42（适度提权，过大多数阈值）
# SBERT 0.50 → 提到 0.65（充分语义认可，满额提权）
# SBERT ≤ 0.27 不触发（防止字面匹配误判）
BOOST_RANGE = (0.27, 0.50)   # SBERT 原始分数区间 [起点, 终点]
BOOST_TARGET = (0.30, 0.65)  # 提权后的目标区间

# ── 意图筛选阈值 ──
# 每个意图的置信度门槛，综合分数 ≥ 此值才被采纳
# 所有意图统一 0.70，后续可根据各意图的实际表现差异化调整
SCORE_THRESHOLDS = {
    "retrieve": 0.70,
    "compare": 0.70,
    "summarize": 0.70,
    "howto": 0.70,
    "reason": 0.70,
    "extract": 0.70,
}

# top_conf 低于此值时触发 LLM 兜底
# 设 0.50：低于一半的语义置信度，需要 LLM 介入判断
LLM_FALLBACK_THRESHOLD = 0.70

# ── 分差过滤 ──
# 非关键词意图与 #1 意图的分数差距占比阈值
# 差距超过此比例则丢弃该意图（认为噪音）
RELATIVE_MARGIN = 0.05

# ── 合法意图组合规则 ──
# 超出以下组合的意图列表会被降级（先尝试匹配二元组，否则只保留 #1）
VALID_SINGLE = {"retrieve", "compare", "summarize", "howto", "reason", "extract"}
VALID_DUAL = [
    {"retrieve", "summarize"},
    {"retrieve", "compare"},
    {"retrieve", "extract"},
    {"retrieve", "reason"},
    {"retrieve", "howto"},
    {"compare", "reason"},
    {"extract", "compare"},
]
VALID_TRIPLE = [
    {"retrieve", "summarize", "compare"},
    {"retrieve", "extract", "compare"},
    {"retrieve", "reason", "summarize"},
    {"retrieve", "compare", "reason"},
]

# ── LLM 调用参数 ──
LLM_RETRY_COUNT = 2         # LLM 调用失败后的重试次数
LLM_TIMEOUT = 30            # 单次 LLM 调用超时（秒）
LLM_RETRY_DELAY = 1.0       # 重试间隔基数（秒，乘以尝试次数）

# ── 增强检索参数 ──
ENTITY_BOOST_FACTOR = 1.2         # 实体软加权提升倍数
TIME_DECAY_LAMBDA = 0.1           # 时间衰减指数系数
MAX_YEAR_RANGE = 20               # 年份硬截断阈值
DEFAULT_QUERY_YEAR = 2026         # 默认查询年份
MIN_RELEVANCE_SCORE = 0.5         # 最小相关度阈值（低于此值视为无相关结果）

# ── 重排序参数 ──
RERANK_MODE = "bm25"              # "bm25" | "cross_encoder"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"  # Cross-Encoder 模型路径
CROSS_ENCODER_DEVICE = "cpu"      # "cpu" | "cuda"

# ── 上下文构建参数 ──
CONTEXT_MAX_DOCS = 15             # 上下文最大文档数
CONTEXT_INCLUDE_METADATA = True   # 是否嵌入元数据
CITATION_ENABLED = True           # 是否启用引用溯源

# ── 查询改写参数 ──
QUERY_REWRITE_ENABLED = True      # 是否启用查询改写
QUERY_REWRITE_VARIANTS = 3        # 改写生成几个变体
QUERY_REWRITE_TOP_K = 15          # 改写后每路召回数

# ── MMR 多路召回参数 ──
MMR_ENABLED = True                # 是否启用 MMR 去重
MMR_LAMBDA = 0.5                  # MMR 多样性-相关性平衡，0.5等权重
MMR_CANDIDATE_POOL = 50           # MMR 候选池大小

# ── 意图路由参数 ──
KEYWORD_PRIORITY = {
    "summarize": ["reason"],
    "reason": ["howto"],
    "howto": ["extract"],
}

CHITCHAT_PATTERNS = [
    r"你是谁",
    r"你叫什么",
    r"你是什么",
    r"你是.{0,3}吗",
    r"我是.{0,3}吗",
    r"你好|嗨|hello|hi",
    r"再见|拜拜|bye",
    r"谢谢|多谢|感谢",
    r"聊天|闲聊",
]

RESOURCE_RULES = [
    (r"论文|文献|paper|article", "paper"),
    (r"代码|github|实现|编程|code", "code"),
    (r"知识图谱|图数据库|graph|neo4j", "kg"),
]

# ── DAG 执行引擎参数 ──
DAG_MAX_WORKERS = 8               # 异步 DAG 最大并发数
DAG_RETRY_COUNT = 1               # 单节点失败重试次数
