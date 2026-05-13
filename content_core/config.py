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
