"""会话记忆管理

短期记忆：维护多轮对话历史，支持上下文注入和指代消解。
长期记忆：(预留) 用户兴趣画像持久化到向量库。

设计：
- session_id 由前端生成（UUID），无注册要求
- 每条消息存储 {role, content, timestamp, tasks, entities}
- 超过 MAX_TURNS 时自动压缩历史
"""

import logging
import re
import json
import os
import time
from typing import Optional
from litellm import acompletion

logger = logging.getLogger(__name__)

# ── 配置 ──
MAX_TURNS = 10  # 超过此轮数触发压缩
MAX_CONTEXT_TOKENS = 2000  # 注入上下文的最大字符数
REFERENCE_PATTERNS = re.compile(r"它|这|那|该|其|它们|这些|那些|上述|以上|这个|那个")

# ── 类型 ──

class Turn:
    """单轮对话"""
    __slots__ = ("role", "content", "timestamp", "tasks", "entities")

    def __init__(self, role: str, content: str, tasks: list = None, entities: list = None):
        self.role = role
        self.content = content
        self.timestamp = time.time()
        self.tasks = tasks or []
        self.entities = entities or []

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tasks": self.tasks,
            "entities": self.entities,
        }


class Session:
    """单个会话（含用户画像）"""

    def __init__(self, session_id: str):
        self.id = session_id
        self.history: list[Turn] = []
        self.created_at = time.time()
        self.last_active = time.time()
        # 用户画像
        self.interest_tags: dict[str, float] = {}  # tag -> 权重（累计衰减）
        self.intent_history: dict[str, int] = {}   # intent -> 频次
        self.favorite_papers: list[str] = []        # 收藏论文 arxiv_id
        self.total_queries = 0

    def add_turn(self, turn: Turn):
        self.history.append(turn)
        self.last_active = time.time()
        self.total_queries += 1

        # 更新意图频次
        if turn.role == "user":
            for t in turn.tasks:
                self.intent_history[t] = self.intent_history.get(t, 0) + 1

        # 更新兴趣标签（实体出现一次权重 +1，每轮衰减 0.98）
        if turn.entities:
            decay = 0.98 ** len(self.history)
            for ent in turn.entities:
                if len(ent) < 2:
                    continue
                current = self.interest_tags.get(ent, 0)
                self.interest_tags[ent] = current * decay + 1.0

    def get_recent(self, n: int = 5) -> list[Turn]:
        return self.history[-n:]

    @property
    def top_interests(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.interest_tags.items(), key=lambda x: x[1], reverse=True)[:n]

    @property
    def top_intents(self, n: int = 3) -> list[tuple[str, int]]:
        return sorted(self.intent_history.items(), key=lambda x: x[1], reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "turns": len(self.history),
            "total_queries": self.total_queries,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "top_interests": self.top_interests,
            "top_intents": self.top_intents,
        }


class SessionManager:
    """会话管理器（内存存储）"""

    def __init__(self, max_sessions: int = 1000, session_ttl: int = 1800):
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id)
            # 淘汰过期会话
            self._evict_stale()
        return self._sessions[session_id]

    def _evict_stale(self):
        now = time.time()
        stale = [sid for sid, s in self._sessions.items()
                 if now - s.last_active > self._session_ttl]
        for sid in stale:
            del self._sessions[sid]
        # 超过最大数量时淘汰最旧的
        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(self._sessions.items(),
                                     key=lambda x: x[1].last_active)
            for sid, _ in sorted_sessions[:len(self._sessions) - self._max_sessions]:
                del self._sessions[sid]

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ── 引用消解 ──

def needs_reference_resolution(query: str) -> bool:
    """判断 query 是否包含待消解的指代"""
    return bool(REFERENCE_PATTERNS.search(query))


async def resolve_references(query: str, history: list[Turn]) -> str:
    """用 LLM 将指代还原为具体名词"""
    if not history or not needs_reference_resolution(query):
        return query

    turns = history[-4:]  # 最近 4 轮
    history_text = "\n".join(
        f"{'用户' if t.role == 'user' else '助手'}: {t.content}"
        for t in turns
    )

    prompt = f"""你是一个指代消解助手。将用户最新问句中所有指代（它、这、那、该、其、它们、这些、那些等）还原为历史对话中的具体名词。
只输出改写后的问句，不要输出任何解释。

历史对话：
{history_text}

最新问句：{query}

改写后的问句："""

    try:
        response = await acompletion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL"),
            max_tokens=256,
        )
        resolved = response.choices[0].message.content.strip()
        logger.info("[REFERENCE RESOLVE] '%s' → '%s'", query[:40], resolved[:60])
        return resolved
    except Exception as e:
        logger.warning("[REFERENCE RESOLVE] 失败: %s, 使用原始 query", e)
        return query


# ── 全局单例 ──
session_manager = SessionManager(max_sessions=1000, session_ttl=1800)
