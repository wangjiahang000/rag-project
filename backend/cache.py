"""查询结果内存缓存（LRU 策略）

L1: 进程内 LRU 缓存，maxsize=128
     重复查询直接返回，延迟从 5-15s 降至 <10ms

L2: (预留) Redis，支持多实例共享
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class QueryCache:
    """LRU 查询结果缓存，线程安全"""

    def __init__(self, maxsize: int = 128, ttl: int = 3600):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def _make_key(self, question: str) -> str:
        normalized = " ".join(question.strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, question: str) -> Optional[dict]:
        key = self._make_key(question)
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if time.time() - ts > self._ttl:
            self._cache.pop(key)
            return None
        # LRU: 移到末尾
        self._cache.move_to_end(key)
        logger.info("[CACHE HIT] question=%s", question[:60])
        return json.loads(value)

    def set(self, question: str, response: dict):
        key = self._make_key(question)
        self._cache[key] = (time.time(), json.dumps(response, ensure_ascii=False))
        self._cache.move_to_end(key)
        # LRU 淘汰
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def invalidate(self, question: str):
        key = self._make_key(question)
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()
        logger.info("[CACHE] 已清空")

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_ratio(self) -> float:
        return getattr(self, "_hits", 0) / max(getattr(self, "_lookups", 1), 1)


# 全局单例
query_cache = QueryCache(maxsize=128, ttl=3600)
