"""进程内指标收集"""

import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsCollector:
    """轻量级进程内指标收集器"""

    def __init__(self):
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._latencies: defaultdict[str, list[float]] = defaultdict(list)
        self._start_time = time.time()

    def inc(self, name: str, tags: dict = None):
        """递增计数器"""
        key = name
        if tags:
            key = f"{name}:{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
        self._counters[key] += 1

    def record(self, name: str, value: float, tags: dict = None):
        """记录延迟等数值"""
        self.inc(name, tags)
        key = name
        if tags:
            key = f"{name}:{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
        self._latencies[key].append(value)
        # 只保留最近 1000 条延迟，避免 OOM
        if len(self._latencies[key]) > 1000:
            self._latencies[key] = self._latencies[key][-500:]

    def snapshot(self) -> dict:
        """获取当前快照"""
        now = time.time()
        uptime = now - self._start_time

        result = {
            "uptime_seconds": uptime,
            "counters": dict(self._counters),
        }

        latencies = {}
        for key, values in self._latencies.items():
            if values:
                latencies[key] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "max_ms": max(values),
                    "p50_ms": sorted(values)[len(values) // 2],
                }
        result["latencies"] = latencies

        return result

    def reset(self):
        self._counters.clear()
        self._latencies.clear()


# 全局单例
metrics = MetricsCollector()
