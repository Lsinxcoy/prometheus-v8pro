"""KPI Collector."""
from __future__ import annotations
import threading
import time
from collections import defaultdict, deque
from typing import Any

class KPICollector:
    def __init__(self, max_history: int = 1000) -> None:
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.RLock()
    
    def record(self, name: str, value: float, tags: dict | None = None) -> None:
        with self._lock:
            self._metrics[name].append({"value": value, "tags": tags or {}, "timestamp": time.time()})
    
    def get_latest(self, name: str = "") -> dict[str, Any]:
        with self._lock:
            if name:
                entries = list(self._metrics.get(name, []))
                return {name: entries[-1] if entries else None}
            return {n: list(v)[-1] if v else None for n, v in self._metrics.items()}
    
    def get_history(self, name: str, limit: int = 100) -> list[dict]:
        with self._lock:
            return list(self._metrics.get(name, []))[-limit:]
    
    @property
    def count(self) -> int:
        return sum(len(v) for v in self._metrics.values())
