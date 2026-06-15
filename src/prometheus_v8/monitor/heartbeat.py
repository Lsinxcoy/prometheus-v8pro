"""Heartbeat Tracker."""
from __future__ import annotations
import threading
import time
from collections import defaultdict
from typing import Any

class HeartbeatTracker:
    def __init__(self, interval: int = 30) -> None:
        self._interval = interval
        self._beats: dict[str, float] = {}
        self._status: dict[str, str] = {}
        self._lock = threading.RLock()
    
    def beat(self, agent_id: str, status: str = "alive") -> None:
        with self._lock:
            self._beats[agent_id] = time.time()
            self._status[agent_id] = status
    
    def is_alive(self, agent_id: str, timeout: float | None = None) -> bool:
        timeout = timeout or self._interval * 3
        with self._lock:
            last = self._beats.get(agent_id, 0)
            return time.time() - last < timeout
    
    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {aid: {"last_beat": t, "status": self._status.get(aid, "unknown"),
                         "alive": self.is_alive(aid)} for aid, t in self._beats.items()}
