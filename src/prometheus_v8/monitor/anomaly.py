"""Anomaly Detector - Z-score based."""
from __future__ import annotations
import math
import threading
from collections import defaultdict, deque
from typing import Any

class AnomalyDetector:
    def __init__(self, threshold: float = 2.0, window: int = 100) -> None:
        self._threshold = threshold
        self._window = window
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._anomalies: list[dict] = []
        self._lock = threading.RLock()
    
    def check(self, metric: str, value: float) -> tuple[bool, float]:
        with self._lock:
            self._history[metric].append(value)
            values = list(self._history[metric])
        
        if len(values) < 3:
            return False, 0.0
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1.0
        
        z_score = abs(value - mean) / std
        is_anomaly = z_score > self._threshold
        
        if is_anomaly:
            self._anomalies.append({"metric": metric, "value": value, "z_score": z_score})
        
        return is_anomaly, z_score
    
    def get_anomalies(self, limit: int = 10) -> list[dict]:
        return self._anomalies[-limit:]
