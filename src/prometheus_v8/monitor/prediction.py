"""Trend Prediction - Linear regression based metric forecasting."""
from __future__ import annotations
import math
import threading
from collections import deque
from typing import Any
from prometheus_v8.monitor.kpi import KPICollector

class TrendPredictor:
    """Simple linear regression-based metric trend prediction."""
    
    def __init__(self, kpi: KPICollector | None = None) -> None:
        self._kpi = kpi
        self._lock = threading.RLock()
    
    def predict(self, metric: str, horizon: int = 5) -> dict[str, Any]:
        """Predict metric values for next N steps using linear regression."""
        if not self._kpi:
            return {"metric": metric, "prediction": [], "trend": "unknown"}
        
        history = self._kpi.get_history(metric, limit=20)
        if len(history) < 3:
            return {"metric": metric, "prediction": [], "trend": "insufficient_data"}
        
        values = [h["value"] for h in history]
        n = len(values)
        
        # Simple linear regression: y = a + b*x
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        ss_xy = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        ss_xx = sum((i - x_mean) ** 2 for i in range(n))
        
        if ss_xx == 0:
            slope = 0
        else:
            slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        
        # Predict next N values
        predictions = [intercept + slope * (n + i) for i in range(horizon)]
        
        # Determine trend
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {"metric": metric, "prediction": predictions, "trend": trend, "slope": slope}
