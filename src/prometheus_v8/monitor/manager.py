"""Monitor Manager - Heartbeat + KPI + Anomaly + Prediction."""
from __future__ import annotations
import logging
import threading
import time
from typing import Any, Optional
from prometheus_v8.monitor.heartbeat import HeartbeatTracker
from prometheus_v8.monitor.kpi import KPICollector
from prometheus_v8.monitor.anomaly import AnomalyDetector

logger = logging.getLogger(__name__)

class MonitorManager:
    """Unified monitoring: heartbeat + KPI + anomaly + prediction."""
    
    def __init__(self, heartbeat_interval: int = 30, anomaly_threshold: float = 2.0) -> None:
        self._heartbeat = HeartbeatTracker(interval=heartbeat_interval)
        self._kpi = KPICollector()
        self._anomaly = AnomalyDetector(threshold=anomaly_threshold)
        self._lock = threading.RLock()
        self._alerts: list[dict] = []
    
    def record_heartbeat(self, agent_id: str, status: str = "alive") -> None:
        self._heartbeat.beat(agent_id, status)
    
    def record_kpi(self, name: str, value: float, tags: dict | None = None) -> None:
        self._kpi.record(name, value, tags)
        # Check for anomaly
        is_anomaly, score = self._anomaly.check(name, value)
        if is_anomaly:
            alert = {"type": "anomaly", "metric": name, "value": value, "score": score, "timestamp": time.time()}
            self._alerts.append(alert)
            logger.warning(f"Anomaly detected: {name}={value} (score={score:.2f})")
    
    def get_dashboard_data(self) -> dict[str, Any]:
        return {
            "heartbeat": self._heartbeat.get_status(),
            "kpi": self._kpi.get_latest(),
            "anomalies": self._anomaly.get_anomalies(limit=10),
            "alerts": self._alerts[-10:],
        }
    
    @property
    def stats(self) -> dict[str, Any]:
        return {"alerts": len(self._alerts), "kpi_count": self._kpi.count}
