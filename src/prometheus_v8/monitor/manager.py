"""Monitor Manager - Heartbeat + KPI + Anomaly + Prediction."""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Optional

from prometheus_v8.monitor.anomaly import AnomalyDetector, AnomalyEvent
from prometheus_v8.monitor.heartbeat import HealthCheck, HealthResult, HeartbeatMonitor
from prometheus_v8.monitor.kpi import KPICollector

logger = logging.getLogger(__name__)


class MonitorManager:
    """Unified monitoring: heartbeat + KPI + anomaly + prediction."""

    def __init__(self, heartbeat_interval: float = 30.0, anomaly_z_threshold: float = 2.5) -> None:
        self._heartbeat = HeartbeatMonitor(check_interval=heartbeat_interval)
        self._kpi = KPICollector()
        self._anomaly = AnomalyDetector(z_threshold=anomaly_z_threshold)
        self._lock = threading.RLock()
        self._alerts: deque[dict] = deque(maxlen=200)

        # Wire anomaly detector to KPI recording
        self._anomaly.add_callback(self._on_anomaly)

    def _on_anomaly(self, event: AnomalyEvent) -> None:
        """Handle anomaly events from the detector."""
        alert = {
            "type": "anomaly",
            "metric": event.metric,
            "value": event.value,
            "z_score": event.z_score,
            "severity": event.severity.value,
            "message": event.message,
            "timestamp": event.timestamp,
        }
        with self._lock:
            self._alerts.append(alert)
        logger.warning(f"Anomaly detected: {event.message}")

    def record_heartbeat(self, agent_id: str, status: str = "healthy") -> None:
        """Record a heartbeat for an agent (compatibility with MCP tools)."""
        self.record_kpi(f"heartbeat.{agent_id}", 1.0, {"status": status})

    def check(self, agent_id: str) -> str:
        """Check the last known status of an agent."""
        # Check heartbeat records for this specific agent
        try:
            for result in self._heartbeat.check_all():
                if result.name == agent_id:
                    return result.status.value
        except Exception as e:
            logger.debug(f"Agent health check failed for {agent_id}: {e}")
            pass
        # Fallback: check KPI records for this agent's heartbeat
        try:
            stats = self._kpi.compute_stats(f"heartbeat.{agent_id}")
            if stats and stats.count > 0:
                return "healthy" if stats.mean > 0 else "degraded"
        except Exception as e:
            logger.debug(f"Agent KPI health check failed for {agent_id}: {e}")
            pass
        return "unknown"

    def get(self, component: str) -> Any:
        """Get a sub-component by name (for MCP tool access)."""
        components = {"heartbeat": self._heartbeat, "kpi": self._kpi, "anomaly": self._anomaly}
        return components.get(component)

    def register_health_check(
        self, name: str, check_fn, interval: float = 30.0, timeout: float = 10.0, critical: bool = False
    ) -> None:
        """Register a health check with the heartbeat monitor."""
        hc = HealthCheck(name=name, check_fn=check_fn, interval=interval, timeout=timeout, critical=critical)
        self._heartbeat.register(hc)

    def check_now(self, name: str) -> Optional[HealthResult]:
        """Run a specific health check immediately."""
        return self._heartbeat.check_now(name)

    def check_all(self) -> list[HealthResult]:
        """Run all due health checks."""
        return self._heartbeat.check_all()

    def record_kpi(self, name: str, value: float, tags: dict | None = None) -> None:
        """Record a KPI value and check for anomalies."""
        labels = tags or {}
        self._kpi.record(name, value, labels=labels)
        # Check for anomaly
        self._anomaly.observe(name, value)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get all monitoring data for dashboard consumption."""
        return {
            "heartbeat": {
                "system_status": self._heartbeat.get_system_status().value,
                "components": {
                    name: {
                        "status": h.status.value,
                        "last_check": h.last_check,
                        "avg_latency_ms": h.avg_latency_ms,
                        "error_rate": h.error_rate,
                        "consecutive_failures": h.consecutive_failures,
                    }
                    for name, h in self._heartbeat.get_all_health().items()
                },
            },
            "kpi": self._kpi.export(),
            "anomalies": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "z_score": a.z_score,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in self._anomaly.get_anomalies(limit=10)
            ],
            "alerts": list(self._alerts)[-10:],
        }

    def start(self) -> None:
        """Start the heartbeat monitor background thread."""
        self._heartbeat.start()

    def stop(self) -> None:
        """Stop the heartbeat monitor background thread."""
        self._heartbeat.stop()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "alerts": len(self._alerts),
            "kpi_count": self._kpi.stats["metrics_registered"],
            "heartbeat": self._heartbeat.stats,
            "anomaly": self._anomaly.stats,
        }
