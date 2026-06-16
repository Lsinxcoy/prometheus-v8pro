"""Heartbeat Monitor - Periodic health checks for all system components."""
from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthResult:
    """Result of a single health check."""
    component: str = ""
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float = 0.0
    message: str = ""
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ComponentHealth:
    """Tracked health state of a component."""
    name: str = ""
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = 0.0
    last_healthy: float = 0.0
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    latency_samples: list[float] = field(default_factory=list)
    error_rate: float = 0.0

class HealthCheck:
    """A health check that can be registered with the monitor."""
    def __init__(self, name: str, check_fn: Callable[[], HealthResult],
                 interval: float = 30.0, timeout: float = 10.0,
                 critical: bool = False) -> None:
        self.name = name
        self.check_fn = check_fn
        self.interval = interval
        self.timeout = timeout
        self.critical = critical
        self._last_run = 0.0

    def is_due(self) -> bool:
        return time.time() - self._last_run >= self.interval

    def run(self) -> HealthResult:
        self._last_run = time.time()
        start = time.time()
        try:
            result = self.check_fn()
            result.component = self.name
            result.latency_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return HealthResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

class Alert:
    """An alert triggered by a health check failure."""
    def __init__(self, component: str, status: HealthStatus, message: str,
                 severity: str = "warning") -> None:
        self.component = component
        self.status = status
        self.message = message
        self.severity = severity
        self.timestamp = time.time()
        self.acknowledged = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component, "status": self.status.value,
            "message": self.message, "severity": self.severity,
            "timestamp": self.timestamp, "acknowledged": self.acknowledged,
        }

class HeartbeatMonitor:
    """Periodic health check monitor with alerting.
    
    Registers health checks for system components and runs them
    on configurable intervals. Triggers alerts when components
    become unhealthy.
    """
    
    def __init__(self, check_interval: float = 30.0,
                 failure_threshold: int = 3,
                 degraded_threshold: int = 1) -> None:
        self._check_interval = check_interval
        self._failure_threshold = failure_threshold
        self._degraded_threshold = degraded_threshold
        self._checks: dict[str, HealthCheck] = {}
        self._health: dict[str, ComponentHealth] = {}
        self._alerts: list[Alert] = []
        self._alert_callbacks: list[Callable[[Alert], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
    
    def register(self, check: HealthCheck) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[check.name] = check
            self._health[check.name] = ComponentHealth(name=check.name)
        logger.info(f"Registered health check: {check.name} (interval={check.interval}s)")
    
    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._health.pop(name, None)
    
    def check_now(self, name: str) -> Optional[HealthResult]:
        """Run a specific health check immediately."""
        with self._lock:
            check = self._checks.get(name)
        if not check:
            return None
        result = check.run()
        self._update_health(result)
        return result
    
    def check_all(self) -> list[HealthResult]:
        """Run all due health checks."""
        results = []
        with self._lock:
            due_checks = [c for c in self._checks.values() if c.is_due()]
        for check in due_checks:
            result = check.run()
            self._update_health(result)
            results.append(result)
        return results
    
    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Heartbeat monitor started")
    
    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Heartbeat monitor stopped")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback to be invoked when an alert is triggered."""
        self._alert_callbacks.append(callback)
    
    def get_alerts(self, unacknowledged_only: bool = True) -> list[Alert]:
        """Get current alerts."""
        with self._lock:
            if unacknowledged_only:
                return [a for a in self._alerts if not a.acknowledged]
            return list(self._alerts)
    
    def acknowledge_alert(self, index: int) -> bool:
        """Acknowledge an alert by index."""
        with self._lock:
            if 0 <= index < len(self._alerts):
                self._alerts[index].acknowledged = True
                return True
        return False
    
    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status of a specific component."""
        with self._lock:
            return self._health.get(name)
    
    def get_all_health(self) -> dict[str, ComponentHealth]:
        """Get health status of all components."""
        with self._lock:
            return dict(self._health)
    
    def get_system_status(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            if not self._health:
                return HealthStatus.UNKNOWN
            statuses = [h.status for h in self._health.values()]
            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                return HealthStatus.UNHEALTHY
            if any(s == HealthStatus.DEGRADED for s in statuses):
                return HealthStatus.DEGRADED
            if all(s == HealthStatus.HEALTHY for s in statuses):
                return HealthStatus.HEALTHY
            return HealthStatus.UNKNOWN
    
    def _update_health(self, result: HealthResult) -> None:
        """Update component health based on check result."""
        with self._lock:
            health = self._health.get(result.component)
            if not health:
                health = ComponentHealth(name=result.component)
                self._health[result.component] = health
            
            health.last_check = result.timestamp
            health.total_checks += 1
            health.status = result.status
            
            # Update latency tracking
            health.latency_samples.append(result.latency_ms)
            if len(health.latency_samples) > 100:
                health.latency_samples = health.latency_samples[-100:]
            health.avg_latency_ms = sum(health.latency_samples) / len(health.latency_samples)
            
            if result.status == HealthStatus.HEALTHY:
                health.last_healthy = result.timestamp
                health.consecutive_failures = 0
            else:
                health.consecutive_failures += 1
                health.total_failures += 1
            
            health.error_rate = health.total_failures / max(1, health.total_checks)
            
            # Trigger alerts
            if health.consecutive_failures >= self._failure_threshold:
                alert = Alert(result.component, result.status,
                            f"Component {result.component} unhealthy: {result.message}",
                            severity="critical" if result.status == HealthStatus.UNHEALTHY else "warning")
                self._alerts.append(alert)
                for cb in self._alert_callbacks:
                    try:
                        cb(alert)
                    except Exception as e:
                        logger.warning(f"Alert callback error: {e}")
            elif health.consecutive_failures >= self._degraded_threshold:
                alert = Alert(result.component, HealthStatus.DEGRADED,
                            f"Component {result.component} degraded: {result.message}",
                            severity="warning")
                self._alerts.append(alert)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Heartbeat check error: {e}")
            time.sleep(self._check_interval)
    
    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "components": len(self._checks),
                "system_status": self.get_system_status().value,
                "total_alerts": len(self._alerts),
                "unacknowledged_alerts": sum(1 for a in self._alerts if not a.acknowledged),
                "components_healthy": sum(1 for h in self._health.values() if h.status == HealthStatus.HEALTHY),
                "components_degraded": sum(1 for h in self._health.values() if h.status == HealthStatus.DEGRADED),
                "components_unhealthy": sum(1 for h in self._health.values() if h.status == HealthStatus.UNHEALTHY),
            }
