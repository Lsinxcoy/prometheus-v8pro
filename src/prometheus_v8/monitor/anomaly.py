"""Anomaly Detector - Z-score and EMA based anomaly detection on metric streams."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """An anomaly detected in a metric stream."""

    metric: str = ""
    value: float = 0.0
    expected: float = 0.0
    deviation: float = 0.0
    z_score: float = 0.0
    severity: AnomalySeverity = AnomalySeverity.LOW
    message: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetricBaseline:
    """Statistical baseline for a metric stream."""

    mean: float = 0.0
    variance: float = 0.0
    stddev: float = 0.0
    ema: float = 0.0  # Exponential moving average
    ema_var: float = 0.0  # EMA of variance
    sample_count: int = 0
    last_update: float = field(default_factory=time.time)


class AnomalyDetector:
    """Multi-metric anomaly detector using Z-score and EMA methods.

    Features:
    - Z-score based detection: flag values > z_threshold standard deviations from mean
    - EMA baseline: tracks exponential moving average as expected value
    - Configurable thresholds per metric
    - Sliding window buffer for streaming detection
    - Severity classification based on deviation magnitude
    - Callback-based alert system
    - Thread-safe operations
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        ema_alpha: float = 0.1,
        window_size: int = 100,
        min_samples: int = 10,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._z_threshold = z_threshold
        self._ema_alpha = ema_alpha
        self._window_size = window_size
        self._min_samples = min_samples
        self._cooldown = cooldown_seconds
        self._buffers: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._baselines: dict[str, MetricBaseline] = {}
        self._anomalies: list[AnomalyEvent] = []
        self._last_alert: dict[str, float] = {}  # metric -> last alert timestamp
        self._callbacks: list[Callable[[AnomalyEvent], None]] = []
        self._lock = threading.RLock()

    def add_callback(self, callback: Callable[[AnomalyEvent], None]) -> None:
        """Add a callback to be invoked when an anomaly is detected."""
        self._callbacks.append(callback)

    def observe(self, metric: str, value: float) -> Optional[AnomalyEvent]:
        """Observe a metric value and check for anomalies.

        Returns an AnomalyEvent if anomaly detected, None otherwise.
        """
        with self._lock:
            buffer = self._buffers[metric]
            buffer.append(value)

            baseline = self._baselines.get(metric)
            if baseline is None:
                baseline = MetricBaseline(ema=value)
                self._baselines[metric] = baseline

            # Update EMA baseline
            alpha = self._ema_alpha
            baseline.ema = alpha * value + (1 - alpha) * baseline.ema
            delta = value - baseline.ema
            baseline.ema_var = alpha * (delta**2) + (1 - alpha) * baseline.ema_var
            baseline.sample_count += 1
            baseline.last_update = time.time()

            # Need minimum samples before detecting anomalies
            if len(buffer) < self._min_samples:
                return None

            # Compute statistics from sliding window
            values = list(buffer)
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
            stddev = math.sqrt(variance) if variance > 0 else 1e-10

            baseline.mean = mean
            baseline.variance = variance
            baseline.stddev = stddev

            # Z-score calculation
            z_score = abs(value - mean) / stddev

            # Check if anomalous (use per-metric threshold if set)
            threshold = (
                self._metric_thresholds.get(metric, self._z_threshold)
                if hasattr(self, "_metric_thresholds")
                else self._z_threshold
            )
            if z_score > threshold:
                # Cooldown check - don't spam alerts
                last_alert_time = self._last_alert.get(metric, 0)
                if time.time() - last_alert_time < self._cooldown:
                    return None

                # Determine severity
                if z_score > 4.0:
                    severity = AnomalySeverity.CRITICAL
                elif z_score > 3.0:
                    severity = AnomalySeverity.HIGH
                elif z_score > 2.5:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW

                event = AnomalyEvent(
                    metric=metric,
                    value=value,
                    expected=mean,
                    deviation=value - mean,
                    z_score=z_score,
                    severity=severity,
                    message=f"Anomaly in {metric}: value={value:.4f}, expected={mean:.4f}, z={z_score:.2f}",
                )

                self._anomalies.append(event)
                self._last_alert[metric] = time.time()

                # Invoke callbacks
                for cb in self._callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        logger.warning(f"Anomaly callback error: {e}")

                return event

            return None

    def observe_batch(self, metrics: dict[str, float]) -> list[AnomalyEvent]:
        """Observe multiple metric values at once."""
        anomalies = []
        for metric, value in metrics.items():
            event = self.observe(metric, value)
            if event:
                anomalies.append(event)
        return anomalies

    def get_baseline(self, metric: str) -> Optional[MetricBaseline]:
        """Get the current baseline for a metric."""
        with self._lock:
            return self._baselines.get(metric)

    def get_anomalies(
        self, since: float | None = None, severity: AnomalySeverity | None = None, limit: int = 100
    ) -> list[AnomalyEvent]:
        """Get detected anomalies with optional filtering."""
        with self._lock:
            result = self._anomalies
            if since:
                result = [a for a in result if a.timestamp >= since]
            if severity:
                result = [a for a in result if a.severity == severity]
            return result[-limit:]

    def clear_anomalies(self) -> None:
        """Clear all recorded anomalies."""
        with self._lock:
            self._anomalies.clear()

    def set_threshold(self, metric: str, z_threshold: float) -> None:
        """Set a custom Z-score threshold for a specific metric."""
        # Store per-metric thresholds
        if not hasattr(self, "_metric_thresholds"):
            self._metric_thresholds = {}
        self._metric_thresholds[metric] = z_threshold

    def get_metric_summary(self, metric: str) -> dict[str, Any]:
        """Get a summary of a metric's current state."""
        with self._lock:
            baseline = self._baselines.get(metric)
            buffer = self._buffers.get(metric, deque())
            if not baseline or not buffer:
                return {"metric": metric, "status": "no_data"}
            return {
                "metric": metric,
                "samples": len(buffer),
                "mean": round(baseline.mean, 4),
                "stddev": round(baseline.stddev, 4),
                "ema": round(baseline.ema, 4),
                "last_value": buffer[-1] if buffer else None,
                "z_threshold": self._z_threshold,
            }

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "metrics_tracked": len(self._baselines),
                "total_anomalies": len(self._anomalies),
                "z_threshold": self._z_threshold,
                "ema_alpha": self._ema_alpha,
                "window_size": self._window_size,
            }
