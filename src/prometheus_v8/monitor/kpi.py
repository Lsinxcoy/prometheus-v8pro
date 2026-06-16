"""KPI Collector - Time-series metric collection and aggregation."""
from __future__ import annotations
import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

@dataclass
class MetricDefinition:
    """Definition of a tracked metric."""
    name: str = ""
    type: MetricType = MetricType.GAUGE
    unit: str = ""
    description: str = ""
    labels: list[str] = field(default_factory=list)
    retention_seconds: float = 3600.0  # 1 hour default

@dataclass
class MetricSample:
    """A single metric sample."""
    value: float = 0.0
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)

@dataclass
class MetricStats:
    """Statistical summary of a metric."""
    name: str = ""
    count: int = 0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    stddev: float = 0.0
    last_value: float = 0.0
    last_timestamp: float = 0.0
    rate_per_second: float = 0.0

class KPICollector:
    """Time-series KPI collector with statistical aggregation.
    
    Supports three metric types:
    - Counter: Monotonically increasing values (e.g., request count)
    - Gauge: Point-in-time values (e.g., memory usage)
    - Histogram: Distribution of values (e.g., latency)
    
    Features:
    - Configurable retention per metric
    - Statistical aggregation (mean, p50, p95, p99, min, max, stddev)
    - Rate calculation for counters
    - Label-based dimensional metrics
    - Thread-safe operations
    """
    
    def __init__(self, default_retention: float = 3600.0,
                 max_samples_per_metric: int = 1000) -> None:
        self._default_retention = default_retention
        self._max_samples = max_samples_per_metric
        self._definitions: dict[str, MetricDefinition] = {}
        self._samples: dict[str, deque[MetricSample]] = defaultdict(lambda: deque(maxlen=self._max_samples))
        self._counters: dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Register built-in metrics
        self._register_builtins()
    
    def _register_builtins(self) -> None:
        """Register built-in Prometheus V8 metrics."""
        builtins = [
            ("memory_node_count", MetricType.GAUGE, "nodes", "Total nodes in memory store"),
            ("memory_edge_count", MetricType.GAUGE, "edges", "Total edges in knowledge graph"),
            ("evolution_generation", MetricType.GAUGE, "", "Current evolution generation"),
            ("evolution_best_fitness", MetricType.GAUGE, "", "Best fitness score achieved"),
            ("evolution_avg_fitness", MetricType.GAUGE, "", "Average fitness in population"),
            ("safety_checks_total", MetricType.COUNTER, "checks", "Total safety checks performed"),
            ("safety_violations_total", MetricType.COUNTER, "violations", "Total safety violations detected"),
            ("safety_violation_rate", MetricType.GAUGE, "ratio", "Current safety violation rate"),
            ("query_latency_ms", MetricType.HISTOGRAM, "ms", "Query latency in milliseconds"),
            ("query_count_total", MetricType.COUNTER, "queries", "Total queries processed"),
            ("consolidation_count", MetricType.COUNTER, "consolidations", "Total memory consolidations"),
            ("consolidation_rate", MetricType.GAUGE, "ratio", "Consolidation success rate"),
            ("dream_cycle_count", MetricType.COUNTER, "cycles", "Total dream cycles completed"),
            ("daily_learning_rounds", MetricType.GAUGE, "rounds", "Learning rounds completed today"),
            ("coral_notes_count", MetricType.GAUGE, "notes", "CORAL reflection notes written"),
            ("coral_skills_count", MetricType.GAUGE, "skills", "CORAL consolidated skills"),
            ("circuit_breaker_state", MetricType.GAUGE, "state", "Circuit breaker state (0=closed,1=open,2=half)"),
            ("trust_verified_ratio", MetricType.GAUGE, "ratio", "Ratio of verified knowledge"),
            ("autonomy_level", MetricType.GAUGE, "level", "Current autonomy level (0-4)"),
            ("active_agents", MetricType.GAUGE, "agents", "Number of active agents"),
        ]
        for name, mtype, unit, desc in builtins:
            self.register(MetricDefinition(name=name, type=mtype, unit=unit, description=desc))
    
    def register(self, definition: MetricDefinition) -> None:
        """Register a new metric definition."""
        with self._lock:
            self._definitions[definition.name] = definition
            if definition.name not in self._samples:
                self._samples[definition.name] = deque(maxlen=self._max_samples)
            if definition.type == MetricType.COUNTER:
                self._counters[definition.name] = 0.0
    
    def record(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a metric value."""
        with self._lock:
            if name not in self._definitions:
                # Auto-register as gauge
                self.register(MetricDefinition(name=name, type=MetricType.GAUGE))
            
            defn = self._definitions[name]
            sample = MetricSample(value=value, labels=labels or {})
            self._samples[name].append(sample)
            
            if defn.type == MetricType.COUNTER:
                self._counters[name] = value
    
    def increment(self, name: str, delta: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0.0
                if name not in self._definitions:
                    self.register(MetricDefinition(name=name, type=MetricType.COUNTER))
            self._counters[name] += delta
            self.record(name, self._counters[name], labels)
    
    def get_value(self, name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        with self._lock:
            samples = self._samples.get(name)
            if not samples:
                return None
            return samples[-1].value
    
    def get_history(self, name: str, limit: int = 100,
                    since: float | None = None) -> list[dict[str, Any]]:
        """Get historical samples for a metric."""
        with self._lock:
            samples = self._samples.get(name, deque())
            result = []
            for s in reversed(samples):
                if since and s.timestamp < since:
                    break
                result.append({"value": s.value, "timestamp": s.timestamp, "labels": s.labels})
                if len(result) >= limit:
                    break
            result.reverse()
            return result
    
    def compute_stats(self, name: str, window_seconds: float | None = None) -> Optional[MetricStats]:
        """Compute statistical summary for a metric."""
        with self._lock:
            samples = self._samples.get(name, deque())
            if not samples:
                return None
            
            now = time.time()
            if window_seconds:
                values = [s.value for s in samples if now - s.timestamp <= window_seconds]
            else:
                values = [s.value for s in samples]
            
            if not values:
                return None
            
            values_sorted = sorted(values)
            n = len(values)
            mean = sum(values) / n
            
            # Standard deviation
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                stddev = math.sqrt(variance)
            else:
                stddev = 0.0
            
            # Percentiles
            def percentile(p: float) -> float:
                idx = int(p / 100 * (n - 1))
                return values_sorted[min(idx, n - 1)]
            
            # Rate calculation for counters
            rate = 0.0
            if len(samples) >= 2 and self._definitions.get(name, MetricDefinition()).type == MetricType.COUNTER:
                first = samples[0]
                last = samples[-1]
                dt = last.timestamp - first.timestamp
                if dt > 0:
                    rate = (last.value - first.value) / dt
            
            return MetricStats(
                name=name, count=n, mean=mean,
                min=values_sorted[0], max=values_sorted[-1],
                p50=percentile(50), p95=percentile(95), p99=percentile(99),
                stddev=stddev, last_value=values[-1],
                last_timestamp=samples[-1].timestamp,
                rate_per_second=rate,
            )
    
    def get_all_stats(self) -> dict[str, MetricStats]:
        """Get stats for all metrics."""
        with self._lock:
            return {name: self.compute_stats(name) for name in self._definitions
                    if self.compute_stats(name) is not None}
    
    def export(self) -> dict[str, Any]:
        """Export all metrics as a dict for dashboard consumption."""
        with self._lock:
            result = {}
            for name in self._definitions:
                stats = self.compute_stats(name)
                if stats:
                    result[name] = {
                        "type": self._definitions[name].type.value,
                        "unit": self._definitions[name].unit,
                        "description": self._definitions[name].description,
                        "current": stats.last_value,
                        "mean": round(stats.mean, 4),
                        "min": round(stats.min, 4),
                        "max": round(stats.max, 4),
                        "p50": round(stats.p50, 4),
                        "p95": round(stats.p95, 4),
                        "p99": round(stats.p99, 4),
                        "count": stats.count,
                        "rate": round(stats.rate_per_second, 6),
                    }
            return result
    
    def cleanup(self) -> int:
        """Remove expired samples. Returns count of removed samples."""
        with self._lock:
            now = time.time()
            removed = 0
            for name, samples in self._samples.items():
                defn = self._definitions.get(name)
                retention = defn.retention_seconds if defn else self._default_retention
                before = len(samples)
                while samples and now - samples[0].timestamp > retention:
                    samples.popleft()
                removed += before - len(samples)
            return removed
    
    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "metrics_registered": len(self._definitions),
                "total_samples": sum(len(s) for s in self._samples.values()),
                "counters": len(self._counters),
            }
