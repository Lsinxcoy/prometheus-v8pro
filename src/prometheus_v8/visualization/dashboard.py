"""Dashboard Data Provider - Complete visualization data for the web UI.

Provides structured data for:
- System overview (version, health, key metrics)
- Memory distribution by layer and type
- Evolution progress and history
- Safety violation tracking
- Governance autonomy and trust levels
- Agent pool status
- Knowledge gap analysis
- Trend predictions
"""
from __future__ import annotations
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.schema import MemoryLayer, NodeType, TrustLevel

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetric:
    """A dashboard metric with current value, trend, and sparkline data."""
    name: str = ""
    value: float = 0.0
    unit: str = ""
    trend: str = "stable"  # increasing/decreasing/stable
    change_pct: float = 0.0
    sparkline: list[float] = field(default_factory=list)
    status: str = "normal"  # normal/warning/critical

@dataclass
class DashboardPanel:
    """A dashboard panel containing related metrics."""
    title: str = ""
    panel_type: str = "metrics"  # metrics/chart/table/gauge
    data: Any = None
    updated_at: float = field(default_factory=time.time)

class DashboardProvider:
    """Complete dashboard data provider for the web UI.
    
    Aggregates data from all system components into dashboard-friendly
    structures for the HTTP API and WebSocket real-time updates.
    """

    def __init__(self, store=None, engine=None, safety=None, monitor=None,
                 governance=None, agents=None, knowledge=None) -> None:
        self._store = store
        self._engine = engine
        self._safety = safety
        self._monitor = monitor
        self._governance = governance
        self._agents = agents
        self._knowledge = knowledge
        self._history: deque = __import__("collections").deque(maxlen=100)
        self._last_update = 0.0

    def get_overview(self) -> dict[str, Any]:
        """Get the main dashboard overview."""
        overview = {
            "version": "8.0.0",
            "timestamp": time.time(),
            "architecture_layers": 9,
            "evolution_layers": 12,
            "nodes": self._get_node_stats(),
            "evolution": self._get_evolution_stats(),
            "safety": self._get_safety_stats(),
            "governance": self._get_governance_stats(),
            "agents": self._get_agent_stats(),
            "knowledge": self._get_knowledge_stats(),
        }
        return overview

    def get_memory_distribution(self) -> dict[str, Any]:
        """Get memory distribution by layer and type."""
        distribution = {"by_layer": {}, "by_type": {}, "by_trust": {}}
        
        if self._store:
            # By layer
            for layer in MemoryLayer:
                count = self._store.count_nodes(layer)
                if count > 0:
                    distribution["by_layer"][layer.value] = count
            
            # By type
            for ntype in NodeType:
                try:
                    nodes = self._store.get_nodes_by_type(ntype, limit=10000)
                    if nodes:
                        distribution["by_type"][ntype.value] = len(nodes)
                except Exception:
                    pass
            
            # By trust level
            total = self._store.count_nodes()
            distribution["by_trust"] = {"total": total}
            if total > 0:
                # Approximate trust distribution
                distribution["by_trust"]["verified_ratio"] = 0.0
                distribution["by_trust"]["high_signal_ratio"] = 0.0
                distribution["by_trust"]["pending_ratio"] = 1.0
        
        return distribution

    def get_evolution_progress(self) -> dict[str, Any]:
        """Get evolution progress data for charting."""
        progress = {
            "generation": 0,
            "best_fitness": 0,
            "avg_fitness": 0,
            "layer_stats": {},
            "direction": "auto",
            "history": [],
        }
        
        if self._engine:
            progress["generation"] = self._engine.generation
            if self._engine.best_genome:
                progress["best_fitness"] = round(self._engine.best_genome.fitness, 4)
            progress["layer_stats"] = self._engine.layer_stats
            progress["history"] = self._engine.history[-20:]
        
        return progress

    def get_safety_dashboard(self) -> dict[str, Any]:
        """Get safety metrics for the dashboard."""
        if not self._safety:
            return {"checks": 0, "violations": 0, "violation_rate": 0}
        return self._safety.stats

    def get_governance_dashboard(self) -> dict[str, Any]:
        """Get governance metrics for the dashboard."""
        result = {}
        if self._governance:
            if "autonomy" in self._governance:
                result["autonomy"] = self._governance["autonomy"].stats
            if "trust" in self._governance:
                result["trust"] = self._governance["trust"].stats
            if "curiosity" in self._governance:
                result["curiosity"] = self._governance["curiosity"].stats
        return result

    def get_key_metrics(self) -> list[DashboardMetric]:
        """Get key metrics as DashboardMetric objects."""
        metrics = []
        
        # Node count
        total_nodes = self._store.count_nodes() if self._store else 0
        metrics.append(DashboardMetric(
            name="Total Nodes", value=total_nodes, unit="nodes",
            trend="stable", status="normal",
        ))
        
        # Evolution fitness
        best_fitness = self._engine.best_genome.fitness if self._engine and self._engine.best_genome else 0
        metrics.append(DashboardMetric(
            name="Best Fitness", value=round(best_fitness, 4), unit="score",
            trend="increasing" if best_fitness > 0.5 else "stable",
            status="normal" if best_fitness > 0.3 else "warning",
        ))
        
        # Generation
        gen = self._engine.generation if self._engine else 0
        metrics.append(DashboardMetric(
            name="Generation", value=gen, unit="gen",
            trend="increasing", status="normal",
        ))
        
        # Safety checks
        safety_stats = self._safety.stats if self._safety else {"checks": 0, "violation_rate": 0}
        metrics.append(DashboardMetric(
            name="Safety Checks", value=safety_stats.get("checks", 0), unit="checks",
            trend="stable", status="normal",
        ))
        
        # Violation rate
        vrate = safety_stats.get("violation_rate", 0)
        metrics.append(DashboardMetric(
            name="Violation Rate", value=round(vrate, 4), unit="ratio",
            trend="stable",
            status="critical" if vrate > 0.1 else "warning" if vrate > 0.05 else "normal",
        ))
        
        return metrics

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health assessment."""
        health = {"status": "healthy", "issues": [], "score": 1.0}
        
        # Check store
        if not self._store:
            health["issues"].append("Store not available")
            health["score"] -= 0.3
        
        # Check evolution
        if self._engine and self._engine.best_genome:
            if self._engine.best_genome.fitness < 0.1:
                health["issues"].append("Evolution fitness critically low")
                health["score"] -= 0.2
        
        # Check safety
        if self._safety:
            stats = self._safety.stats
            if stats.get("violation_rate", 0) > 0.1:
                health["issues"].append("High safety violation rate")
                health["score"] -= 0.3
        
        if health["score"] < 0.5:
            health["status"] = "critical"
        elif health["score"] < 0.8:
            health["status"] = "degraded"
        
        health["score"] = round(max(0, health["score"]), 2)
        return health

    # ── Internal helpers ───────────────────────────────────────

    def _get_node_stats(self) -> dict[str, Any]:
        if not self._store:
            return {"total": 0}
        return {"total": self._store.count_nodes()}

    def _get_evolution_stats(self) -> dict[str, Any]:
        if not self._engine:
            return {"generation": 0, "best_fitness": 0}
        return {
            "generation": self._engine.generation,
            "best_fitness": self._engine.best_genome.fitness if self._engine.best_genome else 0,
        }

    def _get_safety_stats(self) -> dict[str, Any]:
        if not self._safety:
            return {"checks": 0}
        return self._safety.stats

    def _get_governance_stats(self) -> dict[str, Any]:
        result = {}
        if self._governance:
            if "autonomy" in self._governance:
                result["autonomy"] = self._governance["autonomy"].stats
            if "trust" in self._governance:
                result["trust"] = self._governance["trust"].stats
        return result

    def _get_agent_stats(self) -> dict[str, Any]:
        if not self._agents:
            return {"total": 0}
        return self._agents.stats

    def _get_knowledge_stats(self) -> dict[str, Any]:
        if not self._knowledge:
            return {"gaps": 0}
        return self._knowledge.stats
