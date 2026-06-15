"""Dashboard Data Provider for visualization."""
from __future__ import annotations
import time
from typing import Any

class DashboardProvider:
    """Provide dashboard data for the web UI."""
    
    def __init__(self, store=None, engine=None, safety=None, monitor=None) -> None:
        self._store = store
        self._engine = engine
        self._safety = safety
        self._monitor = monitor
    
    def get_overview(self) -> dict[str, Any]:
        """Get system overview for dashboard."""
        overview = {
            "version": "8.0.0",
            "timestamp": time.time(),
            "nodes": {"total": 0},
            "evolution": {"generation": 0, "best_fitness": 0},
            "safety": {"checks": 0, "violation_rate": 0},
        }
        
        if self._store:
            overview["nodes"]["total"] = self._store.count_nodes()
        
        if self._engine:
            overview["evolution"]["generation"] = self._engine.generation
            if self._engine.best_genome:
                overview["evolution"]["best_fitness"] = round(self._engine.best_genome.fitness, 4)
        
        if self._safety:
            overview["safety"] = self._safety.stats
        
        return overview
    
    def get_memory_distribution(self) -> dict[str, int]:
        """Get node count by memory layer."""
        from prometheus_v8.schema import MemoryLayer
        dist = {}
        if self._store:
            for layer in MemoryLayer:
                dist[layer.value] = self._store.count_nodes(layer)
        return dist
    
    def get_evolution_progress(self) -> dict[str, Any]:
        """Get evolution progress data."""
        if not self._engine:
            return {}
        return {
            "generation": self._engine.generation,
            "layer_stats": self._engine.layer_stats,
            "history": self._engine.history[:20],
        }
