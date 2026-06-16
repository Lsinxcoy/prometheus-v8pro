"""Weibull Retention with Recall Reinforcement.

Based on Mnemo + YourMemory research:
- Weibull decay: S(t) = importance × exp(-(t/λ)^k) × (1 + recall_count × 0.2)
- k < 1: rapid initial decay, then slow (like human memory)
- k > 1: slow initial decay, then rapid (skill loss pattern)
- Each recall reinforces and flattens decay (spaced repetition)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from prometheus_v8.schema import MemoryLayer, WeibullParams

# Per-layer Weibull defaults with biological plausibility
WEIBULL_DEFAULTS: dict[MemoryLayer, tuple[float, float, str]] = {
    MemoryLayer.WORKING: (1.0, 0.5, "rapid_decay_slow_tail"),  # k<1: forget fast, some linger
    MemoryLayer.EPISODIC: (7.0, 0.8, "exponential_like"),  # k≈1: steady decay
    MemoryLayer.SEMANTIC: (30.0, 1.2, "resistant_then_rapid"),  # k>1: resist then drop
    MemoryLayer.PROCEDURAL: (365.0, 1.5, "skill_retention"),  # k>1: skills persist
    MemoryLayer.ARCHIVE: (1095.0, 2.0, "near_permanent"),  # k=2: very slow decay
}

# Consolidation thresholds
CONSOLIDATION_THRESHOLDS = {
    "working_to_episodic": {"min_access": 3, "min_importance": 0.3, "max_age_days": 7},
    "episodic_to_semantic": {"min_access": 5, "min_importance": 0.5, "max_age_days": 30},
    "semantic_to_procedural": {"min_access": 10, "min_importance": 0.7, "max_age_days": 90},
    "to_archive": {"min_importance": 0.8, "min_age_days": 30},
}


@dataclass
class WeibullMemoryScore:
    """Complete Weibull memory score with components."""

    raw_retention: float = 0.0
    importance_factor: float = 0.0
    recall_reinforcement: float = 0.0
    composite: float = 0.0
    should_consolidate: bool = False
    consolidation_target: Optional[MemoryLayer] = None


class WeibullRetentionCalculator:
    """Advanced Weibull retention with recall reinforcement and tiered consolidation."""

    def __init__(self) -> None:
        self._defaults = WEIBULL_DEFAULTS

    def compute(
        self, age_days: float, importance: float, lam: float, k: float, consecutive_hits: int = 0, access_count: int = 0
    ) -> WeibullMemoryScore:
        """Compute full Weibull retention score with all factors."""
        # Base Weibull decay
        if age_days <= 0:
            raw_retention = 1.0
        else:
            raw_retention = math.exp(-((age_days / lam) ** k))

        # Importance scaling of lambda (higher importance = slower decay)
        importance_factor = 1.0 + importance * 0.8

        # Recall reinforcement (spaced repetition effect from YourMemory)
        recall_reinforcement = 1.0 + min(consecutive_hits, 20) * 0.2

        # Composite: importance × weibull_decay × recall_boost
        composite = importance * raw_retention * importance_factor * recall_reinforcement
        composite = min(1.0, composite)

        # Check consolidation eligibility
        should_consolidate = False
        target = None
        if access_count >= 3 and importance >= 0.3 and age_days <= 7:
            should_consolidate = True
            target = MemoryLayer.EPISODIC
        if access_count >= 5 and importance >= 0.5 and age_days <= 30:
            should_consolidate = True
            target = MemoryLayer.SEMANTIC
        if access_count >= 10 and importance >= 0.7 and age_days <= 90:
            should_consolidate = True
            target = MemoryLayer.PROCEDURAL

        return WeibullMemoryScore(
            raw_retention=raw_retention,
            importance_factor=importance_factor,
            recall_reinforcement=recall_reinforcement,
            composite=composite,
            should_consolidate=should_consolidate,
            consolidation_target=target,
        )

    def compute_for_node(self, node) -> WeibullMemoryScore:
        """Compute Weibull score for a Node object."""
        return self.compute(
            age_days=node.age_days,
            importance=node.importance,
            lam=node.weibull.lam,
            k=node.weibull.k,
            consecutive_hits=node.consecutive_hits,
            access_count=node.access_count,
        )

    @staticmethod
    def for_layer(layer: MemoryLayer) -> WeibullParams:
        """Get Weibull parameters for a memory layer."""
        defaults = WEIBULL_DEFAULTS.get(layer, (7.0, 0.8, "default"))
        return WeibullParams(lam=defaults[0], k=defaults[1])

    def estimate_half_life(self, lam: float, k: float, importance: float = 1.0) -> float:
        """Estimate the half-life (days until retention = 0.5) for given parameters."""
        # S(t) = importance × exp(-(t/λ)^k) = 0.5
        # exp(-(t/λ)^k) = 0.5 / importance
        if importance <= 0.5:
            return 0.0
        target = 0.5 / importance
        if target >= 1.0:
            return 0.0
        # -(t/λ)^k = ln(target)
        # t = λ × (-ln(target))^(1/k)
        return lam * ((-math.log(target)) ** (1.0 / k))
