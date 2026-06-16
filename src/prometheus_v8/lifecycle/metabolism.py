"""Metabolism - Memory gravity + triage + decay + audit."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from prometheus_v8.schema import MemoryLayer, Node

logger = logging.getLogger(__name__)


class TriageDecision(str, Enum):
    PROMOTE = "promote"  # Upgrade to higher layer
    KEEP = "keep"  # Maintain current layer
    DECAY = "decay"  # Reduce importance
    DELETE = "delete"  # Remove from store
    ARCHIVE = "archive"  # Move to archive


@dataclass
class TriageResult:
    node_id: bytes = b""
    decision: TriageDecision = TriageDecision.KEEP
    reason: str = ""
    score: float = 0.0
    target_layer: MemoryLayer | None = None


class MetabolismEngine:
    """Memory metabolism: gravity + triage + decay + consolidate + audit.

    Memory Gravity Formula:
        G(node) = importance × (1 + log(1 + access_count)) × retention × freshness
        where freshness = exp(-(age_days / λ))
    """

    def __init__(self, store=None, decay_rate: float = 0.95, min_importance_keep: float = 0.3) -> None:
        self._store = store
        self._decay_rate = decay_rate
        self._min_importance_keep = min_importance_keep
        self._audit_log: deque[dict] = deque(maxlen=1000)
        self._triage_count = 0

    def compute_gravity(self, node: Node) -> float:
        """Compute memory gravity score (higher = more valuable)."""
        importance = node.importance
        access_factor = 1.0 + math.log(1 + node.access_count)
        retention = node.retention
        age_days = node.age_days
        freshness = math.exp(-age_days / max(1.0, node.weibull.lam))
        return importance * access_factor * retention * freshness

    def triage(self, node: Node) -> TriageResult:
        """Decide what to do with a memory node."""
        self._triage_count += 1
        gravity = self.compute_gravity(node)

        # Decision logic
        if gravity > 0.7 and node.access_count > 10:
            return TriageResult(
                node_id=node.id,
                decision=TriageDecision.PROMOTE,
                reason=f"high gravity ({gravity:.3f}), frequent access",
                score=gravity,
                target_layer=self._next_layer(node.layer),
            )

        if gravity > 0.3:
            return TriageResult(
                node_id=node.id, decision=TriageDecision.KEEP, reason=f"adequate gravity ({gravity:.3f})", score=gravity
            )

        if gravity > 0.1:
            return TriageResult(
                node_id=node.id, decision=TriageDecision.DECAY, reason=f"low gravity ({gravity:.3f})", score=gravity
            )

        if node.age_days > 90 and node.access_count < 2:
            return TriageResult(
                node_id=node.id,
                decision=TriageDecision.DELETE,
                reason=f"neglected ({gravity:.3f}), old ({node.age_days:.0f}d)",
                score=gravity,
            )

        if node.age_days > 30 and node.importance > 0.5:
            return TriageResult(
                node_id=node.id,
                decision=TriageDecision.ARCHIVE,
                reason=f"valuable but old ({node.age_days:.0f}d)",
                score=gravity,
                target_layer=MemoryLayer.ARCHIVE,
            )

        return TriageResult(
            node_id=node.id, decision=TriageDecision.DECAY, reason=f"very low gravity ({gravity:.3f})", score=gravity
        )

    def decay(self, node: Node) -> Node:
        """Apply exponential decay to node importance."""
        node.importance *= self._decay_rate
        node.updated_at = time.time()
        return node

    def run_metabolism_cycle(self, nodes: list[Node]) -> dict[str, int]:
        """Run full metabolism cycle on all nodes."""
        results = {"promoted": 0, "kept": 0, "decayed": 0, "deleted": 0, "archived": 0}

        for node in nodes:
            triage_result = self.triage(node)

            self._audit_log.append(
                {
                    "node_id": node.id.hex(),
                    "decision": triage_result.decision.value,
                    "gravity": triage_result.score,
                    "reason": triage_result.reason,
                    "timestamp": time.time(),
                }
            )

            if triage_result.decision == TriageDecision.PROMOTE:
                if triage_result.target_layer:
                    node.layer = triage_result.target_layer
                results["promoted"] += 1
            elif triage_result.decision == TriageDecision.KEEP:
                results["kept"] += 1
            elif triage_result.decision == TriageDecision.DECAY:
                self.decay(node)
                results["decayed"] += 1
            elif triage_result.decision == TriageDecision.DELETE:
                if self._store:
                    self._store.delete_node(node.id)
                results["deleted"] += 1
                continue
            elif triage_result.decision == TriageDecision.ARCHIVE:
                node.layer = MemoryLayer.ARCHIVE
                results["archived"] += 1

            if self._store:
                self._store.update_node(node)

        return results

    def dedup(self, nodes: list[Node]) -> list[Node]:
        """Remove duplicate nodes by checksum."""
        seen: dict[str, Node] = {}
        for node in nodes:
            ck = node.payload.checksum
            if ck not in seen or node.importance > seen[ck].importance:
                seen[ck] = node
        return list(seen.values())

    @property
    def stats(self) -> dict[str, Any]:
        return {"triage_count": self._triage_count, "audit_entries": len(self._audit_log)}

    @property
    def audit_log(self) -> list[dict]:
        return list(self._audit_log)[-100:]
