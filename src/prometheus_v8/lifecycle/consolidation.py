"""Consolidation - Working→Episodic→Semantic→Procedural pipeline."""

from __future__ import annotations

import logging
import time

from prometheus_v8.schema import MemoryLayer, Node, Provenance, ProvenanceType

logger = logging.getLogger(__name__)

CONSOLIDATION_RULES = {
    (MemoryLayer.WORKING, MemoryLayer.EPISODIC): {"min_access": 3, "min_importance": 0.3},
    (MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC): {"min_access": 5, "min_importance": 0.5},
    (MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL): {"min_access": 10, "min_importance": 0.7},
    (MemoryLayer.PROCEDURAL, MemoryLayer.ARCHIVE): {"min_importance": 0.8, "age_days": 30},
}


class ConsolidationPipeline:
    """4-stage consolidation: Working→Episodic→Semantic→Procedural→Archive."""

    def __init__(self, store=None, event_bus=None) -> None:
        self._store = store
        self._event_bus = event_bus
        self._consolidated_count = 0
        self._merged_count = 0

    def consolidate(self, nodes: list[Node]) -> list[Node]:
        """Run consolidation pipeline on a list of nodes."""
        results = []
        for node in nodes:
            next_layer = self._get_next_layer(node)
            if next_layer:
                consolidated = self._promote(node, next_layer)
                if consolidated:
                    results.append(consolidated)
                    self._consolidated_count += 1
            else:
                results.append(node)

        # Merge similar nodes at same layer
        merged = self._merge_similar(results)
        self._merged_count += len(merged) - len(results) if len(merged) < len(results) else 0

        return merged

    def _get_next_layer(self, node: Node) -> MemoryLayer | None:
        """Determine if node should be promoted to next layer."""
        layer_order = [MemoryLayer.WORKING, MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL]
        try:
            idx = layer_order.index(node.layer)
        except ValueError:
            return None

        if idx >= len(layer_order) - 1:
            return None

        next_layer = layer_order[idx + 1]
        rules = CONSOLIDATION_RULES.get((node.layer, next_layer), {})

        if node.access_count < rules.get("min_access", 0):
            return None
        if node.importance < rules.get("min_importance", 0):
            return None

        return next_layer

    def _promote(self, node: Node, next_layer: MemoryLayer) -> Node | None:
        """Promote node to next memory layer."""
        import copy

        promoted = copy.deepcopy(node)
        promoted.layer = next_layer
        promoted.updated_at = time.time()
        promoted.decay_hits()  # Reset consecutive hits after promotion
        promoted.provenance = Provenance(source=ProvenanceType.CONSOLIDATION, confidence=0.8)

        if self._store:
            self._store.update_node(promoted)

        return promoted

    def _merge_similar(self, nodes: list[Node]) -> list[Node]:
        """Merge nodes with identical or near-identical content."""
        if len(nodes) <= 1:
            return nodes

        seen_checksums: dict[str, Node] = {}
        result = []

        for node in nodes:
            checksum = node.payload.checksum
            if checksum in seen_checksums:
                existing = seen_checksums[checksum]
                # Merge: keep higher importance
                if node.importance > existing.importance:
                    existing.importance = node.importance
                existing.access_count += node.access_count
                # Delete duplicate
                if self._store:
                    self._store.delete_node(node.id)
            else:
                seen_checksums[checksum] = node
                result.append(node)

        return result

    @property
    def stats(self) -> dict[str, int]:
        return {"consolidated": self._consolidated_count, "merged": self._merged_count}
