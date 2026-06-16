"""Aging Detection - 4-dimension aging: compression/interference/revision/maintenance."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from prometheus_v8.schema import Node

logger = logging.getLogger(__name__)


@dataclass
class AgingReport:
    """4-dimension aging assessment."""

    node_id: str = ""
    compression_aging: float = 0.0  # 0=healthy, 1=severely aged
    interference_aging: float = 0.0
    revision_aging: float = 0.0
    maintenance_aging: float = 0.0
    composite_aging: float = 0.0
    recommendation: str = ""


class AgingDetector:
    """Detect 4 types of memory aging:
    1. Compression aging: Over-summarized content loses critical details
    2. Interference aging: Similar memories confuse each other
    3. Revision aging: Frequent edits degrade stability
    4. Maintenance aging: Stale content becomes outdated
    """

    def __init__(self, store=None) -> None:
        self._store = store
        self._reports: deque[AgingReport] = deque(maxlen=500)

    def assess(self, node: Node, similar_nodes: list[Node] | None = None) -> AgingReport:
        """Assess aging of a single node."""
        report = AgingReport(node_id=node.id.hex())

        # 1. Compression aging: content too short relative to metadata
        content_length = len(node.payload.content)
        metadata_size = len(str(node.metadata))
        if content_length < 50 and metadata_size > content_length * 2:
            report.compression_aging = min(1.0, metadata_size / max(1, content_length) * 0.1)

        # 2. Interference aging: too many similar nodes
        if similar_nodes:
            n_similar = len(similar_nodes)
            report.interference_aging = min(1.0, n_similar * 0.1)

        # 3. Revision aging: too many updates
        update_ratio = node.access_count / max(1, node.age_days)
        if update_ratio > 5:  # More than 5 updates per day
            report.revision_aging = min(1.0, update_ratio * 0.05)

        # 4. Maintenance aging: stale content
        days_since_update = (time.time() - node.updated_at) / 86400
        report.maintenance_aging = min(1.0, days_since_update / 90)  # 90 days = fully aged

        # Composite
        report.composite_aging = (
            0.2 * report.compression_aging
            + 0.3 * report.interference_aging
            + 0.2 * report.revision_aging
            + 0.3 * report.maintenance_aging
        )

        # Recommendation
        if report.composite_aging > 0.7:
            report.recommendation = "archive_or_delete"
        elif report.composite_aging > 0.4:
            report.recommendation = "refresh_or_consolidate"
        else:
            report.recommendation = "healthy"

        self._reports.append(report)
        return report

    def assess_batch(self, nodes: list[Node]) -> list[AgingReport]:
        """Assess aging for a batch of nodes."""
        reports = []
        for node in nodes:
            # Find similar nodes for interference check
            similar = [n for n in nodes if n.id != node.id and self._is_similar(node, n)]
            reports.append(self.assess(node, similar))
        return reports

    def _is_similar(self, n1: Node, n2: Node) -> bool:
        """Quick similarity check."""
        if n1.type != n2.type:
            return False
        w1 = set(n1.payload.content.lower().split()[:20])
        w2 = set(n2.payload.content.lower().split()[:20])
        if not w1 or not w2:
            return False
        return len(w1 & w2) / len(w1 | w2) > 0.5

    @property
    def reports(self) -> list[AgingReport]:
        return list(self._reports)[-100:]

    @property
    def stats(self) -> dict[str, Any]:
        if not self._reports:
            return {"total_assessed": 0}
        avg_aging = sum(r.composite_aging for r in self._reports) / len(self._reports)
        return {"total_assessed": len(self._reports), "avg_aging": round(avg_aging, 3)}
