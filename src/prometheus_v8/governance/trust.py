"""Trust System - 3-level trust annotation + action hooks."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v8.schema import ActionHook, Node, TrustLevel

logger = logging.getLogger(__name__)


@dataclass
class TrustRecord:
    """Trust record for a knowledge item."""

    node_id: str = ""
    level: TrustLevel = TrustLevel.PENDING
    sources: list[str] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    last_used: float = 0.0
    last_verified: float = 0.0
    action_hook: ActionHook | None = None


class TrustManager:
    """3-level trust annotation system with upgrade/downgrade paths.

    PENDING → HIGH_SIGNAL (2+ independent cross-validated sources)
    HIGH_SIGNAL → VERIFIED (actually used and proven effective)
    VERIFIED → HIGH_SIGNAL (effect not as expected)
    Any → DELETE (30 days unused and utility < 2)
    """

    def __init__(
        self,
        high_signal_sources: int = 2,
        verified_usage_count: int = 1,
        stale_days: int = 30,
        stale_utility_threshold: int = 2,
    ) -> None:
        self._high_signal_sources = high_signal_sources
        self._verified_usage_count = verified_usage_count
        self._stale_days = stale_days
        self._stale_utility = stale_utility_threshold
        self._records: dict[str, TrustRecord] = {}
        self._lock = threading.RLock()

    def annotate(self, node: Node, sources: list[str] | None = None) -> TrustLevel:
        """Annotate a node with trust level based on sources and history."""
        record = self._get_or_create(node.id.hex())

        if sources:
            record.sources = list(set(record.sources + sources))

        # Determine trust level
        if len(record.sources) >= self._high_signal_sources and record.level == TrustLevel.PENDING:
            record.level = TrustLevel.HIGH_SIGNAL
            logger.info(f"Node {node.id.hex()[:8]} upgraded to HIGH_SIGNAL ({len(record.sources)} sources)")

        if record.success_count >= self._verified_usage_count and record.level == TrustLevel.HIGH_SIGNAL:
            record.level = TrustLevel.VERIFIED
            logger.info(f"Node {node.id.hex()[:8]} upgraded to VERIFIED ({record.success_count} successes)")

        # Update node
        node.trust_level = record.level
        return record.level

    def record_usage(self, node_id: str, success: bool) -> TrustLevel | None:
        """Record usage of a knowledge item."""
        with self._lock:
            record = self._records.get(node_id)
            if not record:
                return None

            record.usage_count += 1
            record.last_used = time.time()

            if success:
                record.success_count += 1
                # Upgrade check
                if record.level == TrustLevel.HIGH_SIGNAL and record.success_count >= self._verified_usage_count:
                    record.level = TrustLevel.VERIFIED
            else:
                record.fail_count += 1
                # Downgrade check
                if record.level == TrustLevel.VERIFIED and record.fail_count > record.success_count:
                    record.level = TrustLevel.HIGH_SIGNAL
                    logger.info(f"Node {node_id[:8]} downgraded to HIGH_SIGNAL")

        return record.level

    def check_stale(self) -> list[str]:
        """Find stale nodes that should be deleted (30 days unused, utility < 2)."""
        now = time.time()
        stale = []
        with self._lock:
            for nid, record in self._records.items():
                days_unused = (now - record.last_used) / 86400 if record.last_used > 0 else 999
                utility = record.usage_count
                if days_unused > self._stale_days and utility < self._stale_utility:
                    stale.append(nid)
        return stale

    def set_action_hook(self, node_id: str, trigger: str, action: str, priority: int = 5) -> None:
        """Set an action hook for a knowledge item."""
        with self._lock:
            record = self._get_or_create(node_id)
            record.action_hook = ActionHook(trigger=trigger, action=action, priority=priority)

    def check_action_hooks(self, context: str) -> list[ActionHook]:
        """Check which action hooks should trigger given current context."""
        triggered = []
        with self._lock:
            for record in self._records.values():
                if record.action_hook and record.action_hook.should_trigger(context):
                    triggered.append(record.action_hook)
                    record.action_hook.record_trigger()
        return sorted(triggered, key=lambda h: h.priority)

    def _get_or_create(self, node_id: str) -> TrustRecord:
        if node_id not in self._records:
            self._records[node_id] = TrustRecord(node_id=node_id)
        return self._records[node_id]

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            levels = {l: 0 for l in TrustLevel}
            for r in self._records.values():
                levels[r.level] += 1
            return {"total": len(self._records), "levels": {l.value: c for l, c in levels.items()}}
