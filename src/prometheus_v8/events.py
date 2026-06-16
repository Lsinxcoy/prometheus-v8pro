"""Prometheus V8 Event System.

20 event types, PubSub, ConsolidationTrigger, EventLogger, MetricsCollector.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from prometheus_v8.communication.bus import Event as BusEvent

# Re-export BusEvent as Event for backward compatibility with events.py consumers
Event = BusEvent

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    NODE_CREATED = "node_created"
    NODE_DELETED = "node_deleted"
    NODE_UPDATED = "node_updated"
    NODE_ACCESSED = "node_accessed"
    EDGE_CREATED = "edge_created"
    EDGE_DELETED = "edge_deleted"
    CONSOLIDATION_TRIGGERED = "consolidation_triggered"
    CONSOLIDATION_COMPLETED = "consolidation_completed"
    DREAM_STARTED = "dream_started"
    DREAM_COMPLETED = "dream_completed"
    EVOLUTION_STARTED = "evolution_started"
    EVOLUTION_COMPLETED = "evolution_completed"
    MUTATION_APPLIED = "mutation_applied"
    PROMOTION_GRANTED = "promotion_granted"
    PROMOTION_REJECTED = "promotion_rejected"
    SAFETY_VIOLATION = "safety_violation"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
    BROADCAST_SENT = "broadcast_sent"
    BROADCAST_APPROVED = "broadcast_approved"


class EventBus:
    """Thread-safe pub/sub event bus with history and handlers.

    Uses the unified Event from bus.py. EventType values are stored
    in event.event_type as strings.
    """

    def __init__(self, max_history: int = 10000) -> None:
        self._subscribers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._history: list[Event] = []
        self._lock = threading.RLock()
        self._max_history = max_history
        self._event_counts: dict[str, int] = defaultdict(int)
        self._handlers: list[EventHandler] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers[event_type.value].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        with self._lock:
            handlers = self._subscribers.get(event_type.value, [])
            self._subscribers[event_type.value] = [h for h in handlers if h != handler]

    def emit(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
            self._event_counts[event.event_type] += 1

        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

        for eh in self._handlers:
            try:
                eh.handle(event)
            except Exception as e:
                logger.warning(f"EventHandler error: {e}")

    def add_handler(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> list[Event]:
        with self._lock:
            events = self._history
            if event_type:
                events = [e for e in events if e.event_type == event_type.value]
            return events[-limit:]

    def get_metrics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._event_counts)


class EventHandler:
    """Base class for event handlers."""

    def handle(self, event: Event) -> None:
        pass


class ConsolidationTrigger(EventHandler):
    """Auto-trigger consolidation when operation threshold reached."""

    def __init__(self, threshold: int = 100, callback: Optional[Callable] = None) -> None:
        self._count = 0
        self._threshold = threshold
        self._callback = callback

    def handle(self, event: Event) -> None:
        if event.event_type in (
            EventType.NODE_CREATED.value,
            EventType.NODE_UPDATED.value,
            EventType.NODE_ACCESSED.value,
        ):
            self._count += 1
            if self._count >= self._threshold:
                self._count = 0
                consolidation_event = Event(
                    event_type=EventType.CONSOLIDATION_TRIGGERED.value,
                    payload={"triggered_by": "threshold", "count": self._threshold},
                )
                if self._callback:
                    self._callback(consolidation_event)


class EventLogger(EventHandler):
    """JSONL event logger for audit trail."""

    def __init__(self, log_path: str = "data/events.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("prometheus_v8.events.jsonl")

    def handle(self, event: Event) -> None:
        try:
            record = json.dumps(
                {
                    "type": event.event_type,
                    "payload": event.payload,
                    "source": event.source,
                    "timestamp": event.timestamp,
                    "agent_id": event.correlation_id,
                }
            )
            # Use logging instead of direct file I/O for thread safety
            self._logger.info(record)
            # Also write to JSONL file for persistence
            with open(self._path, "a") as f:
                f.write(record + "\n")
        except Exception as e:
            logger.warning(f"EventLogger write error: {e}")


class MetricsCollector(EventHandler):
    """Aggregate event counts per agent."""

    def __init__(self) -> None:
        self._by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def handle(self, event: Event) -> None:
        agent = event.correlation_id or "system"
        self._by_agent[agent][event.event_type] += 1

    def get_agent_metrics(self, agent_id: str) -> dict[str, int]:
        return dict(self._by_agent.get(agent_id, {}))

    def get_all_metrics(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self._by_agent.items()}
