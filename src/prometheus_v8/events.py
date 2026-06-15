"""Prometheus V8 Event System.

20 event types, PubSub, ConsolidationTrigger, EventLogger, MetricsCollector.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

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


@dataclass
class Event:
    """Typed event with payload and metadata."""
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None


class EventBus:
    """Thread-safe pub/sub event bus with history and handlers."""

    def __init__(self, max_history: int = 10000) -> None:
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self._history: list[Event] = []
        self._lock = threading.RLock()
        self._max_history = max_history
        self._event_counts: dict[str, int] = defaultdict(int)
        self._handlers: list[EventHandler] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        with self._lock:
            handlers = self._subscribers.get(event_type, [])
            self._subscribers[event_type] = [h for h in handlers if h != handler]

    def emit(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            self._event_counts[event.type.value] += 1

        handlers = self._subscribers.get(event.type, []) + self._subscribers.get(EventType.NODE_CREATED, [])
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
                events = [e for e in events if e.type == event_type]
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
        if event.type in (EventType.NODE_CREATED, EventType.NODE_UPDATED, EventType.NODE_ACCESSED):
            self._count += 1
            if self._count >= self._threshold:
                self._count = 0
                consolidation_event = Event(
                    type=EventType.CONSOLIDATION_TRIGGERED,
                    payload={"triggered_by": "threshold", "count": self._threshold},
                )
                if self._callback:
                    self._callback(consolidation_event)


class EventLogger(EventHandler):
    """JSONL event logger for audit trail."""

    def __init__(self, log_path: str = "data/events.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, event: Event) -> None:
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps({
                    "type": event.type.value,
                    "payload": event.payload,
                    "source": event.source,
                    "timestamp": event.timestamp,
                    "agent_id": event.agent_id,
                }) + "\n")
        except Exception as e:
            logger.warning(f"EventLogger write error: {e}")


class MetricsCollector(EventHandler):
    """Aggregate event counts per agent."""

    def __init__(self) -> None:
        self._by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def handle(self, event: Event) -> None:
        agent = event.agent_id or "system"
        self._by_agent[agent][event.type.value] += 1

    def get_agent_metrics(self, agent_id: str) -> dict[str, int]:
        return dict(self._by_agent.get(agent_id, {}))

    def get_all_metrics(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self._by_agent.items()}
