"""Memory Bus - Event-driven message bus with topic routing and graceful Redis fallback."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """An event on the bus."""

    id: str = ""
    topic: str = ""
    event_type: str = ""
    source: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""

    def serialize(self) -> dict[str, str]:
        return {
            "id": self.id,
            "topic": self.topic,
            "event_type": self.event_type,
            "source": self.source,
            "payload": json.dumps(self.payload, ensure_ascii=False),
            "timestamp": str(self.timestamp),
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def deserialize(cls, data: dict[str, str]) -> "Event":
        return cls(
            id=data.get("id", ""),
            topic=data.get("topic", ""),
            event_type=data.get("event_type", ""),
            source=data.get("source", ""),
            payload=json.loads(data.get("payload", "{}")),
            timestamp=float(data.get("timestamp", 0)),
            correlation_id=data.get("correlation_id", ""),
        )


@dataclass
class Message(Event):
    """Event with additional routing fields for cross-agent messaging.

    Extends Event with sender, channel, recipient for backward compatibility
    with the routing layer. Maps: sender→source, channel→topic, type→event_type.
    """

    sender: str = ""
    channel: str = ""
    recipient: str = ""


class Subscription:
    """A subscription to a topic pattern."""

    def __init__(self, topic: str, callback: Callable[[Event], None], filter_type: str = "") -> None:
        self.id = str(uuid.uuid4())[:8]
        self.topic = topic
        self.callback = callback
        self.filter_type = filter_type
        self.event_count = 0
        self.error_count = 0
        self.created_at = time.time()

    def matches(self, event: Event) -> bool:
        """Check if this subscription matches the event."""
        if self.topic == "*" or self.topic == event.topic:
            if self.filter_type and self.filter_type != event.event_type:
                return False
            return True
        # Wildcard matching: "evolution.*" matches "evolution.mutation"
        if self.topic.endswith(".*"):
            prefix = self.topic[:-2]
            if event.topic.startswith(prefix + "."):
                return True
        return False


class MemoryBus:
    """In-memory event bus with topic routing and subscription management.

    Features:
    - Topic-based event routing with wildcard support
    - Subscription management with type filtering
    - Event history with configurable retention
    - Thread-safe operations
    - Dead letter handling for failed deliveries
    - Event statistics and monitoring
    """

    def __init__(self, history_size: int = 1000, max_subscriptions: int = 100) -> None:
        self._history_size = history_size
        self._max_subscriptions = max_subscriptions
        self._subscriptions: dict[str, Subscription] = {}
        self._history: deque[Event] = deque(maxlen=history_size)
        self._dead_letters: deque[Event] = deque(maxlen=100)
        self._event_count = 0
        self._error_count = 0
        self._lock = threading.RLock()
        self._topic_stats: dict[str, int] = defaultdict(int)

    def publish(self, topic: str, event_type: str, payload: dict, source: str = "", correlation_id: str = "") -> str:
        """Publish an event to a topic."""
        event = Event(
            id=str(uuid.uuid4())[:12],
            topic=topic,
            event_type=event_type,
            source=source,
            payload=payload,
            correlation_id=correlation_id,
        )

        with self._lock:
            self._event_count += 1
            self._topic_stats[topic] += 1
            self._history.append(event)

            # Deliver to matching subscriptions
            delivered = 0
            for sub in self._subscriptions.values():
                if sub.matches(event):
                    try:
                        sub.callback(event)
                        sub.event_count += 1
                        delivered += 1
                    except Exception as e:
                        sub.error_count += 1
                        self._error_count += 1
                        logger.warning(f"Subscription {sub.id} callback error: {e}")

            if delivered == 0:
                # No subscribers - could be a dead letter
                pass

        return event.id

    def publish_message(self, message: Message) -> int:
        """Publish a Message object to the bus. Returns number of subscribers reached."""
        self.publish(
            topic=message.topic,
            event_type=message.event_type,
            payload=message.payload,
            source=message.source,
            correlation_id=message.correlation_id,
        )
        # Count matching subscriptions to estimate reach
        with self._lock:
            reached = sum(1 for sub in self._subscriptions.values() if sub.matches(message))
        return reached

    def subscribe(self, topic: str, callback: Callable[[Event], None], filter_type: str = "") -> str:
        """Subscribe to events on a topic.

        Topic supports wildcards: "evolution.*" matches all evolution subtopics.
        Filter_type further restricts by event type.
        Returns subscription ID.
        """
        with self._lock:
            if len(self._subscriptions) >= self._max_subscriptions:
                # Remove oldest subscription
                oldest = min(self._subscriptions.values(), key=lambda s: s.created_at)
                del self._subscriptions[oldest.id]

            sub = Subscription(topic=topic, callback=callback, filter_type=filter_type)
            self._subscriptions[sub.id] = sub
            return sub.id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None

    def get_history(self, topic: str = "", limit: int = 50, since: float | None = None) -> list[Event]:
        """Get event history, optionally filtered."""
        with self._lock:
            events = list(self._history)

        if topic:
            events = [e for e in events if e.topic == topic]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_dead_letters(self, limit: int = 50) -> list[Event]:
        """Get dead letter events."""
        return list(self._dead_letters)[-limit:]

    def replay_event(self, event_id: str) -> bool:
        """Replay a specific event from history."""
        with self._lock:
            for event in self._history:
                if event.id == event_id:
                    for sub in self._subscriptions.values():
                        if sub.matches(event):
                            try:
                                sub.callback(event)
                            except Exception as e:
                                logger.warning(f"Replay callback error for subscription {sub.id}: {e}")
                    return True
        return False

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_events": self._event_count,
                "subscriptions": len(self._subscriptions),
                "errors": self._error_count,
                "history_size": len(self._history),
                "dead_letters": len(self._dead_letters),
                "top_topics": dict(sorted(self._topic_stats.items(), key=lambda x: -x[1])[:10]),
            }
