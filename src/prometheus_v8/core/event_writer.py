"""Event-driven writeback - MemoryEventBus + EventDrivenWriter."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MemoryEventBus:
    """Simple publish/subscribe event bus for memory system events."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe a handler to an event type."""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h is not handler
                ]

    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> int:
        """Publish an event. Returns number of handlers invoked."""
        data = data or {}
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))
        count = 0
        for handler in handlers:
            try:
                handler(event_type, data)
                count += 1
            except Exception as e:
                logger.warning(f"Event handler error for '{event_type}': {e}")
        return count

    @property
    def handler_count(self) -> dict[str, int]:
        """Return number of handlers per event type."""
        with self._lock:
            return {k: len(v) for k, v in self._handlers.items()}


class EventDrivenWriter:
    """Listens to memory events and batch-writes to store.

    Monitors: node_accessed, node_created, insight_generated
    Buffer: deque(maxlen=100)
    Flush strategy: auto-flush when buffer reaches batch_size (default 5)
    """

    def __init__(
        self,
        store=None,
        event_bus: MemoryEventBus | None = None,
        batch_size: int = 5,
    ) -> None:
        self._store = store
        self._event_bus = event_bus or MemoryEventBus()
        self._batch_size = batch_size
        self._buffer: deque[dict[str, Any]] = deque(maxlen=100)
        self._lock = threading.RLock()
        self._total_written = 0

        # Subscribe to events
        self._event_bus.subscribe("node_accessed", self._on_event)
        self._event_bus.subscribe("node_created", self._on_event)
        self._event_bus.subscribe("insight_generated", self._on_event)

    def _on_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Handle incoming event: buffer it and maybe flush."""
        with self._lock:
            entry = {
                "event_type": event_type,
                "data": data,
                "timestamp": time.time(),
            }
            self._buffer.append(entry)
            if len(self._buffer) >= self._batch_size:
                self._do_flush()

    def _do_flush(self) -> int:
        """Internal flush: write buffered events to store."""
        if not self._store or not self._buffer:
            return 0
        # Drain buffer
        items = list(self._buffer)
        self._buffer.clear()
        count = 0
        for item in items:
            try:
                node = item.get("data", {}).get("node")
                if node is not None:
                    self._store.update_node(node)
                    count += 1
            except Exception as e:
                logger.debug(f"EventDrivenWriter flush error: {e}")
        self._total_written += count
        return count

    def flush(self) -> int:
        """Force flush all buffered events to store."""
        with self._lock:
            return self._do_flush()

    @property
    def buffer_size(self) -> int:
        """Current number of items in buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "buffer_size": self.buffer_size,
            "total_written": self._total_written,
            "batch_size": self._batch_size,
        }
