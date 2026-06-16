"""Curiosity Queue - Priority-based exploration queue."""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CuriosityItem:
    """An item in the curiosity queue."""

    question: str = ""
    priority: int = 5  # 1=highest, 10=lowest
    source: str = ""
    created_at: float = field(default_factory=time.time)
    explored: bool = False
    result: str = ""
    exploration_count: int = 0

    def __lt__(self, other: CuriosityItem) -> bool:
        return self.priority < other.priority


class CuriosityQueue:
    """Priority-based curiosity queue for autonomous exploration."""

    def __init__(self, max_size: int = 50) -> None:
        self._max_size = max_size
        self._queue: list[CuriosityItem] = []
        self._lock = threading.RLock()
        self._explored_count = 0

    def add(self, question: str, priority: int = 5, source: str = "") -> CuriosityItem:
        """Add a question to the curiosity queue."""
        with self._lock:
            item = CuriosityItem(question=question, priority=priority, source=source)
            heapq.heappush(self._queue, item)
            # Trim to max size
            while len(self._queue) > self._max_size:
                heapq.heappop(self._queue)
            return item

    def pop(self) -> CuriosityItem | None:
        """Get the highest priority unexplored item."""
        with self._lock:
            while self._queue:
                item = heapq.heappop(self._queue)
                if not item.explored:
                    return item
            return None

    def peek(self) -> CuriosityItem | None:
        """Peek at the highest priority item without removing."""
        with self._lock:
            unexplored = [i for i in self._queue if not i.explored]
            return unexplored[0] if unexplored else None

    def mark_explored(self, question: str, result: str = "") -> None:
        """Mark a question as explored."""
        with self._lock:
            for item in self._queue:
                if item.question == question and not item.explored:
                    item.explored = True
                    item.result = result
                    item.exploration_count += 1
                    self._explored_count += 1
                    break

    def replenish(self, items: list[tuple[str, int]]) -> int:
        """Add multiple items to the queue."""
        added = 0
        for question, priority in items:
            self.add(question, priority)
            added += 1
        return added

    def get_pending(self) -> list[CuriosityItem]:
        with self._lock:
            return sorted([i for i in self._queue if not i.explored], key=lambda i: i.priority)

    def get_explored(self) -> list[CuriosityItem]:
        with self._lock:
            return [i for i in self._queue if i.explored]

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            pending = sum(1 for i in self._queue if not i.explored)
            return {
                "total": len(self._queue),
                "pending": pending,
                "explored": self._explored_count,
                "max_size": self._max_size,
            }
