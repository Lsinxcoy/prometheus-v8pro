"""Circuit Breaker - CLOSED→OPEN→HALF_OPEN state machine."""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker with CLOSED→OPEN→HALF_OPEN state machine.

    CLOSED: Normal operation, count failures
    OPEN: All requests blocked, wait for timeout
    HALF_OPEN: Allow one test request, if success → CLOSED, if failure → OPEN
    """

    def __init__(self, threshold: int = 5, timeout: int = 60) -> None:
        self._threshold = threshold
        self._timeout = timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()

    def can_execute(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self._timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker: OPEN → HALF_OPEN")
                    return True
                return False
            if self._state == CircuitState.HALF_OPEN:
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker: HALF_OPEN → CLOSED")

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker: HALF_OPEN → OPEN")
            elif self._failure_count >= self._threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker: CLOSED → OPEN (failures={self._failure_count})")

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "failures": self._failure_count,
                "successes": self._success_count,
                "threshold": self._threshold,
            }

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
