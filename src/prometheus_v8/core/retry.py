"""Exponential backoff retry with temperature cooling and fallback degradation."""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


class ExponentialBackoff:
    """Exponential backoff delay calculator with jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._attempt = 0

    def next_delay(self) -> float:
        """Calculate the next delay with optional jitter."""
        delay = min(self.base_delay * (2 ** self._attempt), self.max_delay)
        if self.jitter:
            delay *= random.uniform(0.5, 1.0)
        self._attempt += 1
        return delay

    def reset(self) -> None:
        """Reset attempt counter."""
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt


class TemperatureCooler:
    """Reduce LLM temperature on each retry for more conservative responses."""

    def __init__(self, initial_temp: float = 0.7, min_temp: float = 0.1, decay: float = 0.5) -> None:
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.decay = decay
        self._current_temp = initial_temp

    def cool(self) -> float:
        """Return current temperature and cool for next attempt."""
        temp = self._current_temp
        self._current_temp = max(self.min_temp, self._current_temp * self.decay)
        return temp

    def reset(self) -> None:
        self._current_temp = self.initial_temp

    @property
    def current_temp(self) -> float:
        return self._current_temp


def with_retry(
    func: Callable | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    fallback: Any = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """Decorator: retry with exponential backoff.

    Can be used as:
        @with_retry
        def my_func(): ...

        @with_retry(max_retries=5, fallback=None)
        def my_func(): ...

    Args:
        func: Function to wrap (when used without arguments).
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        fallback: Value to return if all retries fail. If None, re-raises.
        exceptions: Tuple of exception types to catch and retry on.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            backoff = ExponentialBackoff(max_retries=max_retries, base_delay=base_delay)
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = backoff.next_delay()
                        logger.debug(
                            f"Retry {attempt + 1}/{max_retries} for {fn.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
            # All retries exhausted
            if fallback is not None:
                logger.warning(
                    f"All {max_retries} retries failed for {fn.__name__}, "
                    f"returning fallback. Last error: {last_exc}"
                )
                return fallback
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"All retries exhausted for {fn.__name__}")

        return wrapper

    if func is not None:
        # Used as @with_retry without arguments
        return decorator(func)
    # Used as @with_retry(...) with arguments
    return decorator
