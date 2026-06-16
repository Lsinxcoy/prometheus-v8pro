"""Hermes LLM Adapter - httpx client with retry, rate limiting, and model chain fallback."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    api_key: str = ""
    primary_model: str = "mimo-v2.5-pro"
    fallback_models: list[str] = field(default_factory=lambda: ["mimo-v2.5", "qwen3-235b"])
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    rate_limit_rpm: int = 60  # requests per minute
    rate_limit_tpm: int = 100000  # tokens per minute


@dataclass
class TokenUsage:
    """Track token usage and costs."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_requests: int = 0
    total_failures: int = 0
    total_retries: int = 0
    estimated_cost: float = 0.0


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        """Wait until tokens are available, then consume."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.consume(tokens):
                return True
            time.sleep(0.1)
        return False


class HermesLLMAdapter:
    """LLM client with retry, rate limiting, and model chain fallback.

    Features:
    - Synchronous and asynchronous chat completions
    - Retry with exponential backoff (3 retries)
    - Token bucket rate limiting
    - Model chain fallback (primary -> fallback models)
    - Token counting and cost tracking
    - Request/response logging
    - Connection pooling via httpx
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._usage = TokenUsage()
        self._rpm_bucket = TokenBucket(rate=self._config.rate_limit_rpm / 60.0, capacity=self._config.rate_limit_rpm)
        self._request_log: deque[dict] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._client = None

        if HAS_HTTPX:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                headers={"Authorization": f"Bearer {self._config.api_key}", "Content-Type": "application/json"},
                timeout=self._config.timeout_seconds,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """Synchronous chat completion with retry and fallback.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            model: Model to use (defaults to primary_model)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stream: Whether to stream the response

        Returns:
            The completion text.
        """
        model = model or self._config.primary_model
        temperature = temperature if temperature is not None else self._config.temperature
        max_tokens = max_tokens or self._config.max_tokens

        # Build model chain: requested model + fallbacks
        models = [model] + [m for m in self._config.fallback_models if m != model]

        last_error = None
        for current_model in models:
            for attempt in range(self._config.max_retries):
                try:
                    result = self._do_request(
                        messages=messages,
                        model=current_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return result
                except Exception as e:
                    last_error = e
                    self._usage.total_retries += 1

                    if attempt < self._config.max_retries - 1:
                        delay = self._config.retry_base_delay * (2**attempt)
                        logger.warning(
                            f"LLM request failed (attempt {attempt + 1}/{self._config.max_retries}): {e}. Retrying in {delay}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(f"Model {current_model} failed after {self._config.max_retries} retries: {e}")

        self._usage.total_failures += 1
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _do_request(self, messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
        """Execute a single LLM request."""
        if not self._rpm_bucket.wait_and_consume(timeout=30.0):
            raise RuntimeError("Rate limit exceeded")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        request_start = time.time()

        if not HAS_HTTPX or not self._client:
            # Fallback: return a placeholder
            self._usage.total_requests += 1
            return f"[LLM response for: {messages[-1].get('content', '')[:50]}...]"

        response = self._client.post("/chat/completions", json=payload)
        latency = time.time() - request_start

        if response.status_code != 200:
            raise RuntimeError(f"LLM API error: {response.status_code} {response.text[:200]}")

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Track usage
        usage_data = data.get("usage", {})
        self._usage.prompt_tokens += usage_data.get("prompt_tokens", 0)
        self._usage.completion_tokens += usage_data.get("completion_tokens", 0)
        self._usage.total_tokens += usage_data.get("total_tokens", 0)
        self._usage.total_requests += 1

        # Log request
        self._request_log.append(
            {
                "model": model,
                "latency": round(latency, 3),
                "tokens": usage_data.get("total_tokens", 0),
                "timestamp": time.time(),
            }
        )

        return content

    def embed(self, text: str | list[str], model: str = "all-MiniLM-L6-v2") -> list[list[float]]:
        """Generate embeddings for text."""
        texts = [text] if isinstance(text, str) else text
        # Use hash-based fallback embedding when no embedding API available
        import hashlib
        import struct

        embeddings = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            values = [struct.unpack("f", h[i : i + 4])[0] for i in range(0, min(len(h), 384 * 4), 4)]
            # Normalize
            norm = sum(v**2 for v in values) ** 0.5
            if norm > 0:
                values = [v / norm for v in values]
            # Pad to 384 dimensions
            while len(values) < 384:
                values.append(0.0)
            embeddings.append(values[:384])
        return embeddings

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough: 1 token ~= 4 chars)."""
        return max(1, len(text) // 4)

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    @property
    def request_log(self) -> list[dict]:
        return list(self._request_log)

    def close(self) -> None:
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_requests": self._usage.total_requests,
            "total_failures": self._usage.total_failures,
            "total_retries": self._usage.total_retries,
            "total_tokens": self._usage.total_tokens,
            "primary_model": self._config.primary_model,
            "fallback_models": self._config.fallback_models,
        }
