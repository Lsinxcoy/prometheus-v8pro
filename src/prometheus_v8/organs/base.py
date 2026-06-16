"""Organ Base - LLMClient, Tool, OrganEnv, OrganContext, BaseOrgan."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """External tool definition."""

    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)
    execute_fn: Callable[[dict], str] | None = None

    def execute(self, params: dict) -> str:
        if self.execute_fn:
            return self.execute_fn(params)
        return f"Tool {self.name} not implemented"


@dataclass
class OrganContext:
    """Context passed to organ during execution."""

    task: str = ""
    inputs: dict = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    budget_tokens: int = 4000
    budget_time: int = 60
    metadata: dict = field(default_factory=dict)


@dataclass
class OrganResult:
    """Result from organ execution."""

    success: bool = False
    output: Any = None
    error: str = ""
    tokens_used: int = 0
    time_elapsed: float = 0.0
    metadata: dict = field(default_factory=dict)


class LLMClient:
    """HTTP-based LLM client with multi-provider fallback."""

    def __init__(
        self,
        api_base: str = "https://openrouter.ai/api/v1",
        api_key: str = "",
        model: str = "qwen/qwen3-235b-a22b:free",
        timeout: int = 60,
        max_retries: int = 3,
        fallback_model: str = "",
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._fallback_model = fallback_model
        self._client = httpx.Client(timeout=timeout)
        self._call_count = 0
        self._total_tokens = 0

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Send chat completion request. Returns assistant content."""
        models = [self._model]
        if self._fallback_model and self._fallback_model != self._model:
            models.append(self._fallback_model)

        last_error = ""
        for model in models:
            for attempt in range(self._max_retries):
                try:
                    headers = {"Content-Type": "application/json"}
                    if self._api_key:
                        headers["Authorization"] = f"Bearer {self._api_key}"

                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }

                    resp = self._client.post(f"{self._api_base}/chat/completions", json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    self._call_count += 1
                    self._total_tokens += data.get("usage", {}).get("total_tokens", 0)
                    return content
                except Exception as e:
                    last_error = str(e)
                    if attempt < self._max_retries - 1:
                        time.sleep(2**attempt)

        logger.error(f"LLM call failed after all retries: {last_error}")
        return f"[LLM_ERROR] {last_error}"

    def embed(self, text: str) -> list[float]:
        """Get embedding (hash-based fallback if no embedding API)."""
        from prometheus_v8.core.embedder import Embedder

        return Embedder.hash_embed(text)

    @property
    def stats(self) -> dict[str, int]:
        return {"calls": self._call_count, "total_tokens": self._total_tokens}

    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Async wrapper for complete()."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))

    async def aembed(self, text: str) -> list[float]:
        """Async wrapper for embed()."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(text))


class OrganEnv:
    """Execution environment for organs with tool access."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self._sandbox_enabled = False

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def execute_tool(self, name: str, params: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Tool not found: {name}"
        return tool.execute(params)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


class BaseOrgan(ABC):
    """Abstract base class for all organs."""

    def __init__(self, name: str, llm: LLMClient | None = None, env: OrganEnv | None = None) -> None:
        self._name = name
        self._llm = llm or LLMClient()
        self._env = env or OrganEnv()
        self._execution_count = 0
        self._total_time = 0.0

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def execute(self, context: OrganContext) -> OrganResult:
        """Execute organ's primary function."""
        ...

    def _timed_execute(self, context: OrganContext) -> OrganResult:
        start = time.time()
        try:
            result = self.execute(context)
            result.time_elapsed = time.time() - start
            self._execution_count += 1
            self._total_time += result.time_elapsed
            return result
        except Exception as e:
            return OrganResult(success=False, error=str(e), time_elapsed=time.time() - start)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "executions": self._execution_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(1, self._execution_count),
        }
