"""Hermes LLM Adapter - httpx client with multi-provider fallback."""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
import httpx

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "qwen/qwen3-235b-a22b:free"
    fallback_model: str = ""
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000

class HermesLLMAdapter:
    """HTTP-based LLM adapter with multi-provider fallback."""
    
    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._client = httpx.Client(timeout=self._config.timeout)
        self._call_count = 0
        self._total_tokens = 0
        self._total_time = 0.0
    
    def complete(self, messages: list[dict[str, str]], temperature: float | None = None,
                 max_tokens: int | None = None, model: str | None = None) -> str:
        """Send chat completion request."""
        temp = temperature or self._config.temperature
        mt = max_tokens or self._config.max_tokens
        models = [model or self._config.model]
        if self._config.fallback_model and self._config.fallback_model not in models:
            models.append(self._config.fallback_model)
        
        start = time.time()
        last_error = ""
        for m in models:
            for attempt in range(self._config.max_retries):
                try:
                    headers = {"Content-Type": "application/json"}
                    if self._config.api_key:
                        headers["Authorization"] = f"Bearer {self._config.api_key}"
                    payload = {"model": m, "messages": messages, "temperature": temp, "max_tokens": mt}
                    resp = self._client.post(f"{self._config.api_base.rstrip('/')}/chat/completions",
                                           json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    self._call_count += 1
                    self._total_tokens += data.get("usage", {}).get("total_tokens", 0)
                    self._total_time += time.time() - start
                    return content
                except Exception as e:
                    last_error = str(e)
                    if attempt < self._config.max_retries - 1:
                        time.sleep(2 ** attempt)
        return f"[LLM_ERROR] {last_error}"
    
    def embed(self, text: str) -> list[float]:
        """Get embedding (hash fallback)."""
        import hashlib
        import numpy as np
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
    
    @property
    def stats(self) -> dict[str, Any]:
        return {"calls": self._call_count, "tokens": self._total_tokens,
                "avg_time": self._total_time / max(1, self._call_count)}
