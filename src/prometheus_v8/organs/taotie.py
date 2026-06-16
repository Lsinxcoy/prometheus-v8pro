"""Taotie Organ - Ingest with 7 directions + DNA extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from prometheus_v8.organs.base import BaseOrgan, LLMClient, OrganContext, OrganEnv, OrganResult

logger = logging.getLogger(__name__)

SEVEN_DIRECTIONS = [
    "paper_search",  # arXiv/semantic scholar
    "github_search",  # GitHub repos
    "web_search",  # General web
    "code_analysis",  # Local codebase
    "memory_recall",  # Existing memories
    "pattern_mining",  # Pattern discovery
    "hypothesis_gen",  # Hypothesis generation
]


class TaotieOrgan(BaseOrgan):
    """Ingest organ: consume raw input → extract structured knowledge."""

    def __init__(self, llm: LLMClient | None = None, env: OrganEnv | None = None) -> None:
        super().__init__("taotie", llm, env)
        self._direction_weights: dict[str, float] = {d: 1.0 / len(SEVEN_DIRECTIONS) for d in SEVEN_DIRECTIONS}

    def execute(self, context: OrganContext) -> OrganResult:
        task = context.task
        raw_data = context.inputs

        # 1. Determine best direction(s)
        directions = self._select_directions(task)

        # 2. Extract knowledge from input
        extracted = self._extract(task, raw_data, directions)

        # 3. Extract DNA (core patterns)
        dna = self._extract_dna(extracted)

        return OrganResult(
            success=True,
            output={"extracted": extracted, "dna": dna, "directions": directions},
            metadata={"directions_used": directions},
        )

    def _select_directions(self, task: str) -> list[str]:
        """Select top 2-3 directions based on task content."""
        scores = {}
        task_lower = task.lower()

        direction_keywords = {
            "paper_search": ["paper", "arxiv", "research", "study", "论文"],
            "github_search": ["github", "repo", "code", "project", "仓库"],
            "web_search": ["search", "find", "look up", "搜索", "查找"],
            "code_analysis": ["code", "function", "class", "implement", "代码", "实现"],
            "memory_recall": ["remember", "recall", "previous", "history", "记忆", "历史"],
            "pattern_mining": ["pattern", "trend", "regularity", "模式", "趋势"],
            "hypothesis_gen": ["hypothesis", "predict", "guess", "假设", "预测"],
        }

        for direction, keywords in direction_keywords.items():
            score = sum(1.0 for kw in keywords if kw in task_lower)
            score += self._direction_weights.get(direction, 0.1)
            scores[direction] = score

        sorted_dirs = sorted(scores, key=scores.get, reverse=True)
        return sorted_dirs[:3]

    def _extract(self, task: str, raw_data: dict, directions: list[str]) -> list[dict]:
        """Extract structured knowledge from raw input."""
        results = []

        # Extract from raw data
        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, str) and len(value) > 10:
                    node_type = self._classify_content(value)
                    results.append(
                        {
                            "source": key,
                            "content": value,
                            "type": node_type,
                            "direction": directions[0] if directions else "memory_recall",
                        }
                    )
                elif isinstance(value, (list, dict)):
                    results.append(
                        {
                            "source": key,
                            "content": json.dumps(value, ensure_ascii=False)[:500],
                            "type": "structured_data",
                            "direction": directions[0] if directions else "memory_recall",
                        }
                    )

        # LLM-enhanced extraction for complex tasks
        if len(task) > 20 and self._llm:
            try:
                prompt = f"""Extract key knowledge from this task. Return JSON array of objects with fields: content, type(fact/insight/pattern/hypothesis), confidence(0-1).
Task: {task}
Data: {json.dumps(raw_data, ensure_ascii=False)[:1000]}"""
                response = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1000)
                # Parse LLM response
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if json_match:
                    llm_items = json.loads(json_match.group())
                    results.extend(llm_items)
            except Exception as e:
                logger.warning(f"LLM extraction error: {e}")

        return results

    def _extract_dna(self, extracted: list[dict]) -> dict[str, Any]:
        """Extract DNA patterns - core reusable knowledge structures."""
        dna = {"patterns": [], "keywords": set(), "concepts": []}

        for item in extracted:
            content = item.get("content", "")
            # Extract keywords
            words = re.findall(r"\b[a-zA-Z_]{3,}\b", content)
            dna["keywords"].update(words[:10])

            # Classify pattern
            item_type = item.get("type", "unknown")
            if item_type in ("pattern", "insight"):
                dna["patterns"].append({"content": content[:200], "type": item_type})
            elif item_type in ("fact", "hypothesis"):
                dna["concepts"].append({"content": content[:200], "type": item_type})

        dna["keywords"] = list(dna["keywords"])[:50]
        return dna

    def _classify_content(self, content: str) -> str:
        """Classify content type based on heuristics."""
        content_lower = content.lower()
        if any(w in content_lower for w in ["hypothesis", "predict", "might", "could"]):
            return "hypothesis"
        if any(w in content_lower for w in ["pattern", "always", "never", "tends to"]):
            return "pattern"
        if any(w in content_lower for w in ["insight", "realize", "discover", "发现"]):
            return "insight"
        return "fact"

    def update_direction_weights(self, direction: str, reward: float) -> None:
        """Update direction weights based on reward signal."""
        if direction in self._direction_weights:
            self._direction_weights[direction] = max(0.01, self._direction_weights[direction] + 0.1 * reward)
            total = sum(self._direction_weights.values())
            for d in self._direction_weights:
                self._direction_weights[d] /= total
