"""Direction Selector - Adaptive 3-direction with UCB1 bandit."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DIRECTIONS = ["forward", "lateral", "reverse"]

@dataclass
class DirectionStats:
    direction: str = ""
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.pulls)


class DirectionSelector:
    """UCB1-based direction selector for evolution.

    Forward: improve existing solutions (exploitation)
    Lateral: explore alternative approaches (exploration)
    Reverse: backtrack and try opposite direction (escape local optima)
    """

    def __init__(self, exploration_constant: float = 1.414) -> None:
        self._C = exploration_constant
        self._stats = {d: DirectionStats(direction=d) for d in DIRECTIONS}
        self._total_pulls = 0
        self._history: list[tuple[str, float]] = []

    def select(self, context: dict | None = None) -> str:
        """Select direction using UCB1."""
        self._total_pulls += 1

        # Ensure minimum exploration
        for d, s in self._stats.items():
            if s.pulls < 2:
                return d

        # UCB1 formula
        scores = {}
        for d, s in self._stats.items():
            exploration = math.sqrt(2 * math.log(self._total_pulls) / s.pulls)
            scores[d] = s.avg_reward + self._C * exploration

        # Context bonus
        if context:
            stagnation = context.get("stagnation_count", 0)
            if stagnation > 5:
                scores["reverse"] += 0.5
            elif stagnation > 2:
                scores["lateral"] += 0.3

        best = max(scores, key=scores.get)
        return best

    def update(self, direction: str, reward: float) -> None:
        """Update direction statistics with reward."""
        if direction in self._stats:
            self._stats[direction].pulls += 1
            self._stats[direction].total_reward += reward
            self._history.append((direction, reward))

    def get_stats(self) -> dict[str, dict]:
        return {d: {"pulls": s.pulls, "avg_reward": s.avg_reward} for d, s in self._stats.items()}

    def reset(self) -> None:
        self._stats = {d: DirectionStats(direction=d) for d in DIRECTIONS}
        self._total_pulls = 0
        self._history.clear()
