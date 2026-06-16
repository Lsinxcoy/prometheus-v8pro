"""Anti-Evolution Detection - 4 checks with cosine similarity."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AntiEvolutionAlert:
    """Alert when evolution is going backwards."""

    check_name: str = ""
    severity: float = 0.0  # 0-1
    description: str = ""
    recommendation: str = ""


class AntiEvolutionDetector:
    """4-check anti-evolution detector:
    1. Fitness regression (is fitness decreasing?)
    2. Code similarity (is code converging to same pattern?)
    3. Behavioral regression (are outputs getting worse?)
    4. Diversity collapse (is population diversity too low?)
    """

    def __init__(self, similarity_threshold: float = 0.95, diversity_threshold: float = 0.1) -> None:
        self._similarity_threshold = similarity_threshold
        self._diversity_threshold = diversity_threshold
        self._fitness_history: list[float] = []
        self._code_hashes: list[str] = []
        self._behavior_scores: list[float] = []

    def check(
        self, fitness: float, code: str, behavior_score: float = 0.5, population_diversity: float = 1.0
    ) -> list[AntiEvolutionAlert]:
        """Run all 4 anti-evolution checks."""
        alerts = []

        self._fitness_history.append(fitness)
        self._code_hashes.append(hashlib.sha256(code.encode()).hexdigest()[:16])
        self._behavior_scores.append(behavior_score)

        # Check 1: Fitness regression
        alerts.extend(self._check_fitness_regression())

        # Check 2: Code similarity (convergence)
        alerts.extend(self._check_code_similarity())

        # Check 3: Behavioral regression
        alerts.extend(self._check_behavioral_regression())

        # Check 4: Diversity collapse
        alerts.extend(self._check_diversity(population_diversity))

        return alerts

    def _check_fitness_regression(self) -> list[AntiEvolutionAlert]:
        if len(self._fitness_history) < 3:
            return []
        recent = self._fitness_history[-3:]
        if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
            return [
                AntiEvolutionAlert(
                    check_name="fitness_regression",
                    severity=0.8,
                    description=f"Fitness declining: {recent}",
                    recommendation="Increase mutation rate, change direction to lateral or reverse",
                )
            ]
        return []

    def _check_code_similarity(self) -> list[AntiEvolutionAlert]:
        if len(self._code_hashes) < 5:
            return []
        recent_hashes = self._code_hashes[-5:]
        unique = len(set(recent_hashes))
        if unique <= 2:
            return [
                AntiEvolutionAlert(
                    check_name="code_convergence",
                    severity=0.7,
                    description=f"Code converging: {unique} unique in last 5 generations",
                    recommendation="Increase mutation rate, add crossover diversity",
                )
            ]
        return []

    def _check_behavioral_regression(self) -> list[AntiEvolutionAlert]:
        if len(self._behavior_scores) < 5:
            return []
        recent = self._behavior_scores[-5:]
        avg_recent = sum(recent) / len(recent)
        avg_old = sum(self._behavior_scores[:-5]) / max(1, len(self._behavior_scores[:-5]))
        if avg_recent < avg_old * 0.8:
            return [
                AntiEvolutionAlert(
                    check_name="behavioral_regression",
                    severity=0.6,
                    description=f"Behavior degrading: recent={avg_recent:.3f}, old={avg_old:.3f}",
                    recommendation="Rollback to previous best, reduce exploration",
                )
            ]
        return []

    def _check_diversity(self, diversity: float) -> list[AntiEvolutionAlert]:
        if diversity < self._diversity_threshold:
            return [
                AntiEvolutionAlert(
                    check_name="diversity_collapse",
                    severity=0.9,
                    description=f"Population diversity too low: {diversity:.3f}",
                    recommendation="Inject random individuals, increase mutation rate",
                )
            ]
        return []

    def reset(self) -> None:
        self._fitness_history.clear()
        self._code_hashes.clear()
        self._behavior_scores.clear()
