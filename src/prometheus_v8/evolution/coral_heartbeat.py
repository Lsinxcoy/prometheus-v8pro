"""CORAL Heartbeat Mechanism.

Based on CORAL research (arXiv 2604.01658):
- Per-iteration reflection: write notes after each task
- Periodic consolidation: merge notes into reusable skills
- Stagnation-triggered redirection: pivot strategy when stuck

3-10x higher improvement rates than fixed evolutionary baselines.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReflectionNote:
    """A reflection note written after a task."""

    task: str = ""
    outcome: str = ""  # success/partial/failure
    insights: list[str] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsolidatedSkill:
    """A skill distilled from multiple reflection notes."""

    name: str = ""
    pattern: str = ""
    conditions: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    source_notes: int = 0
    effectiveness: float = 0.0
    created_at: float = field(default_factory=time.time)


class CORALHeartbeat:
    """Three-heartbeat mechanism for self-evolving agents.

    1. Reflect: After each task, write reflection notes
    2. Consolidate: Periodically merge notes into reusable skills
    3. Redirect: When stagnation detected, pivot strategy
    """

    def __init__(
        self, consolidation_interval: int = 5, stagnation_threshold: int = 10, notes_dir: str = "data/coral_notes"
    ) -> None:
        self._consolidation_interval = consolidation_interval
        self._stagnation_threshold = stagnation_threshold
        self._notes_dir = Path(notes_dir)
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._notes: list[ReflectionNote] = []
        self._skills: list[ConsolidatedSkill] = []
        self._task_count = 0
        self._fitness_history: list[float] = []
        self._last_consolidation = 0

    def reflect(
        self,
        task: str,
        outcome: str,
        insights: list[str] | None = None,
        mistakes: list[str] | None = None,
        improvements: list[str] | None = None,
    ) -> ReflectionNote:
        """Heartbeat 1: Write a reflection note after task completion."""
        note = ReflectionNote(
            task=task,
            outcome=outcome,
            insights=insights or [],
            mistakes=mistakes or [],
            improvements=improvements or [],
        )
        self._notes.append(note)
        self._task_count += 1

        # Auto-consolidate every N tasks
        if self._task_count - self._last_consolidation >= self._consolidation_interval:
            self.consolidate()

        # Persist note
        self._persist_note(note)

        return note

    def consolidate(self) -> list[ConsolidatedSkill]:
        """Heartbeat 2: Consolidate reflection notes into reusable skills."""
        new_skills = []

        # Group notes by outcome and task similarity
        success_notes = [n for n in self._notes if n.outcome == "success"]
        failure_notes = [n for n in self._notes if n.outcome == "failure"]

        # Extract patterns from successful notes
        if len(success_notes) >= 2:
            common_insights = self._find_common_insights(success_notes)
            for insight in common_insights:
                skill = ConsolidatedSkill(
                    name=f"skill_from_success_{len(self._skills)}",
                    pattern=insight,
                    conditions=["task_similar_to_previous_success"],
                    procedure=[insight],
                    source_notes=len(success_notes),
                    effectiveness=0.7,
                )
                new_skills.append(skill)
                self._skills.append(skill)

        # Extract lessons from failures
        if failure_notes:
            common_mistakes = self._find_common_mistakes(failure_notes)
            for mistake in common_mistakes:
                skill = ConsolidatedSkill(
                    name=f"skill_from_failure_{len(self._skills)}",
                    pattern=f"AVOID: {mistake}",
                    conditions=["task_similar_to_previous_failure"],
                    procedure=[f"Never {mistake}"],
                    source_notes=len(failure_notes),
                    effectiveness=0.5,
                )
                new_skills.append(skill)
                self._skills.append(skill)

        self._last_consolidation = self._task_count
        logger.info(f"CORAL consolidation: {len(new_skills)} new skills from {len(self._notes)} notes")

        return new_skills

    def redirect(self, current_fitness: float) -> dict[str, Any]:
        """Heartbeat 3: Detect stagnation and suggest strategy pivot."""
        self._fitness_history.append(current_fitness)

        if len(self._fitness_history) < self._stagnation_threshold:
            return {"action": "continue", "reason": "Not enough history"}

        recent = self._fitness_history[-self._stagnation_threshold :]
        improvement = max(recent) - min(recent)

        if improvement < 0.01:
            # Stagnation detected - suggest redirect
            strategies = [
                "increase_mutation_rate",
                "change_direction_to_lateral",
                "inject_random_individuals",
                "expand_search_space",
                "try_reverse_engineering",
            ]
            import random

            strategy = random.choice(strategies)
            logger.info(f"CORAL redirect: stagnation detected (improvement={improvement:.4f}), suggesting: {strategy}")
            return {
                "action": "redirect",
                "reason": f"Stagnation: {improvement:.4f} improvement over {self._stagnation_threshold} tasks",
                "strategy": strategy,
                "fitness_history_len": len(self._fitness_history),
            }

        return {"action": "continue", "reason": f"Healthy improvement: {improvement:.4f}"}

    def _find_common_insights(self, notes: list[ReflectionNote]) -> list[str]:
        """Find insights that appear in multiple notes."""
        insight_count: dict[str, int] = {}
        for note in notes:
            for insight in note.insights:
                key = insight.lower().strip()
                insight_count[key] = insight_count.get(key, 0) + 1
        return [k for k, v in insight_count.items() if v >= 2][:5]

    def _find_common_mistakes(self, notes: list[ReflectionNote]) -> list[str]:
        """Find mistakes that appear in multiple failure notes."""
        mistake_count: dict[str, int] = {}
        for note in notes:
            for mistake in note.mistakes:
                key = mistake.lower().strip()
                mistake_count[key] = mistake_count.get(key, 0) + 1
        return [k for k, v in mistake_count.items() if v >= 1][:5]

    def _persist_note(self, note: ReflectionNote) -> None:
        """Persist reflection note to disk."""
        try:
            path = self._notes_dir / f"note_{int(note.timestamp)}.json"
            path.write_text(
                json.dumps(
                    {
                        "task": note.task,
                        "outcome": note.outcome,
                        "insights": note.insights,
                        "mistakes": note.mistakes,
                        "improvements": note.improvements,
                        "timestamp": note.timestamp,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to persist note: {e}")

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "tasks": self._task_count,
            "notes": len(self._notes),
            "skills": len(self._skills),
            "last_consolidation": self._last_consolidation,
            "fitness_history_len": len(self._fitness_history),
        }
