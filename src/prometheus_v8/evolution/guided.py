"""Goal-Guided Evolution with Objectives and Constraints."""

from __future__ import annotations

import logging
from typing import Any

from prometheus_v8.evolution.goal_system import Goal, GoalSystem
from prometheus_v8.schema import Genome

logger = logging.getLogger(__name__)


class GuidedEvolution:
    """Goal-guided evolution: evolve toward specific objectives with constraints.

    Uses GoalSystem to track active goals and guide evolution direction.
    """

    def __init__(self, goal_system: GoalSystem | None = None) -> None:
        self._goals = goal_system or GoalSystem()

    def evolve_with_goal(self, genome: Genome, max_steps: int = 50) -> tuple[Genome, dict[str, Any]]:
        """Evolve genome toward the current active goal."""
        goal = self._goals.get_active_goal()
        if not goal:
            goal = self._goals.get_next_pending()
            if goal:
                self._goals.activate_goal(goal.id)

        result_info = {"goal": goal.name if goal else "none", "steps": 0, "improvement": 0.0}

        if not goal:
            return genome, result_info

        initial_fitness = genome.fitness

        for step in range(max_steps):
            # Check if goal is achieved
            if genome.fitness >= goal.fitness_target:
                self._goals.complete_goal(goal.id, genome.fitness)
                result_info["improvement"] = genome.fitness - initial_fitness
                break

            # Apply goal-directed modifications
            genome = self._apply_goal_mutation(genome, goal)
            result_info["steps"] = step + 1

            # Update progress
            progress = genome.fitness / max(0.01, goal.fitness_target)
            self._goals.update_progress(goal.id, progress)

        result_info["improvement"] = genome.fitness - initial_fitness
        return genome, result_info

    def _apply_goal_mutation(self, genome: Genome, goal: Goal) -> Genome:
        """Apply mutations directed toward the goal."""
        import random

        from prometheus_v8.organs.darwin import ASTMutator

        if genome.code:
            mutator = ASTMutator()
            mutated_code, mtype = mutator.mutate(genome.code)
            if mutated_code != genome.code:
                genome.code = mutated_code
                # Small fitness boost toward goal
                genome.fitness += random.uniform(-0.05, 0.1)
                genome.fitness = max(0.0, min(1.0, genome.fitness))

        return genome

    def create_goal(self, name: str, fitness_target: float = 0.8, constraints: list[str] | None = None) -> Goal:
        return self._goals.create_goal(name=name, fitness_target=fitness_target, constraints=constraints)
