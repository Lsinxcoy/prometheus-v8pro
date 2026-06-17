"""Organ-Evolution Bridge - Connects L2 organ pipeline with L3 evolution engine.

Implementation lives here; core.bridge re-exports for convenience.

Provides:
- Pipeline-to-genome conversion (promoted variants → Genome list)
- Genome-to-organ-input conversion (evolved genome → pipeline input)
- Full bridge cycle (pipeline → evolution → refined genomes)
- Organ feedback integration (register feedback callbacks with EvolutionLoop)
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_v8.schema import Genome

logger = logging.getLogger(__name__)


class OrganEvolutionBridge:
    """Bridge between organ pipeline (L2) and evolution engine (L3).

    Converts organ pipeline outputs (promoted variants) into genomes
    for the evolution engine, and feeds evolved genomes back as
    organ inputs for further refinement.

    Also connects organ feedback to the EvolutionLoop's OrganFeedbackCollector,
    closing the loop: organs → feedback → evolution → organs.
    """

    def __init__(self, store=None, engine=None, organ_feedback=None) -> None:
        self._store = store
        self._engine = engine
        self._organ_feedback = organ_feedback  # OrganFeedbackCollector from EvolutionLoop
        self._bridged_count = 0
        self._last_organ_results: dict[str, dict[str, Any]] = {}

    def register_organ_feedback(self, organ_name: str, organ_instance: Any) -> None:
        """Register an organ with the feedback collector for evolution loop integration.

        Args:
            organ_name: Name of the organ (e.g., "taotie", "nuwa", "darwin").
            organ_instance: The organ instance. Must have a `stats` property or
                           a method that returns feedback data.
        """
        if self._organ_feedback is None:
            logger.debug(f"No OrganFeedbackCollector available; skipping registration of {organ_name}")
            return

        def _make_feedback_fn(name: str, organ: Any) -> callable:
            """Create a feedback callback that extracts relevant signals from an organ."""
            def feedback_fn() -> dict[str, Any]:
                data: dict[str, Any] = {"organ": name}

                # Extract stats if available
                if hasattr(organ, "stats"):
                    try:
                        data["stats"] = organ.stats
                    except Exception:
                        pass

                # Extract last result if available
                if name in self._last_organ_results:
                    data["last_result"] = self._last_organ_results[name]

                # Extract fitness signal from organ
                if hasattr(organ, "_last_fitness"):
                    try:
                        data["fitness_signal"] = organ._last_fitness
                    except Exception:
                        pass

                # Extract execution count for activity tracking
                if hasattr(organ, "_execution_count"):
                    try:
                        data["execution_count"] = organ._execution_count
                    except Exception:
                        pass

                return data
            return feedback_fn

        self._organ_feedback.register(organ_name, _make_feedback_fn(organ_name, organ_instance))
        logger.info(f"Registered organ '{organ_name}' for evolution feedback")

    def record_organ_result(self, organ_name: str, result: dict[str, Any]) -> None:
        """Record an organ execution result for feedback.

        This should be called after each organ execution so the bridge
        can provide up-to-date feedback to the evolution loop.

        Args:
            organ_name: Name of the organ that produced the result.
            result: The organ execution result dict.
        """
        self._last_organ_results[organ_name] = {
            "success": result.get("success", False),
            "fitness": result.get("fitness", result.get("score", 0.0)),
            "timestamp": result.get("timestamp", 0.0),
        }

    def pipeline_to_genomes(self, pipeline_result: dict) -> list[Genome]:
        """Convert LifePipeline output to Genome list for evolution.

        Args:
            pipeline_result: Output from LifePipeline.run()
                - promoted: list of promoted variants
                - rejected: list of rejected variants

        Returns:
            List of Genome objects ready for evolution.
        """
        genomes = []
        promoted = pipeline_result.get("promoted", [])

        for variant in promoted:
            if isinstance(variant, dict):
                code = variant.get("code", variant.get("output", ""))
                fitness = variant.get("score", variant.get("fitness", 0.3))
            elif isinstance(variant, str):
                code = variant
                fitness = 0.3
            else:
                continue

            if code:
                genome = Genome(code=str(code), fitness=float(fitness))
                genomes.append(genome)

        if not genomes and pipeline_result.get("success", False):
            # Fallback: create genome from pipeline output
            output = pipeline_result.get("output", "")
            if output:
                genomes.append(Genome(code=str(output), fitness=0.3))

        self._bridged_count += len(genomes)
        return genomes

    def genome_to_organ_input(self, genome: Genome) -> dict:
        """Convert an evolved Genome back into organ pipeline input format.

        Args:
            genome: An evolved genome from the evolution engine.

        Returns:
            Dict suitable as input for LifePipeline.run().
        """
        return {
            "task": "refine_evolved_code",
            "code": genome.code,
            "fitness": genome.fitness,
            "skills": list(genome.skills) if genome.skills else [],
        }

    def evolve_pipeline_output(self, pipeline_result: dict, max_generations: int = 5) -> list[Genome]:
        """Full bridge: pipeline output → evolution → refined genomes.

        Args:
            pipeline_result: Output from LifePipeline.run()
            max_generations: Max evolution generations.

        Returns:
            List of evolved Genome objects.
        """
        genomes = self.pipeline_to_genomes(pipeline_result)

        if not genomes or not self._engine:
            return genomes

        evolved = []
        for genome in genomes:
            try:
                result = self._engine.evolve(genome, max_generations=max_generations)
                if result and result.fitness > genome.fitness:
                    evolved.append(result)
                else:
                    evolved.append(genome)  # Keep original if no improvement
            except Exception as e:
                logger.warning(f"Evolution failed for genome: {e}")
                evolved.append(genome)

        return evolved

    @property
    def stats(self) -> dict[str, Any]:
        stats = {"bridged": self._bridged_count}
        if self._organ_feedback:
            stats["registered_organs"] = self._organ_feedback.registered_organs
        stats["organs_with_results"] = list(self._last_organ_results.keys())
        return stats
