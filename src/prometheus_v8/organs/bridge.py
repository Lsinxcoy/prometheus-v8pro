"""Organ-Evolution Bridge - Connects L2 organ pipeline with L3 evolution engine.

Implementation lives here; core.bridge re-exports for convenience.
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
    """

    def __init__(self, store=None, engine=None) -> None:
        self._store = store
        self._engine = engine
        self._bridged_count = 0

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
        return {"bridged": self._bridged_count}
