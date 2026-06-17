"""Darwin Organ - Mutate with AST mutations + GA operators."""

from __future__ import annotations

import logging
import random

from prometheus_v8.core.ast_mutator import ASTMutator
from prometheus_v8.organs.base import BaseOrgan, LLMClient, OrganContext, OrganEnv, OrganResult

logger = logging.getLogger(__name__)


class DarwinOrgan(BaseOrgan):
    """Mutate organ: apply genetic operators to generate variants."""

    def __init__(self, llm: LLMClient | None = None, env: OrganEnv | None = None) -> None:
        super().__init__("darwin", llm, env)
        self._ast_mutator = ASTMutator()

    def execute(self, context: OrganContext) -> OrganResult:
        inputs = context.inputs
        generations = inputs.get("generations", [])

        variants = []
        for gen in generations:
            # Apply mutations
            for _ in range(3):  # 3 variants per generation
                variant = self._mutate_generation(gen, context.task)
                if variant:
                    variants.append(variant)

        # Crossover between top variants
        if len(variants) >= 2:
            crossed = self._crossover(variants[0], variants[1])
            if crossed:
                variants.append(crossed)

        return OrganResult(
            success=True,
            output={"variants": variants, "parent_count": len(generations)},
            metadata={"variant_count": len(variants)},
        )

    def _mutate_generation(self, generation: dict, task: str) -> dict | None:
        """Apply mutation to a single generation."""
        content = generation.get("content", "")
        gen_type = generation.get("type", "")

        if gen_type == "code_patch" and content:
            mutated, mtype = self._ast_mutator.mutate(content)
            return {**generation, "content": mutated, "mutation": mtype, "parent": generation.get("type", "")}

        # For non-code, use LLM or simple text mutation
        if self._llm and len(content) > 50:
            try:
                prompt = f"""Slightly modify this solution while keeping it valid. Change approach or details but not the goal.
Original: {content[:500]}
Task: {task}
Return the modified solution only."""
                response = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=500)
                return {
                    **generation,
                    "content": response,
                    "mutation": "llm_rewrite",
                    "parent": generation.get("type", ""),
                }
            except Exception as e:
                logger.debug(f"LLM rewrite mutation failed: {e}")
                pass

        # Simple text mutation
        words = content.split()
        if len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            words[idx] = words[idx][::-1] if len(words[idx]) > 2 else words[idx]
            return {
                **generation,
                "content": " ".join(words),
                "mutation": "word_shuffle",
                "parent": generation.get("type", ""),
            }

        return None

    def _crossover(self, parent1: dict, parent2: dict) -> dict | None:
        """Crossover two variants."""
        c1 = parent1.get("content", "")
        c2 = parent2.get("content", "")
        if not c1 or not c2:
            return None

        # Simple midpoint crossover
        mid1 = len(c1) // 2
        mid2 = len(c2) // 2
        child_content = c1[:mid1] + c2[mid2:]

        return {
            "type": "crossover",
            "content": child_content,
            "confidence": (parent1.get("confidence", 0.5) + parent2.get("confidence", 0.5)) / 2,
            "mutation": "crossover",
            "parents": [parent1.get("type", ""), parent2.get("type", "")],
        }
