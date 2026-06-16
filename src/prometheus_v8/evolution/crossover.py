"""AST-Level Crossover with Indentation Fix."""

from __future__ import annotations

import ast
import logging
import random
import textwrap

from prometheus_v8.schema import Genome

logger = logging.getLogger(__name__)


class ASTCrossover:
    """AST-level crossover between two code genomes."""

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Perform AST-level crossover between two code genomes."""
        if not parent1.code or not parent2.code:
            return Genome(code=parent1.code or parent2.code)

        try:
            tree1 = ast.parse(parent1.code)
            tree2 = ast.parse(parent2.code)
        except SyntaxError:
            # Fallback to text-level crossover
            return self._text_crossover(parent1, parent2)

        # Get top-level statements
        stmts1 = tree1.body
        stmts2 = tree2.body

        if not stmts1 or not stmts2:
            return Genome(code=parent1.code)

        # Single-point crossover
        point1 = random.randint(1, len(stmts1))
        point2 = random.randint(0, len(stmts2) - 1)

        child_stmts = stmts1[:point1] + stmts2[point2:]
        tree1.body = child_stmts

        try:
            child_code = ast.unparse(tree1)
            child_code = textwrap.dedent(child_code)
        except Exception as e:
            logger.debug(f"AST crossover unparsing failed, using parent code: {e}")
            child_code = parent1.code

        child = Genome(
            code=child_code,
            fitness=(parent1.fitness + parent2.fitness) / 2,
            lineage=[parent1.fingerprint, parent2.fingerprint],
            config={**parent1.config, **parent2.config},
            skills=list(set(parent1.skills + parent2.skills)),
            prompts=parent1.prompts[:],
            tools=list(set(parent1.tools + parent2.tools)),
        )
        return child

    def _text_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Fallback text-level crossover."""
        lines1 = parent1.code.split("\n")
        lines2 = parent2.code.split("\n")

        if not lines1 or not lines2:
            return Genome(code=parent1.code)

        point = random.randint(1, max(1, len(lines1)))
        child_lines = lines1[:point]
        if lines2:
            point2 = random.randint(0, max(0, len(lines2) - 1))
            child_lines.extend(lines2[point2:])

        return Genome(
            code="\n".join(child_lines),
            fitness=(parent1.fitness + parent2.fitness) / 2,
            lineage=[parent1.fingerprint, parent2.fingerprint],
        )
