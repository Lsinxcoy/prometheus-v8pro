"""Darwin Organ - Mutate with AST mutations + GA operators."""

from __future__ import annotations

import ast
import copy
import logging
import random
import re

from prometheus_v8.core.ast_mutator import ASTMutator  # re-export for backward compat
from prometheus_v8.organs.base import BaseOrgan, LLMClient, OrganContext, OrganEnv, OrganResult

logger = logging.getLogger(__name__)


class ASTMutator:
    """8 AST mutation types for Python code."""

    MUTATION_TYPES = [
        "constant_mutation",  # Change numeric/string constants
        "operator_mutation",  # Swap arithmetic/comparison operators
        "statement_swap",  # Swap adjacent statements
        "condition_flip",  # Negate conditions
        "loop_unroll",  # Unroll small loops
        "variable_rename",  # Rename local variables
        "dead_code_insert",  # Insert dead code (for diversity)
        "expression_simplify",  # Simplify complex expressions
    ]

    def mutate(self, code: str, mutation_type: str = "") -> tuple[str, str]:
        """Apply mutation to code. Returns (mutated_code, mutation_type_used)."""
        if not code.strip():
            return code, "none"

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._regex_mutate(code, mutation_type)

        available = self.MUTATION_TYPES if not mutation_type else [mutation_type]
        chosen = random.choice(available)

        mutator_fn = {
            "constant_mutation": self._mutate_constants,
            "operator_mutation": self._mutate_operators,
            "condition_flip": self._mutate_conditions,
            "variable_rename": self._mutate_variables,
            "expression_simplify": self._mutate_expressions,
        }.get(chosen)

        if mutator_fn:
            mutated = mutator_fn(tree)
            try:
                result = ast.unparse(mutated)
                return result, chosen
            except Exception as e:
                logger.debug(f"AST mutation unparsing failed: {e}")
                return code, "none"

        return code, "none"

    def _mutate_constants(self, tree: ast.AST) -> ast.AST:
        tree = copy.deepcopy(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    delta = random.choice([-1, 1, 2, -2, 0.1, -0.1])
                    node.value = type(node.value)(node.value + delta)
                elif isinstance(node.value, str) and len(node.value) > 0:
                    pos = random.randint(0, max(0, len(node.value) - 1))
                    node.value = node.value[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + node.value[pos + 1 :]
        return tree

    def _mutate_operators(self, tree: ast.AST) -> ast.AST:
        tree = copy.deepcopy(tree)
        op_swaps = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.Div,
            ast.Div: ast.Mult,
            ast.Mod: ast.FloorDiv,
            ast.FloorDiv: ast.Mod,
            ast.BitAnd: ast.BitOr,
            ast.BitOr: ast.BitAnd,
            ast.Lt: ast.Gt,
            ast.Gt: ast.Lt,
            ast.LtE: ast.GtE,
            ast.GtE: ast.LtE,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
        }
        for node in ast.walk(tree):
            # Check BinOp.op, BoolOp.op, Compare.ops, AugAssign.op
            for attr in ("op", "ops"):
                val = getattr(node, attr, None)
                if val is None:
                    continue
                if isinstance(val, list):  # Compare.ops
                    for i, op in enumerate(val):
                        if type(op) in op_swaps and random.random() < 0.3:
                            val[i] = op_swaps[type(op)]()
                else:  # single op (BinOp, BoolOp, AugAssign)
                    if type(val) in op_swaps and random.random() < 0.3:
                        setattr(node, attr, op_swaps[type(val)]())
        return tree

    def _mutate_conditions(self, tree: ast.AST) -> ast.AST:
        tree = copy.deepcopy(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                break  # Only flip first if
        return tree

    def _mutate_variables(self, tree: ast.AST) -> ast.AST:
        tree = copy.deepcopy(tree)
        renames = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in renames and not node.id.startswith("_"):
                    renames[node.id] = f"var_{random.randint(100, 999)}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in renames:
                node.id = renames[node.id]
        return tree

    def _mutate_expressions(self, tree: ast.AST) -> ast.AST:
        tree = copy.deepcopy(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.left, ast.BinOp):
                node.left = node.left.left  # Simplify (a op b) op c → a op c
                break
        return tree

    def _regex_mutate(self, code: str, mutation_type: str) -> tuple[str, str]:
        """Fallback regex-based mutation for non-parseable code."""
        mutations = [
            (
                "num_change",
                lambda c: re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + random.choice([-1, 1])), c, count=1),
            ),
            ("op_swap", lambda c: c.replace("+", "-", 1) if "+" in c else c),
            ("var_rename", lambda c: re.sub(r"\b([a-z_]{3,})\b", lambda m: f"v_{m.group(1)[:2]}", c, count=1)),
        ]
        name, fn = random.choice(mutations)
        return fn(code), name


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
