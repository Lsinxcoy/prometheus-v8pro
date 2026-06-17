"""Three-Stage Fitness Evaluation - Static + Dynamic + LLM-as-Judge."""

from __future__ import annotations

import ast
import json
import logging
import re

from prometheus_v8.schema import FitnessResult, Genome

logger = logging.getLogger(__name__)


class ThreeStageFitness:
    """3-stage fitness: static(0.2) + dynamic(0.4) + LLM-as-Judge(0.4)."""

    def __init__(self, llm=None, weights: tuple[float, float, float] = (0.2, 0.4, 0.4)) -> None:
        self._llm = llm
        self._weights = weights
        self._history: list[FitnessResult] = []

    def evaluate(self, genome: Genome) -> FitnessResult:
        """Evaluate genome fitness through 3 stages."""
        static = self._static_analysis(genome)
        dynamic = self._dynamic_validation(genome)
        llm_score = self._llm_judge(genome)

        composite = self._weights[0] * static + self._weights[1] * dynamic + self._weights[2] * llm_score

        result = FitnessResult(
            composite=composite,
            static_score=static,
            dynamic_score=dynamic,
            llm_score=llm_score,
            can_promote=composite >= 0.7 and static >= 0.5 and dynamic >= 0.5,
            details={"genome_fingerprint": genome.fingerprint, "code_length": len(genome.code)},
        )
        self._history.append(result)
        return result

    def _static_analysis(self, genome: Genome) -> float:
        """Stage 1: Static code analysis (syntax, complexity, style)."""
        if not genome.code:
            return 0.1

        score = 0.0
        # Syntax check
        try:
            ast.parse(genome.code)
            score += 0.3
        except SyntaxError:
            return 0.0

        # Complexity (prefer moderate complexity)
        lines = genome.code.count("\n") + 1
        if 10 <= lines <= 200:
            score += 0.2
        elif lines < 10:
            score += 0.1

        # Has docstrings
        if '"""' in genome.code or "'''" in genome.code:
            score += 0.1

        # Has type hints
        if "-> " in genome.code or ": " in genome.code:
            score += 0.1

        # Has error handling
        if "try:" in genome.code or "except" in genome.code:
            score += 0.1

        # Has tests
        if "assert " in genome.code or "def test_" in genome.code:
            score += 0.2

        return min(1.0, score)

    def _dynamic_validation(self, genome: Genome) -> float:
        """Stage 2: Dynamic validation (execution, tests, performance).

        Goes beyond compile() to actually execute code in a sandboxed subprocess
        when possible. Falls back to compile + coverage heuristics.
        """
        if not genome.code:
            return 0.0

        score = 0.0

        # Can it be compiled?
        try:
            compile(genome.code, "<evolution>", "exec")
            score += 0.2
        except Exception as e:
            logger.debug(f"Genome compile check failed: {e}")
            return 0.0

        # Try to actually execute the code in a subprocess sandbox
        execution_score = self._sandbox_execute(genome.code)
        score += execution_score * 0.3  # Up to 0.3 for successful execution

        # Skill coverage
        if genome.skills:
            score += min(0.2, len(genome.skills) * 0.05)

        # Tool coverage
        if genome.tools:
            score += min(0.15, len(genome.tools) * 0.05)

        # Config quality
        if genome.config:
            score += min(0.15, len(genome.config) * 0.05)

        return min(1.0, score)

    def _sandbox_execute(self, code: str) -> float:
        """Execute code in a subprocess sandbox and return success score."""
        import subprocess
        import tempfile

        # Only try execution for code that looks safe (no imports of dangerous modules)
        dangerous_patterns = ["os.system", "subprocess", "shutil.rmtree", "eval(", "exec(", "__import__"]
        for pattern in dangerous_patterns:
            if pattern in code:
                # Skip execution for potentially dangerous code
                return 0.3  # Partial credit for compilable but unsafe code

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                f.write("\n")  # Ensure newline at end
                tmp_path = f.name

            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                timeout=5,  # 5-second timeout
                text=True,
            )

            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            if result.returncode == 0:
                return 1.0  # Clean execution
            elif result.returncode == 1:
                # Runtime error — still partially valid (compiles and starts running)
                return 0.5
            else:
                return 0.2  # Some other error

        except subprocess.TimeoutExpired:
            return 0.1  # Timeout — code runs but too slowly
        except Exception as e:
            logger.debug(f"Sandbox execution failed: {e}")
            return 0.0

    def _llm_judge(self, genome: Genome) -> float:
        """Stage 3: LLM-as-Judge evaluation with rule-based fallback."""
        if not genome.code:
            return 0.2  # No code = low score (not neutral 0.5)

        if self._llm:
            try:
                prompt = f"""Evaluate this code/solution on a 0.0-1.0 scale considering:
1. Correctness: Does it solve the intended problem?
2. Quality: Is it well-structured and maintainable?
3. Safety: Are there any dangerous patterns?
4. Efficiency: Is it reasonably efficient?

Code (first 800 chars):
{genome.code[:800]}

Skills: {genome.skills}
Config: {json.dumps(genome.config)[:200]}

Return ONLY a number between 0.0 and 1.0."""
                response = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=10)
                match = re.search(r"[\d.]+", response)
                if match:
                    return max(0.0, min(1.0, float(match.group())))
            except Exception as e:
                logger.warning(f"LLM judge error: {e}")

        # Rule-based fallback when no LLM available
        # Instead of returning neutral 0.5, actually evaluate the code
        fallback_score = 0.3  # Base score

        # Check for function definitions (structure)
        if "def " in genome.code:
            fallback_score += 0.1
        # Check for return statements (completeness)
        if "return " in genome.code:
            fallback_score += 0.1
        # Check for error handling (robustness)
        if "try:" in genome.code or "except" in genome.code:
            fallback_score += 0.1
        # Check for type hints (quality)
        if "-> " in genome.code:
            fallback_score += 0.05
        # Penalize very short code (likely incomplete)
        if len(genome.code.strip()) < 50:
            fallback_score -= 0.1
        # Bonus for having skills/tools (implies capability)
        if genome.skills:
            fallback_score += min(0.1, len(genome.skills) * 0.03)
        if genome.tools:
            fallback_score += min(0.05, len(genome.tools) * 0.02)

        return max(0.0, min(1.0, fallback_score))
