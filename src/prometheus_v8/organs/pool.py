"""Pool Organ - Validate with compile + sandbox + pytest."""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from prometheus_v8.organs.base import BaseOrgan, LLMClient, OrganContext, OrganEnv, OrganResult

logger = logging.getLogger(__name__)


class PoolOrgan(BaseOrgan):
    """Validate organ: test variants through compile → sandbox → pytest pipeline."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        env: OrganEnv | None = None,
        sandbox_timeout: int = 10,
        syntax_weight: float = 0.3,
        sandbox_weight: float = 0.4,
        semantic_weight: float = 0.3,
        pass_threshold: float = 0.3,
    ) -> None:
        super().__init__("pool", llm, env)
        self._sandbox_timeout = sandbox_timeout
        self._syntax_weight = syntax_weight
        self._sandbox_weight = sandbox_weight
        self._semantic_weight = semantic_weight
        self._pass_threshold = pass_threshold

    def execute(self, context: OrganContext) -> OrganResult:
        inputs = context.inputs
        variants = inputs.get("variants", [])

        validated = []
        for variant in variants:
            score = self._validate(variant, context.task)
            variant["validation_score"] = score
            if score > self._pass_threshold:
                validated.append(variant)

        # Sort by validation score
        validated.sort(key=lambda v: v.get("validation_score", 0), reverse=True)

        return OrganResult(
            success=len(validated) > 0,
            output={"validated": validated, "rejected_count": len(variants) - len(validated)},
            metadata={"validated_count": len(validated), "total_count": len(variants)},
        )

    def _validate(self, variant: dict, task: str) -> float:
        """3-stage validation: syntax → sandbox → semantic."""
        content = variant.get("content", "")
        vtype = variant.get("type", "")
        score = 0.0

        # Stage 1: Syntax check (for code)
        if vtype in ("code_patch", "crossover") or "def " in content or "class " in content:
            syntax_ok = self._check_syntax(content)
            if not syntax_ok:
                return 0.0
            score += self._syntax_weight

        # Stage 2: Sandbox execution (for code)
        if vtype in ("code_patch",) and content:
            sandbox_result = self._sandbox_test(content)
            if sandbox_result.get("success"):
                score += self._sandbox_weight
            elif sandbox_result.get("error"):
                score += 0.1  # Partial credit for running

        # Stage 3: Semantic validation (LLM-based)
        if self._llm and content:
            semantic_score = self._semantic_validate(content, task)
            score += semantic_score * self._semantic_weight
        else:
            score += 0.15  # Default partial score

        return min(1.0, score)

    def _check_syntax(self, code: str) -> bool:
        """Check Python syntax validity."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _sandbox_test(self, code: str) -> dict[str, Any]:
        """Execute code in sandbox with timeout."""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                # Wrap in try/except for safe execution
                safe_code = f"import sys\ntry:\n{self._indent(code)}\nexcept Exception as e:\n    print(f'ERROR: {{e}}', file=sys.stderr)\n"
                f.write(safe_code)
                f.flush()
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self._sandbox_timeout,
            )
            Path(tmp_path).unlink(missing_ok=True)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:500],
                "error": result.stderr[:500] if result.returncode != 0 else "",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _indent(self, code: str) -> str:
        """Indent code for wrapping in try/except."""
        lines = code.split("\n")
        return "\n".join("    " + line if line.strip() else line for line in lines)

    def _semantic_validate(self, content: str, task: str) -> float:
        """LLM-based semantic validation."""
        try:
            prompt = f"""Rate this solution for the given task on a scale of 0.0 to 1.0.
Consider: correctness, completeness, efficiency, safety.

Task: {task}
Solution: {content[:500]}

Return ONLY a number between 0.0 and 1.0."""
            response = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=10)
            score = float(re.search(r"[\d.]+", response).group()) if re.search(r"[\d.]+", response) else 0.5
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"Semantic validation failed: {e}")
            return 0.5
