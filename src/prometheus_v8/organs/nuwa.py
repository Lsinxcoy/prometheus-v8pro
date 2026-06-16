"""Nuwa Organ - Generate with LLM-driven patch generation."""

from __future__ import annotations

import json
import logging
import re

from prometheus_v8.organs.base import BaseOrgan, LLMClient, OrganContext, OrganEnv, OrganResult

logger = logging.getLogger(__name__)


class NuwaOrgan(BaseOrgan):
    """Generate organ: create new knowledge/code/solutions from extracted data."""

    def __init__(self, llm: LLMClient | None = None, env: OrganEnv | None = None) -> None:
        super().__init__("nuwa", llm, env)

    def execute(self, context: OrganContext) -> OrganResult:
        task = context.task
        inputs = context.inputs

        extracted = inputs.get("extracted", [])
        dna = inputs.get("dna", {})
        inputs.get("directions", [])

        # Generate solutions based on extracted knowledge
        generations = []

        # 1. Generate from DNA patterns
        if dna.get("patterns"):
            for pattern in dna["patterns"][:3]:
                gen = self._generate_from_pattern(task, pattern)
                if gen:
                    generations.append(gen)

        # 2. LLM-driven generation
        if self._llm and extracted:
            llm_gen = self._llm_generate(task, extracted, context.constraints)
            if llm_gen:
                generations.append(llm_gen)

        # 3. Code patch generation if task involves code
        if any(kw in task.lower() for kw in ["code", "implement", "fix", "refactor", "代码"]):
            code_gen = self._generate_code_patch(task, inputs)
            if code_gen:
                generations.append(code_gen)

        if not generations:
            generations.append({"type": "placeholder", "content": f"Generated for: {task}", "confidence": 0.3})

        return OrganResult(
            success=True,
            output={"generations": generations, "task": task},
            metadata={"generation_count": len(generations)},
        )

    def _generate_from_pattern(self, task: str, pattern: dict) -> dict | None:
        """Generate solution by applying a known pattern."""
        return {
            "type": "pattern_application",
            "pattern": pattern.get("content", "")[:100],
            "content": f"Applying pattern '{pattern.get('type', 'unknown')}' to: {task}",
            "confidence": 0.6,
        }

    def _llm_generate(self, task: str, extracted: list, constraints: list[str]) -> dict | None:
        """Use LLM to generate solution."""
        try:
            knowledge = "\n".join(f"- {e.get('content', '')[:200]}" for e in extracted[:5])
            constraint_str = "\n".join(f"- {c}" for c in constraints) if constraints else "None"
            prompt = f"""Based on the following knowledge, generate a solution for the task.

Knowledge:
{knowledge}

Constraints:
{constraint_str}

Task: {task}

Return JSON: {{"solution": "...", "approach": "...", "confidence": 0.0-1.0}}"""

            response = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1500)
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {"type": "llm_generated", **result}
            return {"type": "llm_generated", "solution": response[:500], "confidence": 0.5}
        except Exception as e:
            logger.warning(f"LLM generation error: {e}")
            return None

    def _generate_code_patch(self, task: str, inputs: dict) -> dict | None:
        """Generate code patch for implementation tasks."""
        existing_code = inputs.get("code", "")
        if not existing_code:
            return None

        try:
            prompt = f"""Generate a code patch for the following task. Return the patch in unified diff format.

Existing code:
```
{existing_code[:2000]}
```

Task: {task}

Return ONLY the patch in unified diff format."""

            response = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=2000)
            return {"type": "code_patch", "content": response, "confidence": 0.5}
        except Exception as e:
            logger.warning(f"Code patch generation error: {e}")
            return None
