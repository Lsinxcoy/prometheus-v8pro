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
            generations.append(self._generate_rule_based(task, inputs))

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

        if self._llm:
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

        # Rule-based code patch: extract function signatures and suggest improvements
        return self._rule_based_code_patch(task, existing_code)

    def _generate_rule_based(self, task: str, inputs: dict) -> dict:
        """Rule-based generation when LLM is unavailable.

        Analyzes the task keywords and available inputs to produce
        a structured, task-specific generation instead of a generic placeholder.
        """
        task_lower = task.lower()
        confidence = 0.4  # Base confidence for rule-based (higher than placeholder 0.3)

        # Detect task type and generate accordingly
        if any(kw in task_lower for kw in ["code", "implement", "fix", "refactor", "代码", "实现", "修复"]):
            code = inputs.get("code", "")
            if code:
                # Analyze code structure
                lines = code.split("\n")
                func_count = sum(1 for l in lines if l.strip().startswith("def "))
                class_count = sum(1 for l in lines if l.strip().startswith("class "))
                return {
                    "type": "rule_based_code",
                    "content": f"Code analysis: {len(lines)} lines, {func_count} functions, {class_count} classes. "
                               f"Suggest refactoring for: {task}",
                    "approach": "structural_analysis",
                    "confidence": min(0.6, confidence + 0.1 * min(func_count, 5)),
                }
            return {
                "type": "rule_based_code",
                "content": f"Code generation template for: {task}",
                "approach": "template_scaffold",
                "confidence": confidence,
            }

        if any(kw in task_lower for kw in ["test", "verify", "validate", "测试", "验证"]):
            return {
                "type": "rule_based_test",
                "content": f"Test strategy for: {task}",
                "approach": "boundary_and_edge_cases",
                "confidence": confidence + 0.05,
            }

        if any(kw in task_lower for kw in ["optimize", "improve", "enhance", "优化", "改进"]):
            dna = inputs.get("dna", {})
            patterns = dna.get("patterns", [])
            if patterns:
                return {
                    "type": "rule_based_optimize",
                    "content": f"Apply {len(patterns)} known optimization patterns to: {task}",
                    "approach": "pattern_application",
                    "confidence": confidence + 0.1 * min(len(patterns), 3),
                }
            return {
                "type": "rule_based_optimize",
                "content": f"Optimization analysis for: {task}",
                "approach": "profiling_guided",
                "confidence": confidence,
            }

        if any(kw in task_lower for kw in ["design", "architect", "plan", "设计", "架构"]):
            return {
                "type": "rule_based_design",
                "content": f"Design proposal for: {task}",
                "approach": "decomposition_and_interfaces",
                "confidence": confidence,
            }

        # Generic knowledge synthesis
        extracted = inputs.get("extracted", [])
        if extracted:
            return {
                "type": "rule_based_synthesis",
                "content": f"Synthesized from {len(extracted)} knowledge items: {task}",
                "approach": "knowledge_aggregation",
                "confidence": confidence + 0.05 * min(len(extracted), 5),
            }

        return {
            "type": "rule_based_generic",
            "content": f"Task analysis for: {task}",
            "approach": "decomposition",
            "confidence": 0.35,
        }

    def _rule_based_code_patch(self, task: str, code: str) -> dict:
        """Rule-based code patch generation when LLM is unavailable."""
        lines = code.split("\n")
        suggestions = []

        # Detect common issues
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Bare except clauses
            if stripped.startswith("except:"):
                suggestions.append(f"Line {i+1}: Replace bare 'except:' with 'except Exception:'")
            # Mutable default arguments
            if "def " in stripped and ("=[])" in stripped or "={})" in stripped):
                suggestions.append(f"Line {i+1}: Mutable default argument detected, use None instead")
            # Missing type hints in function definitions
            if stripped.startswith("def ") and "->" not in stripped and ":" in stripped:
                suggestions.append(f"Line {i+1}: Consider adding return type hint")

        if suggestions:
            return {
                "type": "rule_based_patch",
                "content": "\n".join(suggestions[:10]),
                "confidence": 0.45,
            }

        return {
            "type": "rule_based_patch",
            "content": f"No obvious issues found; manual review needed for: {task}",
            "confidence": 0.3,
        }
