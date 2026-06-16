"""Guard Organ - Promote with SafeHarbor + PlanValidator."""

from __future__ import annotations

import json
import logging
import re

from prometheus_v8.organs.base import BaseOrgan, LLMClient, OrganContext, OrganEnv, OrganResult
from prometheus_v8.safety.plan_validator import PlanValidator
from prometheus_v8.safety.safe_harbor import SafeHarborChecker

logger = logging.getLogger(__name__)


class GuardOrgan(BaseOrgan):
    """Promote organ: final safety check before promotion to production."""

    def __init__(
        self, llm: LLMClient | None = None, env: OrganEnv | None = None, confidence_threshold: float = 0.5
    ) -> None:
        super().__init__("guard", llm, env)
        self._safe_harbor = SafeHarborChecker()
        self._plan_validator = PlanValidator()
        self._confidence_threshold = confidence_threshold

    def execute(self, context: OrganContext) -> OrganResult:
        inputs = context.inputs
        validated = inputs.get("validated", [])

        promoted = []
        rejected = []

        for variant in validated:
            # 1. SafeHarbor check
            harbor_ok, harbor_reason = self._safe_harbor.check(variant.get("content", ""))
            if not harbor_ok:
                rejected.append({**variant, "rejection_reason": f"safe_harbor: {harbor_reason}"})
                continue

            # 2. Plan validation (for code changes)
            if variant.get("type") in ("code_patch", "crossover"):
                plan_ok, plan_reason = self._plan_validator.validate_step(variant.get("content", ""))
                if not plan_ok:
                    rejected.append({**variant, "rejection_reason": f"plan_validator: {plan_reason}"})
                    continue

            # 3. Confidence threshold
            if variant.get("validation_score", 0) < self._confidence_threshold:
                rejected.append({**variant, "rejection_reason": "low_confidence"})
                continue

            # 4. LLM final review
            if self._llm:
                review = self._llm_review(variant, context.task)
                if review.get("approved", False):
                    variant["promotion_score"] = review.get("score", 0.7)
                    promoted.append(variant)
                else:
                    rejected.append({**variant, "rejection_reason": f"llm_review: {review.get('reason', 'unknown')}"})
            else:
                variant["promotion_score"] = variant.get("validation_score", 0.5)
                promoted.append(variant)

        return OrganResult(
            success=len(promoted) > 0,
            output={"promoted": promoted, "rejected": rejected},
            metadata={"promoted_count": len(promoted), "rejected_count": len(rejected)},
        )

    def _llm_review(self, variant: dict, task: str) -> dict:
        """LLM-based final review."""
        try:
            content = variant.get("content", "")[:1000]
            prompt = f"""Review this solution for safety and quality. Is it safe to promote to production?

Task: {task}
Solution: {content}
Score: {variant.get("validation_score", 0)}

Return JSON: {{"approved": true/false, "score": 0.0-1.0, "reason": "..."}}"""
            response = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=200)
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"approved": False, "reason": "llm_parse_error"}
        except Exception as e:
            logger.warning(f"LLM review error: {e}")
            return {"approved": False, "reason": "llm_unavailable"}
