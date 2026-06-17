"""Daily Learning Cycle - 5-step: Learn->Reflect->Reason->Derive->Apply.

From MiMo V5 dropped feature, restored and enhanced.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v8.schema import create_learning_node

logger = logging.getLogger(__name__)


@dataclass
class LearningRound:
    """One round of daily learning."""

    id: str = ""
    topic: str = ""
    learned: str = ""
    reflected: str = ""
    reasoned: str = ""
    derived: str = ""
    applied: str = ""
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DailyLearningCycle:
    """5-step daily learning cycle:
    1. LEARN: Acquire new knowledge from exploration
    2. REFLECT: Review what was learned, identify gaps
    3. REASON: Apply reasoning to derive implications
    4. DERIVE: Extract actionable principles
    5. APPLY: Apply derived principles to current tasks

    With quota management: max 20 rounds/day, revision every 5 rounds.
    """

    def __init__(self, store=None, llm=None, daily_quota: int = 20, revision_interval: int = 5) -> None:
        self._store = store
        self._llm = llm
        self._daily_quota = daily_quota
        self._revision_interval = revision_interval
        self._rounds_today = 0
        self._today = time.strftime("%Y-%m-%d")
        self._all_rounds: list[LearningRound] = []
        self._revision_count = 0

    def run_cycle(self, topic: str, content: str) -> LearningRound:
        """Run one 5-step learning cycle."""
        self._check_day_reset()

        if self._rounds_today >= self._daily_quota:
            logger.info("Daily learning quota reached")
            return LearningRound(topic=topic, score=0.0)

        round_id = f"lr_{self._rounds_today}_{int(time.time())}"
        lr = LearningRound(id=round_id, topic=topic)

        lr.learned = self._learn(topic, content)
        lr.reflected = self._reflect(lr.learned)
        lr.reasoned = self._reason(lr.learned, lr.reflected)
        lr.derived = self._derive(lr.reasoned)
        lr.applied = self._apply(lr.derived)
        lr.score = self._score_round(lr)

        node = create_learning_node(
            content=f"[{topic}] {lr.derived}",
            importance=lr.score,
        )
        if self._store:
            self._store.add_node(node)

        self._rounds_today += 1
        self._all_rounds.append(lr)

        if self._rounds_today % self._revision_interval == 0:
            self._force_revision()

        return lr

    def _learn(self, topic: str, content: str) -> str:
        """Step 1: Acquire knowledge."""
        if self._llm:
            try:
                prompt = f"Summarize key points about: {topic}\nContent: {content[:500]}"
                resp = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.3)
                return resp[:500]
            except Exception as e:
                logger.warning(f"Learn step failed: {e}")

        # Rule-based: extract key sentences from content
        return self._extract_key_points(topic, content)

    def _reflect(self, learned: str) -> str:
        """Step 2: Reflect on what was learned."""
        if self._llm:
            try:
                prompt = f"Reflect on this knowledge. What might be missing or wrong?\n{learned[:300]}"
                resp = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.5)
                return resp[:300]
            except Exception as e:
                logger.warning(f"Reflect step failed: {e}")

        # Rule-based: structured reflection checklist
        return self._rule_based_reflect(learned)

    def _reason(self, learned: str, reflected: str) -> str:
        """Step 3: Reason about implications."""
        if self._llm:
            try:
                prompt = f"What are the implications and consequences?\nLearned: {learned[:200]}\nReflection: {reflected[:200]}"
                resp = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.5)
                return resp[:300]
            except Exception as e:
                logger.warning(f"Reason step failed: {e}")

        # Rule-based: identify implications from learned+reflected
        return self._rule_based_reason(learned, reflected)

    def _derive(self, reasoned: str) -> str:
        """Step 4: Derive actionable principles."""
        if self._llm:
            try:
                prompt = (
                    f"Extract one actionable principle from this reasoning:\n{reasoned[:300]}\nFormat: When X, do Y"
                )
                resp = self._llm.complete([{"role": "user", "content": prompt}], temperature=0.3)
                return resp[:200]
            except Exception as e:
                logger.warning(f"Derive step failed: {e}")

        # Rule-based: derive principle from reasoning keywords
        return self._rule_based_derive(reasoned)

    def _apply(self, derived: str) -> str:
        """Step 5: Apply to current context."""
        if self._store:
            try:
                # Search for relevant existing knowledge to update
                results = self._store.search(derived[:50], limit=3)
                if results:
                    # Reinforce existing related knowledge
                    for node in results:
                        node.access_count += 1
                        node.consecutive_hits += 1
                    return f"Applied: reinforced {len(results)} related knowledge nodes with '{derived[:80]}'"
                else:
                    return f"Applied: no existing knowledge to reinforce, principle queued: {derived[:80]}"
            except Exception as e:
                logger.warning(f"Apply step failed: {e}")
                return f"Application queued: {derived[:100]}"
        return f"Application queued: {derived[:100]}"

    # --- Rule-based fallback methods ---

    def _extract_key_points(self, topic: str, content: str) -> str:
        """Extract key sentences from content using heuristic rules."""
        if not content:
            return f"Topic: {topic} (no content available)"

        sentences = [s.strip() for s in content.replace("。", ".").replace("！", "!").replace("？", "?").split(".") if s.strip()]
        if not sentences:
            return content[:300]

        # Prioritize sentences that contain topic keywords or are definition-like
        topic_words = set(topic.lower().split())
        scored = []
        for s in sentences[:20]:  # Limit processing
            score = 0
            s_lower = s.lower()
            for w in topic_words:
                if w in s_lower:
                    score += 2
            if any(kw in s_lower for kw in ["is", "are", "means", "defines", "是", "即", "指"]):
                score += 1
            if len(s) > 20:  # Prefer substantial sentences
                score += 1
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:5]]
        return ". ".join(top)[:500]

    def _rule_based_reflect(self, learned: str) -> str:
        """Structured reflection when LLM is unavailable."""
        concerns = []
        if len(learned) < 50:
            concerns.append("Knowledge appears incomplete (very short)")
        if not any(kw in learned.lower() for kw in ["because", "since", "due to", "因为", "由于"]):
            concerns.append("Missing causal explanations")
        if not any(kw in learned.lower() for kw in ["example", "instance", "例如", "比如"]):
            concerns.append("No concrete examples provided")
        if not any(kw in learned.lower() for kw in ["limitation", "constraint", "限制", "约束"]):
            concerns.append("Limitations not discussed")

        if concerns:
            return "Reflection concerns: " + "; ".join(concerns)
        return "Reflection: knowledge appears well-structured but should verify accuracy"

    def _rule_based_reason(self, learned: str, reflected: str) -> str:
        """Rule-based reasoning from learned and reflected content."""
        implications = []

        # Detect potential implications from reflection concerns
        if "incomplete" in reflected.lower() or "不完整" in reflected:
            implications.append("Additional research needed to fill knowledge gaps")
        if "causal" in reflected.lower():
            implications.append("Understanding root causes may reveal new optimization opportunities")
        if "examples" in reflected.lower() or "example" in reflected.lower():
            implications.append("Concrete examples should be gathered to validate theory")

        # Detect scope from learned content
        if any(kw in learned.lower() for kw in ["all", "every", "always", "所有", "总是"]):
            implications.append("Universal claims need careful verification")

        if implications:
            return "Reasoning: " + "; ".join(implications)
        return "Reasoning: apply logical deduction from available knowledge"

    def _rule_based_derive(self, reasoned: str) -> str:
        """Derive an actionable principle from reasoning."""
        # Try to extract a When/Do pattern from reasoning
        if "gap" in reasoned.lower() or "缺口" in reasoned:
            return "When knowledge gaps are identified, prioritize targeted research before proceeding"
        if "causal" in reasoned.lower() or "原因" in reasoned:
            return "When root causes are understood, design interventions at the causal level"
        if "verify" in reasoned.lower() or "验证" in reasoned:
            return "When claims lack evidence, implement verification before relying on them"
        if "optimization" in reasoned.lower() or "优化" in reasoned:
            return "When optimization opportunities exist, measure baseline before applying changes"

        # Generic principle from reasoning content
        if len(reasoned) > 20:
            return f"When encountering similar situations, apply: {reasoned[:80]}"
        return "Principle: apply insight from reasoning"

    def _score_round(self, lr: LearningRound) -> float:
        """Score the quality of a learning round."""
        score = 0.0
        if lr.learned and len(lr.learned) > 50:
            score += 0.2
        if lr.reflected and len(lr.reflected) > 30:
            score += 0.2
        if lr.reasoned and len(lr.reasoned) > 30:
            score += 0.2
        if lr.derived and ("When" in lr.derived or "do" in lr.derived):
            score += 0.3
        if lr.applied:
            score += 0.1
        return min(1.0, score)

    def _force_revision(self) -> None:
        """Force revision round (from knowledge-conversion solution)."""
        self._revision_count += 1
        logger.info(f"Revision round #{self._revision_count} triggered at round {self._rounds_today}")

    def _check_day_reset(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            self._rounds_today = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "rounds_today": self._rounds_today,
            "daily_quota": self._daily_quota,
            "total_rounds": len(self._all_rounds),
            "revisions": self._revision_count,
        }
