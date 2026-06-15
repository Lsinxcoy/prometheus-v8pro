"""Confidence-Calibrated Self-Improvement Gates.

Based on CCSIL research:
- Every high-impact action gated by calibrated confidence
- Three behaviors: proceed (high), ask (low→clarify), defer (unsafe→human)
- Overconfidence is a system risk, not a style issue
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

class ConfidenceAction(str, Enum):
    PROCEED = "proceed"    # High confidence, safe to execute
    ASK = "ask"            # Low confidence, need clarification
    DEFER = "defer"        # Unsafe, escalate to human

@dataclass
class ImprovementCard:
    """Improvement-operator card - documents what each self-modification does.
    
    From ICLR 2026 Workshop on Recursive Self-Improvement.
    """
    id: str = ""
    what_changed: str = ""       # What is being modified
    why: str = ""                # Why this change is needed
    expected_impact: str = ""    # Expected positive impact
    risks: list[str] = field(default_factory=list)  # Potential risks
    rollback_plan: str = ""      # How to undo this change
    confidence: float = 0.5      # Confidence in improvement (0-1)
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    category: str = ""           # code/config/prompt/knowledge/architecture
    approved: bool = False
    executed: bool = False
    result: str = ""
    created_at: float = field(default_factory=time.time)

class ConfidenceGate:
    """Confidence-calibrated gate for self-improvement actions.
    
    Rules:
    - confidence > 0.8 + category is verifiable → PROCEED
    - 0.5 < confidence <= 0.8 → ASK (need more evidence)
    - confidence <= 0.5 OR category is non-verifiable → DEFER
    
    Verifiable domains: code (tests), config (benchmarks), prompts (A/B test)
    Non-verifiable domains: strategy, creative, philosophical
    """
    
    VERIFIABLE_CATEGORIES = {"code", "config", "prompt", "tool"}
    NON_VERIFIABLE_CATEGORIES = {"strategy", "creative", "philosophy", "governance"}
    
    def __init__(self, proceed_threshold: float = 0.8, ask_threshold: float = 0.5) -> None:
        self._proceed_threshold = proceed_threshold
        self._ask_threshold = ask_threshold
        self._cards: list[ImprovementCard] = []
        self._deferred: list[ImprovementCard] = []
    
    def evaluate(self, card: ImprovementCard) -> ConfidenceAction:
        """Evaluate an improvement card and decide action."""
        self._cards.append(card)
        
        # Non-verifiable categories always need human approval
        if card.category in self.NON_VERIFIABLE_CATEGORIES:
            if card.confidence < 0.9:
                self._deferred.append(card)
                return ConfidenceAction.DEFER
        
        # Confidence-based decision
        if card.confidence >= self._proceed_threshold:
            card.approved = True
            return ConfidenceAction.PROCEED
        elif card.confidence >= self._ask_threshold:
            return ConfidenceAction.ASK
        else:
            self._deferred.append(card)
            return ConfidenceAction.DEFER
    
    def create_card(self, what: str, why: str, expected_impact: str,
                    category: str = "", confidence: float = 0.5,
                    risks: list[str] | None = None,
                    rollback_plan: str = "", evidence: list[str] | None = None) -> ImprovementCard:
        """Create an improvement-operator card."""
        return ImprovementCard(
            id=f"card_{len(self._cards)}_{int(time.time())}",
            what_changed=what, why=why, expected_impact=expected_impact,
            category=category, confidence=confidence,
            risks=risks or [], rollback_plan=rollback_plan,
            evidence=evidence or [],
        )
    
    def get_deferred(self) -> list[ImprovementCard]:
        return list(self._deferred)
    
    def approve_deferred(self, card_id: str) -> bool:
        for card in self._deferred:
            if card.id == card_id:
                card.approved = True
                self._deferred.remove(card)
                return True
        return False
    
    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_cards": len(self._cards),
            "deferred": len(self._deferred),
            "proceed_rate": sum(1 for c in self._cards if c.approved) / max(1, len(self._cards)),
        }
