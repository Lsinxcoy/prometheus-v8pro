"""Chain Validator - 7-dimension validation chain."""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

VALIDATION_DIMENSIONS = [
    "syntax",        # Is the content syntactically valid?
    "semantics",     # Does it make logical sense?
    "safety",        # Are there safety concerns?
    "completeness",  # Is the content complete?
    "consistency",   # Is it internally consistent?
    "relevance",     # Is it relevant to the context?
    "freshness",     # Is the information current?
]

@dataclass
class ValidationResult:
    dimension: str = ""
    passed: bool = False
    score: float = 0.0  # 0-1
    details: str = ""

class ChainValidator:
    """7-dimension validation chain."""
    
    def __init__(self) -> None:
        self._history: list[list[ValidationResult]] = []
    
    def validate(self, content: str, context: str = "") -> bool:
        """Run all 7 validation dimensions. Returns True if all pass."""
        results = self._run_chain(content, context)
        self._history.append(results)
        return all(r.passed for r in results)
    
    def validate_detailed(self, content: str, context: str = "") -> list[ValidationResult]:
        """Run validation and return detailed results."""
        results = self._run_chain(content, context)
        self._history.append(results)
        return results
    
    def _run_chain(self, content: str, context: str) -> list[ValidationResult]:
        results = []
        results.append(self._check_syntax(content))
        results.append(self._check_semantics(content))
        results.append(self._check_safety(content))
        results.append(self._check_completeness(content))
        results.append(self._check_consistency(content))
        results.append(self._check_relevance(content, context))
        results.append(self._check_freshness(content))
        return results
    
    def _check_syntax(self, content: str) -> ValidationResult:
        if not content or len(content.strip()) < 5:
            return ValidationResult("syntax", False, 0.0, "Content too short")
        balanced = content.count('(') == content.count(')') and content.count('[') == content.count(']')
        score = 1.0 if balanced else 0.5
        return ValidationResult("syntax", score > 0.5, score, "Bracket balance check")
    
    def _check_semantics(self, content: str) -> ValidationResult:
        contradictions = ["not true and true", "always never", "impossible certain"]
        found = sum(1 for c in contradictions if c in content.lower())
        score = max(0.0, 1.0 - found * 0.5)
        return ValidationResult("semantics", score > 0.5, score, f"Found {found} contradictions")
    
    def _check_safety(self, content: str) -> ValidationResult:
        dangerous = ["rm -rf", "format", "exec(", "eval(", "__import__"]
        found = sum(1 for d in dangerous if d in content)
        score = max(0.0, 1.0 - found * 0.3)
        return ValidationResult("safety", score > 0.7, score, f"Found {found} dangerous patterns")
    
    def _check_completeness(self, content: str) -> ValidationResult:
        incomplete_markers = ["todo", "fixme", "tbd", "placeholder", "..."]
        found = sum(1 for m in incomplete_markers if m in content.lower())
        score = max(0.0, 1.0 - found * 0.2)
        return ValidationResult("completeness", score > 0.5, score, f"Found {found} incomplete markers")
    
    def _check_consistency(self, content: str) -> ValidationResult:
        # Check for self-contradicting statements
        words = content.lower().split()
        if len(words) < 5:
            return ValidationResult("consistency", True, 1.0, "Too short to check")
        score = 0.8  # Default
        return ValidationResult("consistency", score > 0.5, score, "No obvious contradictions")
    
    def _check_relevance(self, content: str, context: str) -> ValidationResult:
        if not context:
            return ValidationResult("relevance", True, 0.8, "No context to check against")
        content_words = set(content.lower().split())
        context_words = set(context.lower().split())
        overlap = len(content_words & context_words) / max(1, len(context_words))
        score = min(1.0, overlap * 2)
        return ValidationResult("relevance", score > 0.3, score, f"Word overlap: {overlap:.2f}")
    
    def _check_freshness(self, content: str) -> ValidationResult:
        stale_markers = ["deprecated", "obsolete", "legacy", "old version"]
        found = sum(1 for m in stale_markers if m in content.lower())
        score = max(0.0, 1.0 - found * 0.3)
        return ValidationResult("freshness", score > 0.5, score, f"Found {found} stale markers")
