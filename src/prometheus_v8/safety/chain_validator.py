"""Chain Validator - 7-dimension validation chain."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

VALIDATION_DIMENSIONS = [
    "syntax",  # Is the content syntactically valid?
    "semantics",  # Does it make logical sense?
    "safety",  # Are there safety concerns?
    "completeness",  # Is the content complete?
    "consistency",  # Is it internally consistent?
    "relevance",  # Is it relevant to the context?
    "freshness",  # Is the information current?
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
        self._history: deque[list[ValidationResult]] = deque(maxlen=1000)

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
        score = 1.0
        # Bracket balance
        for open_ch, close_ch in [("(", ")"), ("[", "]"), ("{", "}")]:
            balance = 0
            for ch in content:
                if ch == open_ch:
                    balance += 1
                elif ch == close_ch:
                    balance -= 1
                if balance < 0:
                    score -= 0.2
                    break
            if balance != 0:
                score -= 0.2
        # Quote balance (simple check)
        for quote in ['"', "'"]:
            count = content.count(quote)
            if count % 2 != 0:
                score -= 0.1
        # Try compile
        try:
            compile(content, "<check>", "exec")
        except SyntaxError:
            score -= 0.3
        score = max(0.0, min(1.0, score))
        return ValidationResult("syntax", score > 0.5, score, "Syntax check (brackets + quotes + compile)")

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
        """Check internal consistency of the content."""
        words = content.lower().split()
        if len(words) < 5:
            return ValidationResult("consistency", True, 1.0, "Too short to check")

        score = 0.8
        # Check indentation consistency
        lines = content.split("\n")
        indent_styles = set()
        for line in lines:
            if line.startswith("    "):
                indent_styles.add("space4")
            elif line.startswith("\t"):
                indent_styles.add("tab")
            elif line.startswith("  ") and not line.startswith("    "):
                indent_styles.add("space2")
        if len(indent_styles) > 1:
            score -= 0.2  # Mixed indentation
        # Check bracket matching
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for ch in content:
            if ch in brackets:
                stack.append(ch)
            elif ch in brackets.values():
                if stack and brackets.get(stack[-1]) == ch:
                    stack.pop()
        if stack:
            score -= 0.2  # Unmatched brackets
        score = max(0.0, min(1.0, score))
        return ValidationResult("consistency", score > 0.5, score, "Consistency check (indentation + brackets)")

    def _check_relevance(self, content: str, context: str) -> ValidationResult:
        if not context:
            return ValidationResult("relevance", True, 0.8, "No context to check against")
        content_words = set(content.lower().split())
        context_words = set(context.lower().split())
        overlap = len(content_words & context_words) / max(1, len(context_words))
        score = min(1.0, overlap * 2)
        return ValidationResult("relevance", score > 0.3, score, f"Word overlap: {overlap:.2f}")

    def _check_freshness(self, content: str) -> ValidationResult:
        """Check if content uses up-to-date patterns."""
        stale_patterns = [
            "deprecated",
            "legacy",
            "obsolete",
            "outdated",
            "old_api",
            "python2",
            "py2",
            "print_statement",
            "xrange",
            "unicode",
            "raw_input",
            "execfile",
            "reduce(",
        ]
        content_lower = content.lower()
        stale_count = sum(1 for p in stale_patterns if p in content_lower)
        score = max(0.0, 1.0 - stale_count * 0.15)
        return ValidationResult("freshness", score > 0.5, score, f"Found {stale_count} stale patterns")
