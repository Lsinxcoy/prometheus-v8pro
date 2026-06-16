"""Forbidden Operations - 20 forbidden patterns in 5 categories."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

FORBIDDEN_PATTERNS = {
    "destructive": [
        r"\brm\s+-rf\s+/",
        r"\bformat\s+[A-Z]:",
        r"\bdel\s+/[sS]",
        r"\bshutdown\s+",
        r"\bkill\s+-9\s+1\b",
    ],
    "data_exfiltration": [
        r"\bcurl\s+.*\|\s*sh\b",
        r"\bwget\s+.*\|\s*sh\b",
        r"\bscp\s+.*@",
        r"\baws\s+s3\s+cp\s+.*--no-sign",
    ],
    "privilege_escalation": [
        r"\bsudo\s+su\b",
        r"\bchmod\s+777\b",
        r"\bchown\s+root\b",
        r"\bpasswd\s+root\b",
    ],
    "network_attack": [
        r"\bdd\s+if=/dev/zero",
        r"\bping\s+-f\b",
        r"\bnmap\s+-sS\b",
        r"\bhping3\b",
    ],
    "code_injection": [
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\b__import__\s*\(",
        r"\bos\.system\s*\(",
    ],
}


class ForbiddenOpsChecker:
    """Check content against 20 forbidden operation patterns in 5 categories."""

    def __init__(self, custom_patterns: dict[str, list[str]] | None = None) -> None:
        self._patterns = FORBIDDEN_PATTERNS.copy()
        if custom_patterns:
            for category, patterns in custom_patterns.items():
                self._patterns.setdefault(category, []).extend(patterns)
        self._compiled: dict[str, list[re.Pattern]] = {}
        for category, patterns in self._patterns.items():
            self._compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def check(self, content: str) -> list[str]:
        """Check content for forbidden operations. Returns list of violations."""
        violations = []
        for category, patterns in self._compiled.items():
            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    violations.append(f"{category}: matched '{match.group()}' with pattern '{pattern.pattern}'")
        return violations

    def is_safe(self, content: str) -> bool:
        return len(self.check(content)) == 0

    def add_pattern(self, category: str, pattern: str) -> None:
        self._patterns.setdefault(category, []).append(pattern)
        self._compiled.setdefault(category, []).append(re.compile(pattern, re.IGNORECASE))

    def list_categories(self) -> dict[str, int]:
        return {cat: len(pats) for cat, pats in self._patterns.items()}
