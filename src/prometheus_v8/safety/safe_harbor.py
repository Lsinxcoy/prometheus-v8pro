"""Safe Harbor - 5 rule pairs (prohibition + exemption)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RulePair:
    """A prohibition rule with its exemption condition."""

    name: str = ""
    prohibition: str = ""  # Regex pattern for prohibited content
    exemption: str = ""  # Regex pattern for exempted content
    description: str = ""
    severity: str = "high"  # low/medium/high/critical

    def check(self, content: str) -> tuple[bool, str]:
        """Check content against this rule pair. Returns (is_safe, reason)."""
        if not re.search(self.prohibition, content, re.IGNORECASE):
            return True, ""  # No prohibition triggered

        # Prohibition triggered, check exemption
        if self.exemption and re.search(self.exemption, content, re.IGNORECASE):
            return True, f"Exempted: {self.name}"

        return False, f"Prohibited: {self.name} - {self.description}"


DEFAULT_RULE_PAIRS = [
    RulePair(
        "code_execution",
        r"\bexec\s*\(|\beval\s*\(|\bos\.system\s*\(",
        r"\bdef\s+test_\w+|\bdef\s+sandbox_\w+",
        "Unsandboxed code execution",
        "critical",
    ),
    RulePair(
        "file_deletion",
        r"\brm\s+-rf\b|\bshutil\.rmtree\b|\bos\.remove\b",
        r"\btemp_dir_\w+|\bcache_clean_\w+",
        "Destructive file operations",
        "critical",
    ),
    RulePair(
        "network_access",
        r"\brequests\.(get|post|put|delete)\s*\(|\burllib\b|\bhttpx\.client\b",
        r"localhost|127\.0\.0\.1|api_base|health_check",
        "External network access",
        "high",
    ),
    RulePair(
        "env_modification",
        r"\bos\.environ\[|\.env\b|API_KEY|SECRET|PASSWORD",
        r"config\.get|environ\.get|safe_load",
        "Environment variable modification",
        "high",
    ),
    RulePair(
        "privilege_escalation",
        r"\bsudo\b|\bchmod\s+777\b|\bchown\s+root\b",
        r"Dockerfile|docker",
        "Privilege escalation",
        "critical",
    ),
]


class SafeHarborChecker:
    """5 rule pairs: check exemption BEFORE prohibition (exemption-first)."""

    def __init__(self, rules: list[RulePair] | None = None, threshold: float = 0.7) -> None:
        self._rules = rules or DEFAULT_RULE_PAIRS
        self._threshold = threshold
        self._check_count = 0
        self._violation_count = 0

    def check(self, content: str) -> tuple[bool, str]:
        """Check content against all rule pairs. Returns (is_safe, reason)."""
        self._check_count += 1
        violations = []

        for rule in self._rules:
            is_safe, reason = rule.check(content)
            if not is_safe:
                violations.append(reason)

        if violations:
            self._violation_count += 1
            return False, "; ".join(violations)

        return True, ""

    def add_rule(self, rule: RulePair) -> None:
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    @property
    def stats(self) -> dict[str, Any]:
        return {"checks": self._check_count, "violations": self._violation_count, "rules": len(self._rules)}
