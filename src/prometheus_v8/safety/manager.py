"""Safety Manager - Aggregating 6 safety modules."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from prometheus_v8.safety.chain_validator import ChainValidator
from prometheus_v8.safety.circuit_breaker import CircuitBreaker
from prometheus_v8.safety.dynamic_security import DynamicSecurityManager
from prometheus_v8.safety.forbidden_ops import ForbiddenOpsChecker
from prometheus_v8.safety.plan_validator import PlanValidator
from prometheus_v8.safety.safe_harbor import SafeHarborChecker

logger = logging.getLogger(__name__)


@dataclass
class SafetyVerdict:
    """Combined safety verdict."""

    allowed: bool = True
    risk_level: str = "low"  # low/medium/high/critical
    reason: str = ""  # Human-readable reason for the verdict
    violations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0


class SafetyManager:
    """Unified safety manager aggregating 6 modules."""

    def __init__(self, circuit_breaker_threshold: int = 5, safe_harbor_threshold: float = 0.7) -> None:
        self._circuit_breaker = CircuitBreaker(threshold=circuit_breaker_threshold)
        self._forbidden_ops = ForbiddenOpsChecker()
        self._chain_validator = ChainValidator()
        self._dynamic_security = DynamicSecurityManager()
        self._safe_harbor = SafeHarborChecker(threshold=safe_harbor_threshold)
        self._plan_validator = PlanValidator()
        self._check_count = 0
        self._violation_count = 0
        self._audit_log: deque[dict] = deque(maxlen=10000)

    def check(self, content: str, context: dict | None = None, plan: list[str] | None = None) -> SafetyVerdict:
        """Run all safety checks."""
        self._check_count += 1
        verdict = SafetyVerdict(checks_total=6)

        # 1. Circuit breaker
        if not self._circuit_breaker.can_execute():
            verdict.allowed = False
            verdict.reason = "Circuit breaker is open"
            verdict.violations.append("circuit_breaker_open")
            verdict.risk_level = "critical"

        # 2. Forbidden operations
        forbidden_results = self._forbidden_ops.check(content)
        if forbidden_results:
            verdict.violations.extend(forbidden_results)
            verdict.allowed = False
            verdict.reason = f"Forbidden operations detected: {', '.join(forbidden_results)}"
            verdict.risk_level = "critical"
        else:
            verdict.checks_passed += 1

        # 3. Chain validation
        chain_ok = self._chain_validator.validate(content)
        if not chain_ok:
            verdict.violations.append("chain_validation_failed")
            verdict.reason = "Chain validation failed"
            verdict.risk_level = "high"
        else:
            verdict.checks_passed += 1

        # 4. Dynamic security
        risk = self._dynamic_security.assess_risk(content, context)
        if risk == "critical":
            verdict.allowed = False
            verdict.reason = "Dynamic security assessment: critical"
            verdict.risk_level = "critical"
        elif risk == "high":
            verdict.risk_level = "high"
        else:
            verdict.checks_passed += 1

        # 5. Safe harbor
        harbor_ok, harbor_reason = self._safe_harbor.check(content)
        if not harbor_ok:
            verdict.violations.append(f"safe_harbor: {harbor_reason}")
            verdict.reason = f"Safe harbor violation: {harbor_reason}"
        else:
            verdict.checks_passed += 1

        # 6. Plan validation
        if plan:
            plan_ok, plan_reason = self._plan_validator.validate_plan(plan)
            if not plan_ok:
                verdict.violations.append(f"plan_validator: {plan_reason}")
                verdict.reason = f"Plan validation failed: {plan_reason}"
            else:
                verdict.checks_passed += 1
        else:
            verdict.checks_passed += 1

        if verdict.violations:
            self._violation_count += 1
            self._circuit_breaker.record_failure()
        else:
            verdict.reason = "All checks passed"
            self._circuit_breaker.record_success()

        self._record_audit(content, verdict)
        return verdict

    def check_operation(self, operation: str, content: str) -> SafetyVerdict:
        """Quick check for a specific operation type."""
        return self.check(content, context={"operation": operation})

    def _record_audit(self, content: str, verdict: SafetyVerdict) -> None:
        """Record safety check to append-only audit log."""
        self._audit_log.append({
            "timestamp": time.time(),
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
            "allowed": verdict.allowed,
            "risk_level": verdict.risk_level,
            "reason": verdict.reason,
            "violations": verdict.violations[:],
            "checks_passed": verdict.checks_passed,
        })

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Return recent audit log entries."""
        return list(self._audit_log)[-limit:]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "checks": self._check_count,
            "violations": self._violation_count,
            "violation_rate": self._violation_count / max(1, self._check_count),
        }
