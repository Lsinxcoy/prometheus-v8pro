"""Safety Manager - Aggregating 6 safety modules."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.safety.circuit_breaker import CircuitBreaker
from prometheus_v8.safety.forbidden_ops import ForbiddenOpsChecker
from prometheus_v8.safety.chain_validator import ChainValidator
from prometheus_v8.safety.dynamic_security import DynamicSecurityManager
from prometheus_v8.safety.safe_harbor import SafeHarborChecker
from prometheus_v8.safety.plan_validator import PlanValidator

logger = logging.getLogger(__name__)

@dataclass
class SafetyVerdict:
    """Combined safety verdict."""
    allowed: bool = True
    risk_level: str = "low"  # low/medium/high/critical
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
    
    def check(self, content: str, context: dict | None = None, plan: list[str] | None = None) -> SafetyVerdict:
        """Run all safety checks."""
        self._check_count += 1
        verdict = SafetyVerdict(checks_total=6)
        
        # 1. Circuit breaker
        if not self._circuit_breaker.can_execute():
            verdict.allowed = False
            verdict.violations.append("circuit_breaker_open")
            verdict.risk_level = "critical"
        
        # 2. Forbidden operations
        forbidden_results = self._forbidden_ops.check(content)
        if forbidden_results:
            verdict.violations.extend(forbidden_results)
            verdict.allowed = False
            verdict.risk_level = "critical"
        else:
            verdict.checks_passed += 1
        
        # 3. Chain validation
        chain_ok = self._chain_validator.validate(content)
        if not chain_ok:
            verdict.violations.append("chain_validation_failed")
            verdict.risk_level = "high"
        else:
            verdict.checks_passed += 1
        
        # 4. Dynamic security
        risk = self._dynamic_security.assess_risk(content, context)
        if risk == "critical":
            verdict.allowed = False
            verdict.risk_level = "critical"
        elif risk == "high":
            verdict.risk_level = "high"
        else:
            verdict.checks_passed += 1
        
        # 5. Safe harbor
        harbor_ok, harbor_reason = self._safe_harbor.check(content)
        if not harbor_ok:
            verdict.violations.append(f"safe_harbor: {harbor_reason}")
        else:
            verdict.checks_passed += 1
        
        # 6. Plan validation
        if plan:
            plan_ok, plan_reason = self._plan_validator.validate_plan(plan)
            if not plan_ok:
                verdict.violations.append(f"plan_validator: {plan_reason}")
            else:
                verdict.checks_passed += 1
        else:
            verdict.checks_passed += 1
        
        if verdict.violations:
            self._violation_count += 1
            self._circuit_breaker.record_failure()
        else:
            self._circuit_breaker.record_success()
        
        return verdict
    
    def check_operation(self, operation: str, content: str) -> SafetyVerdict:
        """Quick check for a specific operation type."""
        return self.check(content, context={"operation": operation})
    
    @property
    def stats(self) -> dict[str, Any]:
        return {"checks": self._check_count, "violations": self._violation_count,
                "violation_rate": self._violation_count / max(1, self._check_count)}
