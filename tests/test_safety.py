"""Safety tests."""
from prometheus_v8.safety.manager import SafetyManager
from prometheus_v8.safety.circuit_breaker import CircuitBreaker, CircuitState
from prometheus_v8.safety.forbidden_ops import ForbiddenOpsChecker
from prometheus_v8.safety.plan_validator import PlanValidator

def test_circuit_breaker():
    cb = CircuitBreaker(threshold=3)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute()
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert not cb.can_execute()

def test_forbidden_ops():
    checker = ForbiddenOpsChecker()
    assert not checker.is_safe("rm -rf /")
    assert checker.is_safe("print('hello')")

def test_plan_validator():
    pv = PlanValidator()
    ok, reason = pv.validate_step("print('hello')")
    assert ok
    ok, reason = pv.validate_step("exec(malicious_code)")
    assert not ok

def test_safety_manager():
    sm = SafetyManager()
    verdict = sm.check("print('hello world')")
    assert verdict.allowed
    verdict = sm.check("rm -rf /")
    assert not verdict.allowed
