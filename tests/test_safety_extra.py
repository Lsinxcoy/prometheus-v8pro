"""Additional safety tests."""

import pytest


class TestChainValidator:
    def test_validate_clean_code(self):
        from prometheus_v8.safety.chain_validator import ChainValidator

        cv = ChainValidator()
        assert cv.validate("x = 1 + 2")

    def test_validate_dangerous_code(self):
        from prometheus_v8.safety.chain_validator import ChainValidator

        cv = ChainValidator()
        results = cv.validate_detailed("exec('malicious')")
        # validate_detailed returns list[ValidationResult]
        assert isinstance(results, list)
        assert any(not r.passed for r in results)


class TestSafeHarbor:
    def test_check_dangerous(self):
        from prometheus_v8.safety.safe_harbor import SafeHarborChecker

        sh = SafeHarborChecker()
        ok, reason = sh.check("exec(os.system('rm -rf /'))")
        assert not ok

    def test_check_safe(self):
        from prometheus_v8.safety.safe_harbor import SafeHarborChecker

        sh = SafeHarborChecker()
        ok, reason = sh.check("x = 1 + 2")
        assert ok


class TestDynamicSecurity:
    def test_initial_level(self):
        from prometheus_v8.safety.dynamic_security import DynamicSecurityManager

        dsm = DynamicSecurityManager()
        # Check that stats includes level info
        stats = dsm.stats
        assert "level" in stats


class TestConfidenceGate:
    def test_evaluate_high_confidence(self):
        from prometheus_v8.safety.confidence_gate import ConfidenceGate, ImprovementCard

        cg = ConfidenceGate()
        card = ImprovementCard(
            id="test_1",
            what_changed="optimize_search",
            why="Improve search speed",
            expected_impact="2x faster",
            confidence=0.9,
            rollback_plan="revert commit",
        )
        action = cg.evaluate(card)
        assert action.value in ("proceed", "ask", "defer")

    def test_low_confidence(self):
        from prometheus_v8.safety.confidence_gate import ConfidenceGate, ImprovementCard

        cg = ConfidenceGate()
        card = ImprovementCard(
            id="test_2",
            what_changed="risky_refactor",
            why="Rebuild entire module",
            expected_impact="unknown",
            confidence=0.1,
            rollback_plan="undo",
            risks=["could break everything"],
        )
        action = cg.evaluate(card)
        # Low confidence should result in ask or defer
        assert action.value in ("ask", "defer")


class TestSafetyVerdict:
    def test_verdict_has_reason(self):
        from prometheus_v8.safety.manager import SafetyVerdict

        v = SafetyVerdict(allowed=True, reason="All checks passed")
        assert v.reason == "All checks passed"

    def test_safety_check_returns_reason(self):
        from prometheus_v8.safety.manager import SafetyManager

        sm = SafetyManager()
        v = sm.check("x = 1 + 2")
        assert hasattr(v, "reason")
        assert v.reason != ""
