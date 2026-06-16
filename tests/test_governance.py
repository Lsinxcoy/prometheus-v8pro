"""Tests for governance modules."""

import pytest
from prometheus_v8.governance.autonomy import AutonomyController, AutonomyLevel


class TestAutonomyController:
    def test_l0_allows_all(self):
        ac = AutonomyController(default_level=AutonomyLevel.L0_FULL_AUTO)
        allowed, level, reason = ac.can_execute("any_op")
        assert allowed is True

    def test_l4_forbids_all(self):
        ac = AutonomyController(default_level=AutonomyLevel.L4_FORBIDDEN)
        allowed, level, reason = ac.can_execute("any_op")
        assert allowed is False

    def test_l2_no_callback_allows(self):
        ac = AutonomyController(default_level=AutonomyLevel.L2_CONFIRM)
        # Without callback configured, L2 should default allow (fail-open)
        allowed, level, reason = ac.can_execute("normal_op")
        assert allowed is True

    def test_l3_no_callback_denies(self):
        ac = AutonomyController(default_level=AutonomyLevel.L3_APPROVAL)
        # Without callback configured, L3 should default deny (fail-closed)
        allowed, level, reason = ac.can_execute("dangerous_op")
        assert allowed is False


class TestTrustManager:
    def test_annotate(self):
        from prometheus_v8.governance.trust import TrustManager
        from prometheus_v8.schema import create_fact_node

        tm = TrustManager()
        node = create_fact_node(content="test", importance=0.5)
        level = tm.annotate(node, sources=["agent"])
        assert level is not None


class TestCuriosityQueue:
    def test_add_pop(self):
        from prometheus_v8.governance.curiosity import CuriosityQueue

        cq = CuriosityQueue()
        cq.add("Why does X happen?", priority=8)
        item = cq.pop()
        assert item is not None
        assert "X" in item.question

    def test_empty_pop(self):
        from prometheus_v8.governance.curiosity import CuriosityQueue

        cq = CuriosityQueue()
        item = cq.pop()
        assert item is None
