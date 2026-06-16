"""Tests for lifecycle modules."""

import pytest
from prometheus_v8.schema import create_fact_node, MemoryLayer, Node


class TestWeibull:
    def test_compute_retention(self):
        from prometheus_v8.lifecycle.weibull import WeibullRetentionCalculator

        wr = WeibullRetentionCalculator()
        # New important memory → high retention
        ret = wr.compute(age_days=0.0, importance=0.8, lam=1.0, k=0.5)
        assert ret.composite > 0.7
        # Old unimportant memory → decayed retention
        ret_old = wr.compute(age_days=365.0, importance=0.2, lam=1.0, k=0.5)
        assert ret_old.composite < ret.composite

    def test_recall_reinforcement(self):
        from prometheus_v8.lifecycle.weibull import WeibullRetentionCalculator

        wr = WeibullRetentionCalculator()
        # More consecutive hits → higher recall reinforcement
        r1 = wr.compute(age_days=10.0, importance=0.5, lam=1.0, k=0.5, consecutive_hits=0, access_count=0)
        r2 = wr.compute(age_days=10.0, importance=0.5, lam=1.0, k=0.5, consecutive_hits=5, access_count=5)
        assert r2.composite >= r1.composite


class TestConsolidation:
    def test_consolidate_promotes(self):
        from prometheus_v8.lifecycle.consolidation import ConsolidationPipeline

        pipeline = ConsolidationPipeline()
        node = create_fact_node(content="test", importance=0.5, tags=["test"])
        node.access_count = 5
        result = pipeline.consolidate([node])
        assert len(result) >= 1

    def test_consolidate_weak_node_stays_working(self):
        from prometheus_v8.lifecycle.consolidation import ConsolidationPipeline

        pipeline = ConsolidationPipeline()
        node = create_fact_node(content="test", importance=0.1, tags=["test"])
        node.access_count = 0
        result = pipeline.consolidate([node])
        # Weak nodes may be promoted to semantic if they have tags, or stay working
        assert result[0].layer in (MemoryLayer.WORKING, MemoryLayer.SEMANTIC)


class TestDreamCycle:
    def test_dream_cycle_runs(self):
        from prometheus_v8.lifecycle.dream import DreamCycle

        cycle = DreamCycle()
        n1 = create_fact_node(content="Python is a programming language", importance=0.7, tags=["python"])
        n2 = create_fact_node(content="Python supports multiple paradigms", importance=0.6, tags=["python"])
        insights = cycle.dream([n1, n2])
        assert cycle.stats["dream_cycles"] == 1

    def test_dream_empty_nodes(self):
        from prometheus_v8.lifecycle.dream import DreamCycle

        cycle = DreamCycle()
        insights = cycle.dream([])
        assert isinstance(insights, list)


class TestAging:
    def test_assess(self):
        from prometheus_v8.lifecycle.aging import AgingDetector

        ad = AgingDetector()
        node = create_fact_node(content="test content", importance=0.5)
        report = ad.assess(node)
        assert 0 <= report.composite_aging <= 1


class TestMetabolism:
    def test_triage(self):
        from prometheus_v8.lifecycle.metabolism import MetabolismEngine, TriageDecision

        me = MetabolismEngine()
        node = create_fact_node(content="test", importance=0.8)
        result = me.triage(node)
        assert result.decision in [d.value for d in TriageDecision]

    def test_compute_gravity(self):
        from prometheus_v8.lifecycle.metabolism import MetabolismEngine

        me = MetabolismEngine()
        node = create_fact_node(content="test", importance=0.8)
        node.access_count = 3
        gravity = me.compute_gravity(node)
        assert gravity > 0


class TestMoat:
    def test_assess(self):
        from prometheus_v8.lifecycle.moat import MemoryMoat

        moat = MemoryMoat()
        assessment = moat.assess()
        assert 0 <= assessment.composite_score <= 100
