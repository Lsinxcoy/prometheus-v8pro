"""Tests for knowledge layer: KnowledgeLayer, ConversionResult, ActionHook, KnowledgeGap."""

import pytest

from prometheus_v8.knowledge.layer import (
    KnowledgeLayer,
    KnowledgeGap,
    ActionHook,
    ConversionResult,
)
from prometheus_v8.schema import Node, NodeType, TrustLevel, create_fact_node


class TestKnowledgeLayer:
    """Tests for KnowledgeLayer: conversion, revision trigger, action hooks."""

    def test_structured_to_unstructured_fact(self):
        kl = KnowledgeLayer(store=None, llm=None)
        node = create_fact_node(content="the sky is blue", importance=0.6)
        result = kl.structured_to_unstructured(node)
        assert result.success
        assert result.target_type == "natural_language"
        assert "sky is blue" in result.output_content

    def test_structured_to_unstructured_insight(self):
        kl = KnowledgeLayer(store=None, llm=None)
        node = Node(type=NodeType.INSIGHT, payload=Node.__dataclass_fields__["payload"].default_factory())
        node.payload.content = "patterns emerge from data"
        result = kl.structured_to_unstructured(node)
        assert result.success
        assert "insight" in result.output_content.lower()

    def test_unstructured_to_structured(self):
        kl = KnowledgeLayer(store=None, llm=None)
        result = kl.unstructured_to_structured("Python is a programming language")
        assert result.success
        assert result.target_type == "structured"

    def test_action_hook_matching(self):
        kl = KnowledgeLayer(store=None, llm=None)
        hook = kl.add_action_hook("error", "run diagnostics", confidence=0.8)
        assert isinstance(hook, ActionHook)

        matching = kl.check_action_hooks("an error occurred in the system")
        assert len(matching) == 1

    def test_action_hook_execution(self):
        kl = KnowledgeLayer(store=None, llm=None)
        kl.add_action_hook("deploy", "run tests first")

        results = kl.execute_action_hooks("time to deploy to production")
        assert len(results) == 1
        assert "deploy" in results[0]

    def test_revision_trigger(self):
        kl = KnowledgeLayer(store=None, llm=None, revision_interval=2)
        node = create_fact_node(content="test1", importance=0.5)
        kl.structured_to_unstructured(node)
        kl.structured_to_unstructured(node)
        # After 2 operations, revision should have been triggered
        assert kl.stats["revisions"] >= 1

    def test_gap_detection_no_store(self):
        kl = KnowledgeLayer(store=None, llm=None)
        gaps = kl.detect_gaps()
        # Without a store, all domains should be gaps
        assert isinstance(gaps, list)

    def test_gap_recommendations(self):
        kl = KnowledgeLayer(store=None, llm=None)
        recs = kl.get_gap_recommendations()
        assert isinstance(recs, list)

    def test_stats(self):
        kl = KnowledgeLayer(store=None, llm=None)
        stats = kl.stats
        assert "operations" in stats
        assert "domains_tracked" in stats
        assert stats["domains_tracked"] == 8


class TestActionHook:
    """Tests for ActionHook matching and execution."""

    def test_matches(self):
        hook = ActionHook(trigger="deploy", action="run tests")
        assert hook.matches("deploy to production") is True
        assert hook.matches("write code") is False

    def test_execute(self):
        hook = ActionHook(trigger="error", action="alert")
        result = hook.execute()
        assert result == "alert"
        assert hook.execution_count == 1


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_creation(self):
        cr = ConversionResult(
            source_type="fact",
            target_type="natural_language",
            input_content="test",
            output_content="It is known that test.",
            success=True,
            fidelity=0.8,
        )
        assert cr.success
        assert cr.fidelity == 0.8
