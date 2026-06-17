"""Tests for new evolution modules: checkpoint, evolution_loop, harness."""

import os
import tempfile
import time

import pytest

from prometheus_v8.evolution.checkpoint import CheckpointData, EvolutionCheckpoint
from prometheus_v8.evolution.evolution_loop import (
    EvolutionLoop,
    LoopConfig,
    LoopState,
    LoopStatus,
    OrganFeedbackCollector,
)
from prometheus_v8.evolution.harness import (
    ArchitectureSnapshot,
    HarnessAction,
    HarnessComponent,
    HarnessDesign,
    HarnessDesignSystem,
    HarnessEvaluation,
    HarnessEvolutionEngine,
)
from prometheus_v8.schema import Genome


# ── Checkpoint Tests ──


class TestEvolutionCheckpoint:
    def test_save_and_load(self):
        """Checkpoint save → load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = EvolutionCheckpoint(checkpoint_dir=tmpdir, interval_generations=5)
            ckpt.save(
                generation=10,
                best_fitness=0.75,
                genome_data={"fitness": 0.75, "skills": ["test"]},
                layer_stats=[{"name": "L0", "count": 10}],
                history_tail=[],
                metadata={"restart_count": 0},
            )
            loaded = ckpt.load_latest()
            assert loaded is not None
            assert loaded.generation == 10
            assert loaded.best_fitness == 0.75
            assert loaded.genome_data["fitness"] == 0.75
            assert loaded.genome_data["skills"] == ["test"]
            assert loaded.layer_stats[0]["name"] == "L0"

    def test_load_nonexistent(self):
        """Loading from empty directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = EvolutionCheckpoint(checkpoint_dir=tmpdir)
            assert ckpt.load_latest() is None

    def test_should_save_interval(self):
        """should_save returns True at interval boundaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = EvolutionCheckpoint(checkpoint_dir=tmpdir, interval_generations=10)
            # First call: _last_save_gen=-1, so gen 10 - (-1) = 11 >= 10 → True
            assert ckpt.should_save(10, 0.0) is True
            # After saving, _last_save_gen=10, so gen 15 - 10 = 5 < 10 → False (if no milestone)
            ckpt._last_save_gen = 10
            ckpt._last_milestone_fitness = 0.0
            assert ckpt.should_save(15, 0.0) is False
            # gen 20 - 10 = 10 >= 10 → True
            assert ckpt.should_save(20, 0.0) is True

    def test_should_save_milestone(self):
        """should_save returns True for fitness milestones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = EvolutionCheckpoint(checkpoint_dir=tmpdir, interval_generations=100)
            ckpt._last_save_gen = 0
            ckpt._last_milestone_fitness = 0.5
            # 5% improvement threshold (default milestone_threshold=0.05)
            assert ckpt.should_save(15, 0.55) is True
            assert ckpt.should_save(15, 0.51) is False

    def test_cleanup_old_checkpoints(self):
        """Old periodic checkpoints are cleaned up when keep_last_n is exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = EvolutionCheckpoint(checkpoint_dir=tmpdir, interval_generations=1, keep_last_n=3)
            for i in range(5):
                ckpt.save(
                    generation=i + 1,
                    best_fitness=0.1 * (i + 1),
                    genome_data={},
                    layer_stats=[],
                    history_tail=[],
                    metadata={},
                )
            # Only keep_last_n periodic + milestones should remain
            files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            assert len(files) <= 5  # 3 periodic + up to 2 milestones


class TestCheckpointData:
    def test_creation(self):
        """CheckpointData can be created with defaults."""
        cd = CheckpointData(generation=5, best_fitness=0.8)
        assert cd.generation == 5
        assert cd.best_fitness == 0.8
        assert cd.genome_data == {}
        assert cd.layer_stats == []


# ── OrganFeedbackCollector Tests ──


class TestOrganFeedbackCollector:
    def test_register_and_collect(self):
        """Register feedback callbacks and collect from them."""
        collector = OrganFeedbackCollector()
        collector.register("taotie", lambda: {"score": 0.8, "items_extracted": 5})
        collector.register("nuwa", lambda: {"score": 0.6, "items_generated": 3})

        feedback = collector.collect()
        assert "taotie" in feedback
        assert "nuwa" in feedback
        assert feedback["taotie"]["score"] == 0.8
        assert feedback["nuwa"]["items_generated"] == 3

    def test_unregister(self):
        """Unregistered organs are not collected."""
        collector = OrganFeedbackCollector()
        collector.register("taotie", lambda: {"score": 0.8})
        collector.unregister("taotie")

        feedback = collector.collect()
        assert "taotie" not in feedback
        assert collector.registered_organs == []

    def test_error_in_callback(self):
        """Errors in feedback callbacks are logged and skipped."""
        collector = OrganFeedbackCollector()
        collector.register("good", lambda: {"score": 1.0})
        collector.register("bad", lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        feedback = collector.collect()
        assert "good" in feedback
        assert "bad" not in feedback

    def test_registered_organs_list(self):
        """registered_organs returns list of names."""
        collector = OrganFeedbackCollector()
        collector.register("a", lambda: {})
        collector.register("b", lambda: {})
        assert set(collector.registered_organs) == {"a", "b"}


# ── Harness Design System Tests ──


class TestHarnessDesignSystem:
    def test_analyze_empty_snapshot(self):
        """Analyze with empty snapshot produces no critical designs."""
        ds = HarnessDesignSystem()
        snapshot = ArchitectureSnapshot()
        designs = ds.analyze(snapshot)
        # Empty snapshot may produce designs for missing capabilities
        for d in designs:
            assert isinstance(d, HarnessDesign)
            assert d.expected_impact >= 0

    def test_analyze_underperforming_layer(self):
        """Underperforming layer is detected."""
        ds = HarnessDesignSystem()
        snapshot = ArchitectureSnapshot(
            layer_names=["L0", "L1", "L2"],
            layer_execution_counts=[100, 100, 100],
            layer_avg_deltas=[0.1, 0.05, 0.0001],
        )
        designs = ds.analyze(snapshot)
        underperforming = [d for d in designs if d.action == HarnessAction.MODIFY_LAYER_WEIGHT]
        assert len(underperforming) >= 1
        assert underperforming[0].parameters["layer_id"] == 2  # L2 is underperforming

    def test_analyze_stagnation(self):
        """High stagnation episodes trigger search space expansion."""
        ds = HarnessDesignSystem()
        snapshot = ArchitectureSnapshot(stagnation_episodes=5)
        designs = ds.analyze(snapshot)
        expand = [d for d in designs if d.action == HarnessAction.EXPAND_SEARCH_SPACE]
        assert len(expand) >= 1

    def test_evaluate_approved(self):
        """Design with >= 50% expected impact is approved."""
        ds = HarnessDesignSystem()
        design = HarnessDesign(expected_impact=0.1)
        evaluation = ds.evaluate(design, before_fitness=0.5, after_fitness=0.55)
        assert evaluation.approved is True
        assert abs(evaluation.actual_impact - 0.05) < 1e-9

    def test_evaluate_rejected(self):
        """Design with < 50% expected impact is rejected."""
        ds = HarnessDesignSystem()
        design = HarnessDesign(expected_impact=0.1)
        evaluation = ds.evaluate(design, before_fitness=0.5, after_fitness=0.51)
        assert evaluation.approved is False


# ── Harness Evolution Engine Tests ──


class TestHarnessEvolutionEngine:
    def test_take_snapshot(self):
        """Snapshot captures engine architecture."""
        engine = HarnessEvolutionEngine(evaluation_interval_generations=1)

        # Mock engine with layers
        class MockLayer:
            name = "L0"
            _execution_count = 10
            _total_fitness_delta = 0.5

        class MockEngine:
            _layers = [MockLayer()]
            _best_genome = Genome(code="test", fitness=0.8)
            _generation = 50

        mock = MockEngine()
        snapshot = engine.take_snapshot(mock)
        assert snapshot.layer_count == 1
        assert snapshot.layer_names == ["L0"]
        assert snapshot.layer_execution_counts == [10]
        assert snapshot.total_generations == 50

    def test_evaluate_and_apply_skips_before_interval(self):
        """Harness skips evaluation before interval is reached."""
        engine = HarnessEvolutionEngine(evaluation_interval_generations=100)
        genome = Genome(code="test")
        results = engine.evaluate_and_apply(None, genome)
        assert results == []

    def test_risk_rejection(self):
        """High-risk designs are rejected."""
        ds = HarnessDesignSystem()
        # Force a high-risk design by making a snapshot that triggers one
        # then manually test the risk check
        engine = HarnessEvolutionEngine(
            design_system=ds,
            max_risk_level=0.3,
            evaluation_interval_generations=1,
        )
        engine._generation_counter = 100  # Past interval

        # Inject a high-risk design
        high_risk = HarnessDesign(
            action=HarnessAction.ADAPT_STRATEGY,
            risk_level=0.9,
            expected_impact=0.1,
        )
        ds._design_history.append(high_risk)

        # We need the analyze to return a high-risk design
        original_analyze = ds.analyze

        def mock_analyze(snapshot):
            return [high_risk]

        ds.analyze = mock_analyze

        genome = Genome(code="test", fitness=0.5)
        evaluations = engine.evaluate_and_apply(None, genome)
        # High-risk design should be rejected
        assert len(evaluations) == 0

    def test_stats(self):
        """Stats returns expected fields."""
        engine = HarnessEvolutionEngine()
        stats = engine.stats
        assert "generation_counter" in stats
        assert "applied_count" in stats
        assert "rejected_count" in stats


# ── Evolution Loop Tests ──


class TestEvolutionLoop:
    def test_config_defaults(self):
        """LoopConfig has reasonable defaults."""
        config = LoopConfig()
        assert config.cycle_interval_seconds == 60.0
        assert config.max_generations_per_cycle == 5
        assert config.checkpoint_interval == 10
        assert config.stagnation_restart_threshold == 30

    def test_status_initial(self):
        """Loop starts in IDLE state."""
        genome = Genome(code="test")
        loop = EvolutionLoop(engine=None, genome=genome)
        assert loop.status.state == LoopState.IDLE
        assert loop.status.total_cycles == 0

    def test_organ_feedback_integration(self):
        """Loop can access its OrganFeedbackCollector."""
        genome = Genome(code="test")
        feedback = OrganFeedbackCollector()
        feedback.register("test_organ", lambda: {"signal": 1.0})
        loop = EvolutionLoop(engine=None, genome=genome, organ_feedback=feedback)
        assert "test_organ" in loop.organ_feedback.registered_organs

    def test_callbacks_registration(self):
        """Event callbacks can be registered."""
        genome = Genome(code="test")
        loop = EvolutionLoop(engine=None, genome=genome)
        improved = []
        stagnated = []
        restarted = []
        loop.on_fitness_improved(lambda f, g: improved.append((f, g)))
        loop.on_stagnation(lambda g: stagnated.append(g))
        loop.on_restart(lambda r: restarted.append(r))
        # Callbacks are stored (no direct way to test firing without a real engine)
        assert len(loop._on_fitness_improved) == 1
        assert len(loop._on_stagnation) == 1
        assert len(loop._on_restart) == 1


# ── OrganEvolutionBridge Enhanced Tests ──


class TestOrganEvolutionBridgeEnhanced:
    def test_register_organ_feedback(self):
        """Bridge can register organs with OrganFeedbackCollector."""
        from prometheus_v8.organs.bridge import OrganEvolutionBridge

        feedback = OrganFeedbackCollector()
        bridge = OrganEvolutionBridge(organ_feedback=feedback)

        class MockOrgan:
            name = "test_organ"
            stats = {"executions": 5}

        bridge.register_organ_feedback("test_organ", MockOrgan())
        assert "test_organ" in feedback.registered_organs

        collected = feedback.collect()
        assert "test_organ" in collected
        assert collected["test_organ"]["stats"]["executions"] == 5

    def test_record_organ_result(self):
        """Bridge records organ results for feedback."""
        from prometheus_v8.organs.bridge import OrganEvolutionBridge

        bridge = OrganEvolutionBridge()
        bridge.record_organ_result("taotie", {"success": True, "fitness": 0.8})
        assert "taotie" in bridge.stats["organs_with_results"]

    def test_stats_includes_feedback_info(self):
        """Bridge stats include feedback collector info when available."""
        from prometheus_v8.organs.bridge import OrganEvolutionBridge

        feedback = OrganFeedbackCollector()
        bridge = OrganEvolutionBridge(organ_feedback=feedback)
        bridge.register_organ_feedback("organ1", type("Mock", (), {"stats": {}})())
        stats = bridge.stats
        assert "registered_organs" in stats
        assert "organ1" in stats["registered_organs"]

    def test_no_feedback_collector(self):
        """Bridge works without OrganFeedbackCollector."""
        from prometheus_v8.organs.bridge import OrganEvolutionBridge

        bridge = OrganEvolutionBridge()
        # register_organ_feedback should not raise
        bridge.register_organ_feedback("test", type("Mock", (), {"stats": {}})())
        stats = bridge.stats
        assert "bridged" in stats


# ── DailyLearning Rule-based Fallback Tests ──


class TestDailyLearningFallback:
    def test_learn_without_llm(self):
        """Learn step produces key points without LLM."""
        from prometheus_v8.lifecycle.daily_learning import DailyLearningCycle

        cycle = DailyLearningCycle(llm=None)
        learned = cycle._learn("evolution", "Evolution is the process of change. It happens through mutation. Natural selection drives it forward.")
        assert len(learned) > 20  # Should extract meaningful content
        assert "evolution" in learned.lower() or "change" in learned.lower()

    def test_reflect_without_llm(self):
        """Reflect step produces structured concerns without LLM."""
        from prometheus_v8.lifecycle.daily_learning import DailyLearningCycle

        cycle = DailyLearningCycle(llm=None)
        reflected = cycle._reflect("Short")
        assert "concern" in reflected.lower() or "incomplete" in reflected.lower()

    def test_derive_without_llm(self):
        """Derive step produces When/Do principle without LLM."""
        from prometheus_v8.lifecycle.daily_learning import DailyLearningCycle

        cycle = DailyLearningCycle(llm=None)
        derived = cycle._derive("Reasoning: gap identified in test coverage")
        assert "When" in derived or "Principle" in derived

    def test_full_cycle_without_llm(self):
        """Full learning cycle works without LLM."""
        from prometheus_v8.lifecycle.daily_learning import DailyLearningCycle

        cycle = DailyLearningCycle(llm=None, daily_quota=100)
        result = cycle.run_cycle("testing", "Testing is important because it verifies correctness. Examples include unit tests and integration tests.")
        assert result.learned  # Not empty
        assert result.reflected  # Not empty
        assert result.reasoned  # Not empty
        assert result.derived  # Not empty
        assert result.score > 0  # Should have some score


# ── Nuwa Rule-based Generation Tests ──


class TestNuwaRuleBased:
    def test_code_task_without_llm(self):
        """Nuwa generates code analysis for code tasks without LLM."""
        from prometheus_v8.organs.nuwa import NuwaOrgan
        from prometheus_v8.organs.base import OrganContext

        organ = NuwaOrgan(llm=None)
        organ._llm = None  # Force no LLM to test rule-based path
        ctx = OrganContext(
            task="implement the refactoring code",
            inputs={"code": "def foo():\n    return 1\n\ndef bar():\n    pass\n"},
        )
        result = organ.execute(ctx)
        assert result.success
        gens = result.output["generations"]
        # Code task with code input triggers _generate_code_patch → _rule_based_code_patch
        types = [g["type"] for g in gens]
        assert "rule_based_patch" in types or any(t.startswith("rule_based") for t in types)

    def test_test_task_without_llm(self):
        """Nuwa generates test strategy for test tasks without LLM."""
        from prometheus_v8.organs.nuwa import NuwaOrgan
        from prometheus_v8.organs.base import OrganContext

        organ = NuwaOrgan(llm=None)
        organ._llm = None  # Force no LLM to test rule-based path
        ctx = OrganContext(task="verify the test coverage", inputs={})
        result = organ.execute(ctx)
        assert result.success
        types = [g["type"] for g in result.output["generations"]]
        assert "rule_based_test" in types

    def test_optimize_task_with_patterns(self):
        """Nuwa uses patterns for optimization tasks — patterns take priority over rule_based."""
        from prometheus_v8.organs.nuwa import NuwaOrgan
        from prometheus_v8.organs.base import OrganContext

        organ = NuwaOrgan(llm=None)
        organ._llm = None  # Force no LLM to test rule-based path
        ctx = OrganContext(
            task="optimize the performance",
            inputs={"dna": {"patterns": [{"type": "caching", "content": "cache results"}]}},
        )
        result = organ.execute(ctx)
        assert result.success
        types = [g["type"] for g in result.output["generations"]]
        # DNA patterns generate pattern_application first; rule_based only used as fallback
        assert "pattern_application" in types

    def test_optimize_task_without_patterns(self):
        """Nuwa generates rule_based_optimize when no patterns available."""
        from prometheus_v8.organs.nuwa import NuwaOrgan
        from prometheus_v8.organs.base import OrganContext

        organ = NuwaOrgan(llm=None)
        organ._llm = None  # Force no LLM
        ctx = OrganContext(
            task="optimize the performance",
            inputs={},  # No DNA patterns
        )
        result = organ.execute(ctx)
        assert result.success
        types = [g["type"] for g in result.output["generations"]]
        assert "rule_based_optimize" in types

    def test_rule_based_code_patch(self):
        """Nuwa generates code patch suggestions without LLM."""
        from prometheus_v8.organs.nuwa import NuwaOrgan
        from prometheus_v8.organs.base import OrganContext

        organ = NuwaOrgan(llm=None)
        organ._llm = None  # Force no LLM
        ctx = OrganContext(
            task="fix the code",
            inputs={"code": "try:\n    x = 1\nexcept:\n    pass"},
        )
        result = organ.execute(ctx)
        assert result.success
        # Should detect bare except and produce rule_based_patch
        types = [g["type"] for g in result.output["generations"]]
        assert "rule_based_patch" in types
