"""Integration tests."""
import tempfile
import os
from prometheus_v8.schema import (
    create_fact_node, create_insight_node, create_episode_node,
    create_mutation_node, create_dream_node, create_curiosity_node,
    NodeType, MemoryLayer, TrustLevel,
)
from prometheus_v8.core.store import SQLiteStore
from prometheus_v8.lifecycle.metabolism import MetabolismEngine, TriageDecision
from prometheus_v8.safety.manager import SafetyManager
from prometheus_v8.evolution.engine import UnifiedEvolutionEngine
from prometheus_v8.governance.autonomy import AutonomyController, AutonomyLevel
from prometheus_v8.governance.trust import TrustManager
from prometheus_v8.governance.curiosity import CuriosityQueue

def test_full_pipeline():
    """Test the full pipeline: create → store → triage → safety → evolve."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        store = SQLiteStore(db_path)
        
        # Create and store nodes
        n1 = create_fact_node("Python is a programming language", importance=0.7, tags=["python"])
        n2 = create_insight_node("Type hints improve code quality", importance=0.8, tags=["python", "typing"])
        n3 = create_episode_node("Debugged memory leak in production", importance=0.6, tags=["debugging"])
        
        store.add_node(n1)
        store.add_node(n2)
        store.add_node(n3)
        
        assert store.count_nodes() >= 3
        
        # Search
        results = store.search_fts("Python")
        assert len(results) >= 1
        
        # Metabolism triage
        metabolism = MetabolismEngine(store=store)
        nodes = [n1, n2, n3]
        for node in nodes:
            result = metabolism.triage(node)
            assert result.decision in [TriageDecision.PROMOTE, TriageDecision.KEEP, 
                                        TriageDecision.DECAY, TriageDecision.DELETE, TriageDecision.ARCHIVE]
        
        # Safety check
        safety = SafetyManager()
        verdict = safety.check("print('hello world')")
        assert verdict.allowed
        
        verdict = safety.check("rm -rf /")
        assert not verdict.allowed
        
        # Autonomy
        ctrl = AutonomyController()
        can, level, reason = ctrl.can_execute("knowledge_compress")
        assert can  # L0 full auto
        
        can, level, reason = ctrl.can_execute("bypass_safety")
        assert not can  # L4 forbidden
        
        # Trust
        tm = TrustManager()
        level = tm.annotate(n1, sources=["source_a", "source_b"])
        assert level == TrustLevel.HIGH_SIGNAL
        
        # Curiosity
        cq = CuriosityQueue()
        cq.add("How does Weibull decay work?", priority=3)
        cq.add("What is CORAL heartbeat?", priority=5)
        item = cq.pop()
        assert item is not None
        assert item.priority == 3
        
    finally:
        store.close()
        os.unlink(db_path)

def test_evolution_engine():
    """Test the unified evolution engine."""
    engine = UnifiedEvolutionEngine()
    from prometheus_v8.schema import Genome
    genome = Genome(code="def hello():\n    return 'world'", fitness=0.3)
    
    result = engine.evolve(genome, max_generations=5, fitness_threshold=0.99)
    assert result is not None
    assert engine.generation <= 5

def test_safety_modules():
    """Test all safety modules together."""
    from prometheus_v8.safety.circuit_breaker import CircuitBreaker, CircuitState
    from prometheus_v8.safety.forbidden_ops import ForbiddenOpsChecker
    from prometheus_v8.safety.safe_harbor import SafeHarborChecker
    from prometheus_v8.safety.plan_validator import PlanValidator
    from prometheus_v8.safety.confidence_gate import ConfidenceGate, ImprovementCard, ConfidenceAction
    
    # Circuit breaker
    cb = CircuitBreaker(threshold=3)
    assert cb.can_execute()
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    
    # Forbidden ops
    foc = ForbiddenOpsChecker()
    assert not foc.is_safe("exec(malicious)")
    assert foc.is_safe("print('safe')")
    
    # Safe harbor
    sh = SafeHarborChecker()
    ok, reason = sh.check("def safe_function(): pass")
    assert ok
    
    # Plan validator
    pv = PlanValidator()
    ok, _ = pv.validate_plan(["read_data", "process_data", "output_results"])
    assert ok
    
    # Confidence gate
    gate = ConfidenceGate()
    card = gate.create_card("optimize_search", "Improve search speed", "2x faster",
                          category="code", confidence=0.9, rollback_plan="revert commit")
    action = gate.evaluate(card)
    assert action == ConfidenceAction.PROCEED
