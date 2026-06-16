"""Tests for MiroFish-inspired mechanisms: InsightForge, Ontology, Temporal, Persona, EventWriter, Retry."""
import time
import pytest


class TestInsightForge:
    """Test deep search with sub-query decomposition."""

    def test_deep_search_without_llm(self):
        """Without LLM, deep_search falls back to regular search."""
        from prometheus_v8.core.hybrid_search import HybridSearchEngine
        engine = HybridSearchEngine(store=None, vector_backend=None, llm=None)
        result = engine.deep_search("test query", k=5)
        assert isinstance(result, list)

    def test_decompose_query_without_llm(self):
        """Without LLM, _decompose_query returns original query."""
        from prometheus_v8.core.hybrid_search import HybridSearchEngine
        engine = HybridSearchEngine(store=None, vector_backend=None, llm=None)
        subqueries = engine._decompose_query("complex test query")
        assert isinstance(subqueries, list)
        assert len(subqueries) >= 1
        assert "complex test query" in subqueries


class TestDynamicOntology:
    """Test dynamic ontology generation."""

    def test_create_dynamic_ontology(self):
        from prometheus_v8.core.ontology_generator import DynamicOntology
        ont = DynamicOntology()
        assert len(ont.entity_types) == 0
        assert len(ont.relation_types) == 0

    def test_register_entity_type(self):
        from prometheus_v8.core.ontology_generator import DynamicOntology
        ont = DynamicOntology()
        ont.register_entity_type(name="TestEntity", description="A test entity")
        assert "TestEntity" in ont.entity_types
        assert ont.entity_types["TestEntity"].description == "A test entity"

    def test_register_relation_type(self):
        from prometheus_v8.core.ontology_generator import DynamicOntology
        ont = DynamicOntology()
        ont.register_relation_type(name="TEST_REL", description="A test relation")
        assert "TEST_REL" in ont.relation_types

    def test_generate_from_text_without_llm(self):
        from prometheus_v8.core.ontology_generator import OntologyGenerator
        gen = OntologyGenerator(llm=None)
        ont = gen.generate_from_text("Some text about cats and dogs")
        assert ont is not None

    def test_node_type_registry(self):
        from prometheus_v8.schema import NodeTypeRegistry
        NodeTypeRegistry.register("TEST_DYNAMIC", description="Dynamic test type")
        result = NodeTypeRegistry.get("TEST_DYNAMIC")
        assert result is not None


class TestTemporalSearch:
    """Test time-range filtering."""

    def test_get_nodes_by_time_range(self):
        import tempfile, os
        from prometheus_v8.core.store import SQLiteStore
        from prometheus_v8.schema import Node, NodeType, MemoryLayer, create_fact_node

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteStore(db_path)

        # Create nodes with different time ranges
        n1 = create_fact_node(content="Recent fact", importance=0.8)
        n1.valid_from = time.time() - 100
        n1.valid_to = time.time() + 10000
        store.add_node(n1)

        n2 = create_fact_node(content="Old fact", importance=0.5)
        n2.valid_from = time.time() - 100000
        n2.valid_to = time.time() - 50000
        store.add_node(n2)

        # Search for active nodes
        active = store.get_nodes_by_time_range(time.time() - 200, time.time() + 200)
        payloads = [n.payload.content for n in active if hasattr(n.payload, 'content')]
        assert "Recent fact" in payloads
        assert "Old fact" not in payloads

        store.close()
        for attempt in range(5):
            try:
                os.unlink(db_path)
                break
            except PermissionError:
                time.sleep(0.1)

    def test_hybrid_search_time_range(self):
        """HybridSearchEngine.search accepts time_range parameter."""
        from prometheus_v8.core.hybrid_search import HybridSearchEngine
        engine = HybridSearchEngine(store=None, vector_backend=None, llm=None)
        result = engine.search("test", k=5, time_range=(0.0, time.time()))
        assert isinstance(result, list)


class TestPersonaGenerator:
    """Test Agent persona generation."""

    def test_create_persona_profile(self):
        from prometheus_v8.schema import PersonaProfile
        pp = PersonaProfile()
        assert pp.mbti == ""
        assert pp.sentiment_bias == 0.0
        assert pp.stance == "neutral"

    def test_persona_with_values(self):
        from prometheus_v8.schema import PersonaProfile
        pp = PersonaProfile(
            mbti="INTJ",
            sentiment_bias=-0.3,
            stance="observer",
            influence_weight=2.5,
            interested_topics=["AI", "memory"],
        )
        assert pp.mbti == "INTJ"
        assert pp.sentiment_bias == -0.3
        assert len(pp.interested_topics) == 2

    def test_persona_generator_without_llm(self):
        from prometheus_v8.lifecycle.persona import PersonaGenerator
        from prometheus_v8.schema import create_fact_node
        gen = PersonaGenerator(llm=None)
        node = create_fact_node(content="Test agent", importance=0.5)
        profile = gen.generate_from_node(node)
        assert profile is not None


class TestEventWriter:
    """Test event-driven memory writing."""

    def test_memory_event_bus_publish_subscribe(self):
        from prometheus_v8.core.event_writer import MemoryEventBus
        bus = MemoryEventBus()
        received = []
        bus.subscribe("test_event", lambda etype, data: received.append(data))
        bus.publish("test_event", {"key": "value"})
        assert len(received) >= 1
        assert received[0]["key"] == "value"

    def test_event_driven_writer_creation(self):
        from prometheus_v8.core.event_writer import EventDrivenWriter, MemoryEventBus
        bus = MemoryEventBus()
        writer = EventDrivenWriter(store=None, event_bus=bus)
        assert writer is not None


class TestRetry:
    """Test exponential backoff retry."""

    def test_with_retry_decorator(self):
        from prometheus_v8.core.retry import with_retry

        @with_retry(max_retries=3, base_delay=0.01)
        def success_func():
            return 42

        result = success_func()
        assert result == 42

    def test_exponential_backoff(self):
        from prometheus_v8.core.retry import ExponentialBackoff
        eb = ExponentialBackoff(base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert eb.next_delay() == 1.0
        assert eb.next_delay() == 2.0
        assert eb.next_delay() == 4.0
