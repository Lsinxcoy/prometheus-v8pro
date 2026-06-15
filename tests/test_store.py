"""Store tests."""
import pytest
from prometheus_v8.schema import create_fact_node, create_insight_node, NodeType, MemoryLayer

def test_add_and_get_node(tmp_db, sample_node):
    node_id = tmp_db.add_node(sample_node)
    retrieved = tmp_db.get_node(node_id)
    assert retrieved is not None
    assert retrieved.payload.content == "Test fact node"

def test_delete_node(tmp_db, sample_node):
    node_id = tmp_db.add_node(sample_node)
    assert tmp_db.delete_node(node_id)
    assert tmp_db.get_node(node_id) is None

def test_search_fts(tmp_db):
    n1 = create_fact_node("Python programming language", importance=0.7)
    n2 = create_fact_node("Machine learning algorithms", importance=0.8)
    tmp_db.add_node(n1)
    tmp_db.add_node(n2)
    results = tmp_db.search_fts("Python")
    assert len(results) >= 1

def test_count_nodes(tmp_db):
    n1 = create_fact_node("Fact 1")
    n2 = create_insight_node("Insight 1")
    tmp_db.add_node(n1)
    tmp_db.add_node(n2)
    assert tmp_db.count_nodes() >= 2
