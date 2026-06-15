"""Schema tests."""
from prometheus_v8.schema import (
    Node, Edge, NodeType, EdgeType, MemoryLayer, MemoryScope,
    ProvenanceType, Veracity, TrustLevel, WeibullParams,
    create_fact_node, create_insight_node, create_episode_node,
    generate_uuidv7, compute_checksum, compute_weibull_retention,
)

def test_generate_uuidv7():
    uid = generate_uuidv7()
    assert len(uid) == 16
    assert isinstance(uid, bytes)

def test_compute_checksum():
    cs = compute_checksum("hello world")
    assert len(cs) == 16
    assert compute_checksum("hello world") == cs

def test_weibull_retention():
    r = compute_weibull_retention(0, 0.5, 7.0, 0.8)
    assert r == 0.5
    r_old = compute_weibull_retention(30, 0.5, 7.0, 0.8)
    assert r_old < 0.5

def test_node_creation():
    node = create_fact_node("Test fact", importance=0.8)
    assert node.type == NodeType.FACT
    assert node.layer == MemoryLayer.SEMANTIC
    assert node.importance == 0.8
    assert len(node.id) == 16

def test_node_touch():
    node = create_fact_node("Test")
    initial_access = node.access_count
    node.touch()
    assert node.access_count == initial_access + 1

def test_trust_levels():
    assert TrustLevel.PENDING.value == "pending"
    assert TrustLevel.HIGH_SIGNAL.value == "high_signal"
    assert TrustLevel.VERIFIED.value == "verified"

def test_weibull_params_for_layer():
    wp = WeibullParams.for_layer(MemoryLayer.WORKING)
    assert wp.lam == 1.0
    wp = WeibullParams.for_layer(MemoryLayer.ARCHIVE)
    assert wp.lam == 1095.0
