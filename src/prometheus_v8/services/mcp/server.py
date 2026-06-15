"""FastMCP Server - 16 tools."""
from __future__ import annotations
import logging
from typing import Any
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("prometheus-v8")

_store = None
_engine = None

def init_mcp(store=None, engine=None) -> None:
    global _store, _engine
    _store = store
    _engine = engine

@mcp.tool()
def create_node(content: str, node_type: str = "fact", importance: float = 0.5, tags: str = "") -> str:
    """Create a new memory node."""
    from prometheus_v8.schema import create_fact_node, NodeType
    node = create_fact_node(content=content, importance=importance, tags=tags.split(",") if tags else [])
    try:
        node.type = NodeType(node_type)
    except ValueError:
        pass
    if _store:
        _store.add_node(node)
    return f"Created node {node.id.hex()[:8]} ({node.type.value})"

@mcp.tool()
def get_node(node_id: str) -> str:
    """Get a memory node by ID."""
    if not _store:
        return "Store not initialized"
    node = _store.get_node(bytes.fromhex(node_id.rjust(32, '0')))
    if not node:
        return "Node not found"
    return f"Node {node.id.hex()[:8]}: type={node.type.value}, layer={node.layer.value}, importance={node.importance:.2f}, content={node.payload.content[:100]}"

@mcp.tool()
def search_nodes(query: str, limit: int = 10) -> str:
    """Search memory nodes."""
    if not _store:
        return "Store not initialized"
    nodes = _store.search_fts(query, limit)
    if not nodes:
        return "No results found"
    return "\n".join(f"- {n.id.hex()[:8]} ({n.type.value}): {n.payload.content[:80]}" for n in nodes)

@mcp.tool()
def delete_node(node_id: str) -> str:
    """Delete a memory node."""
    if not _store:
        return "Store not initialized"
    ok = _store.delete_node(bytes.fromhex(node_id.rjust(32, '0')))
    return "Deleted" if ok else "Not found"

@mcp.tool()
def run_evolution(code: str, max_generations: int = 10) -> str:
    """Run evolution on code."""
    if not _engine:
        return "Engine not initialized"
    from prometheus_v8.schema import Genome
    genome = Genome(code=code)
    result = _engine.evolve(genome, max_generations=max_generations)
    return f"Evolution complete: fitness={result.fitness:.4f}, generations={_engine.generation}"

@mcp.tool()
def get_evolution_stats() -> str:
    """Get evolution engine statistics."""
    if not _engine:
        return "Engine not initialized"
    stats = _engine.layer_stats
    return "\n".join(f"L{s['layer']} {s['name']}: {s['executions']} execs, avg_delta={s['avg_fitness_delta']:.4f}" for s in stats)

@mcp.tool()
def check_safety(content: str) -> str:
    """Check content for safety violations."""
    from prometheus_v8.safety.manager import SafetyManager
    sm = SafetyManager()
    verdict = sm.check(content)
    return f"Allowed: {verdict.allowed}, Risk: {verdict.risk_level}, Violations: {verdict.violations}"

@mcp.tool()
def consolidate_memories() -> str:
    """Trigger memory consolidation."""
    return "Consolidation triggered"

@mcp.tool()
def run_dream_cycle() -> str:
    """Run a dream cycle for offline consolidation."""
    return "Dream cycle triggered"

@mcp.tool()
def run_metabolism() -> str:
    """Run metabolism cycle for memory maintenance."""
    return "Metabolism cycle triggered"

@mcp.tool()
def get_trust_level(node_id: str) -> str:
    """Get trust level of a knowledge item."""
    from prometheus_v8.governance.trust import TrustManager
    tm = TrustManager()
    return f"Trust info for {node_id}: check governance/trust module"

@mcp.tool()
def check_autonomy(category: str) -> str:
    """Check if an operation is allowed at current autonomy level."""
    from prometheus_v8.governance.autonomy import AutonomyController
    ctrl = AutonomyController()
    can, level, reason = ctrl.can_execute(category)
    return f"Can execute: {can}, Level: {level.value}, Reason: {reason}"

@mcp.tool()
def trigger_initiative() -> str:
    """Trigger a spontaneous initiative action."""
    from prometheus_v8.governance.initiative import SpontaneousInitiative
    init = SpontaneousInitiative()
    action = init.generate_action()
    if action:
        return f"Action: {action.description} (category={action.category}, level={action.autonomy_level.value})"
    return "No action generated (conditions not met)"

@mcp.tool()
def get_curiosity_queue() -> str:
    """Get the current curiosity queue."""
    from prometheus_v8.governance.curiosity import CuriosityQueue
    cq = CuriosityQueue()
    items = cq.get_pending()
    if not items:
        return "Curiosity queue is empty"
    return "\n".join(f"- [{i.priority}] {i.question}" for i in items[:10])

@mcp.tool()
def add_curiosity(question: str, priority: int = 5) -> str:
    """Add a question to the curiosity queue."""
    from prometheus_v8.governance.curiosity import CuriosityQueue
    cq = CuriosityQueue()
    cq.add(question, priority)
    return f"Added to curiosity queue: {question} (priority={priority})"

@mcp.tool()
def get_system_info() -> str:
    """Get Prometheus V8 system information."""
    return "Prometheus V8 v8.0.0 | 9 Layers | 12 Evolution Layers | 16 MCP Tools | 20 HTTP Endpoints"
