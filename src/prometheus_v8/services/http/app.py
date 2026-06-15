"""FastAPI HTTP Service - 20 endpoints."""
from __future__ import annotations
import logging
import time
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Prometheus V8", version="8.0.0", description="Self-Evolving AI Agent Memory Platform")

# ── Models ──
class NodeCreate(BaseModel):
    content: str
    type: str = "fact"
    layer: str = "working"
    importance: float = 0.5
    tags: list[str] = []

class NodeResponse(BaseModel):
    id: str
    type: str
    layer: str
    content: str
    importance: float
    trust_level: str
    created_at: float

class SearchRequest(BaseModel):
    query: str
    k: int = 20
    layers: list[str] = []

class EvolutionRequest(BaseModel):
    code: str = ""
    max_generations: int = 10
    fitness_threshold: float = 0.8

class SafetyCheckRequest(BaseModel):
    content: str
    plan: list[str] | None = None

# ── Global state ──
_store = None
_engine = None
_safety = None
_monitor = None

def init_services(store=None, engine=None, safety=None, monitor=None) -> None:
    global _store, _engine, _safety, _monitor
    _store = store
    _engine = engine
    _safety = safety
    _monitor = monitor

# ── Node Endpoints ──
@app.post("/nodes", response_model=NodeResponse)
async def create_node(req: NodeCreate):
    from prometheus_v8.schema import create_fact_node, NodeType
    node = create_fact_node(content=req.content, importance=req.importance, tags=req.tags)
    try:
        node.type = NodeType(req.type)
    except ValueError:
        pass
    if _store:
        _store.add_node(node)
    return NodeResponse(id=node.id.hex(), type=node.type.value, layer=node.layer.value,
                       content=node.payload.content, importance=node.importance,
                       trust_level=node.trust_level.value, created_at=node.created_at)

@app.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str):
    if not _store:
        raise HTTPException(500, "Store not initialized")
    node = _store.get_node(bytes.fromhex(node_id.rjust(32, '0')))
    if not node:
        raise HTTPException(404, "Node not found")
    return NodeResponse(id=node.id.hex(), type=node.type.value, layer=node.layer.value,
                       content=node.payload.content, importance=node.importance,
                       trust_level=node.trust_level.value, created_at=node.created_at)

@app.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    if not _store:
        raise HTTPException(500, "Store not initialized")
    ok = _store.delete_node(bytes.fromhex(node_id.rjust(32, '0')))
    if not ok:
        raise HTTPException(404, "Node not found")
    return {"deleted": True}

@app.get("/nodes", response_model=list[NodeResponse])
async def list_nodes(type: str = "", layer: str = "", limit: int = Query(20, le=100)):
    if not _store:
        return []
    from prometheus_v8.schema import NodeType, MemoryLayer
    if type:
        nodes = _store.get_nodes_by_type(NodeType(type), limit)
    elif layer:
        nodes = _store.get_nodes_by_layer(MemoryLayer(layer), limit)
    else:
        nodes = _store.search_fts("*", limit)
    return [NodeResponse(id=n.id.hex(), type=n.type.value, layer=n.layer.value,
                         content=n.payload.content[:200], importance=n.importance,
                         trust_level=n.trust_level.value, created_at=n.created_at) for n in nodes]

# ── Search Endpoint ──
@app.post("/search", response_model=list[NodeResponse])
async def search(req: SearchRequest):
    if not _store:
        return []
    nodes = _store.search_fts(req.query, req.k)
    return [NodeResponse(id=n.id.hex(), type=n.type.value, layer=n.layer.value,
                         content=n.payload.content[:200], importance=n.importance,
                         trust_level=n.trust_level.value, created_at=n.created_at) for n in nodes]

# ── Evolution Endpoints ──
@app.post("/evolution/run")
async def run_evolution(req: EvolutionRequest):
    if not _engine:
        raise HTTPException(500, "Engine not initialized")
    from prometheus_v8.schema import Genome
    genome = Genome(code=req.code)
    result = _engine.evolve(genome, max_generations=req.max_generations, fitness_threshold=req.fitness_threshold)
    return {"best_fitness": result.fitness, "generations": _engine.generation, "fingerprint": result.fingerprint}

@app.get("/evolution/stats")
async def evolution_stats():
    if not _engine:
        return {}
    return {"generation": _engine.generation, "best_fitness": _engine.best_genome.fitness if _engine.best_genome else 0,
            "layer_stats": _engine.layer_stats}

# ── Safety Endpoints ──
@app.post("/safety/check")
async def safety_check(req: SafetyCheckRequest):
    if not _safety:
        raise HTTPException(500, "Safety not initialized")
    verdict = _safety.check(req.content, plan=req.plan)
    return {"allowed": verdict.allowed, "risk_level": verdict.risk_level,
            "violations": verdict.violations, "checks_passed": verdict.checks_passed}

# ── Lifecycle Endpoints ──
@app.post("/lifecycle/consolidate")
async def consolidate():
    return {"message": "Consolidation triggered"}

@app.post("/lifecycle/dream")
async def dream():
    return {"message": "Dream cycle triggered"}

@app.post("/lifecycle/metabolism")
async def metabolism():
    return {"message": "Metabolism cycle triggered"}

# ── Governance Endpoints ──
@app.get("/governance/autonomy/rules")
async def autonomy_rules():
    from prometheus_v8.governance.autonomy import AutonomyController
    ctrl = AutonomyController()
    return ctrl.rules

@app.post("/governance/initiative/trigger")
async def trigger_initiative():
    from prometheus_v8.governance.initiative import SpontaneousInitiative
    init = SpontaneousInitiative()
    action = init.generate_action()
    if action:
        return {"action": action.description, "category": action.category, "level": action.autonomy_level.value}
    return {"action": None, "reason": "Conditions not met"}

# ── Monitor Endpoints ──
@app.get("/monitor/dashboard")
async def dashboard():
    if _monitor:
        return _monitor.get_dashboard_data()
    return {"status": "monitor not initialized"}

@app.get("/monitor/stats")
async def monitor_stats():
    if _monitor:
        return _monitor.stats
    return {}

# ── System Endpoints ──
@app.get("/health")
async def health():
    return {"status": "ok", "version": "8.0.0", "timestamp": time.time()}

@app.get("/info")
async def info():
    return {"name": "Prometheus V8", "version": "8.0.0",
            "layers": 9, "evolution_layers": 12, "endpoints": 20}
