"""HTTP API Server - FastAPI-based REST API with 20 endpoints."""
from __future__ import annotations
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query, Body, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

def create_app(store=None, engine=None, safety=None, monitor=None, governance=None) -> Any:
    """Create the FastAPI application with all endpoints.
    
    Returns the FastAPI app if fastapi is available, otherwise a mock app.
    """
    if not HAS_FASTAPI:
        return _create_mock_app()
    
    app = FastAPI(
        title="Prometheus V8",
        description="Self-Evolving AI Agent Memory Platform",
        version="8.0.0",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store references
    app.state.store = store
    app.state.engine = engine
    app.state.safety = safety
    app.state.monitor = monitor
    app.state.governance = governance

    # ── Health & Info ──────────────────────────────────────────
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "8.0.0", "timestamp": time.time()}

    @app.get("/info")
    async def system_info():
        info = {"version": "8.0.0", "layers": 9, "evolution_layers": 12}
        if store:
            info["nodes"] = store.count_nodes()
        if engine:
            info["generation"] = engine.generation
        return info

    # ── Node CRUD ──────────────────────────────────────────────
    @app.post("/nodes")
    async def create_node(content: str = Body(..., embed=True),
                          node_type: str = "fact",
                          importance: float = 0.5,
                          tags: list[str] = Query(default=[])):
        if not store:
            raise HTTPException(503, "Store not available")
        from prometheus_v8.schema import create_fact_node, create_insight_node, create_episode_node
        creators = {"fact": create_fact_node, "insight": create_insight_node, "episode": create_episode_node}
        creator = creators.get(node_type, create_fact_node)
        node = creator(content=content, importance=importance, tags=tags)
        node_id = store.add_node(node)
        return {"id": node_id.hex(), "type": node_type, "content": content}

    @app.get("/nodes/{node_id}")
    async def get_node(node_id: str):
        if not store:
            raise HTTPException(503, "Store not available")
        nid = bytes.fromhex(node_id)
        node = store.get_node(nid)
        if not node:
            raise HTTPException(404, "Node not found")
        return {"id": node_id, "type": node.type.value, "content": node.payload.content,
                "importance": node.importance, "layer": node.layer.value,
                "trust_level": node.trust_level.value, "access_count": node.access_count}

    @app.delete("/nodes/{node_id}")
    async def delete_node(node_id: str):
        if not store:
            raise HTTPException(503, "Store not available")
        nid = bytes.fromhex(node_id)
        if store.delete_node(nid):
            return {"deleted": True}
        raise HTTPException(404, "Node not found")

    @app.get("/nodes")
    async def list_nodes(layer: str = Query(default=""),
                         node_type: str = Query(default=""),
                         limit: int = Query(default=20, le=100)):
        if not store:
            raise HTTPException(503, "Store not available")
        from prometheus_v8.schema import MemoryLayer, NodeType
        if layer:
            nodes = store.get_nodes_by_layer(MemoryLayer(layer), limit)
        elif node_type:
            nodes = store.get_nodes_by_type(NodeType(node_type), limit)
        else:
            nodes = store.search_fts("*", limit)
        return [{"id": n.id.hex(), "type": n.type.value, "content": n.payload.content[:100],
                 "importance": n.importance, "layer": n.layer.value} for n in nodes]

    # ── Search ─────────────────────────────────────────────────
    @app.get("/search")
    async def search_nodes(q: str = Query(...), limit: int = Query(default=10, le=50)):
        if not store:
            raise HTTPException(503, "Store not available")
        nodes = store.search_fts(q, limit)
        return [{"id": n.id.hex(), "content": n.payload.content[:200],
                 "importance": n.importance, "type": n.type.value} for n in nodes]

    # ── Graph ──────────────────────────────────────────────────
    @app.post("/edges")
    async def create_edge(source_id: str = Body(...), target_id: str = Body(...),
                          edge_type: str = Body(default="related"),
                          weight: float = Body(default=1.0)):
        if not store:
            raise HTTPException(503, "Store not available")
        from prometheus_v8.schema import Edge, EdgeType, generate_uuidv7
        edge = Edge(id=generate_uuidv7(), source_id=bytes.fromhex(source_id),
                    target_id=bytes.fromhex(target_id), type=EdgeType(edge_type), weight=weight)
        store.add_edge(edge)
        return {"id": edge.id.hex(), "type": edge_type}

    @app.get("/edges/{node_id}")
    async def get_edges(node_id: str, direction: str = Query(default="both")):
        if not store:
            raise HTTPException(503, "Store not available")
        nid = bytes.fromhex(node_id)
        edges = store.get_edges(nid)
        return [{"id": e.id.hex(), "source": e.source_id.hex(), "target": e.target_id.hex(),
                 "type": e.type.value, "weight": e.weight} for e in edges]

    # ── Evolution ──────────────────────────────────────────────
    @app.post("/evolution/evolve")
    async def evolve(code: str = Body(...), generations: int = Body(default=5),
                     fitness_threshold: float = Body(default=0.99)):
        if not engine:
            raise HTTPException(503, "Evolution engine not available")
        from prometheus_v8.schema import Genome
        genome = Genome(code=code, fitness=0.3)
        result = engine.evolve(genome, max_generations=generations, fitness_threshold=fitness_threshold)
        return {"generation": engine.generation, "best_fitness": result.fitness if result else 0}

    @app.get("/evolution/status")
    async def evolution_status():
        if not engine:
            return {"generation": 0, "best_fitness": 0}
        return {"generation": engine.generation, "best_fitness": engine.best_genome.fitness if engine.best_genome else 0}

    # ── Safety ─────────────────────────────────────────────────
    @app.post("/safety/check")
    async def safety_check(action: str = Body(..., embed=True)):
        if not safety:
            raise HTTPException(503, "Safety manager not available")
        verdict = safety.check(action)
        return {"allowed": verdict.allowed, "reason": verdict.reason, "risk_level": verdict.risk_level}

    @app.get("/safety/stats")
    async def safety_stats():
        if not safety:
            return {"checks": 0, "violations": 0}
        return safety.stats

    # ── Monitor ────────────────────────────────────────────────
    @app.get("/monitor/kpi")
    async def monitor_kpi():
        if not monitor:
            return {}
        kpi = monitor.get("kpi")
        if kpi:
            return kpi.export()
        return {}

    @app.get("/monitor/health")
    async def monitor_health():
        if not monitor:
            return {"system": "unknown"}
        heartbeat = monitor.get("heartbeat")
        if heartbeat:
            return {"system_status": heartbeat.get_system_status().value,
                    "components": heartbeat.stats}
        return {}

    # ── Governance ─────────────────────────────────────────────
    @app.get("/governance/autonomy")
    async def governance_autonomy():
        if not governance:
            return {"level": 0}
        ctrl = governance.get("autonomy")
        if ctrl:
            return ctrl.stats
        return {}

    @app.get("/governance/trust")
    async def governance_trust():
        if not governance:
            return {}
        tm = governance.get("trust")
        if tm:
            return tm.stats
        return {}

    @app.get("/governance/curiosity")
    async def governance_curiosity():
        if not governance:
            return {}
        cq = governance.get("curiosity")
        if cq:
            return cq.stats
        return {}

    # ── Dashboard ──────────────────────────────────────────────
    @app.get("/dashboard")
    async def dashboard():
        from prometheus_v8.visualization.dashboard import DashboardProvider
        dp = DashboardProvider(store=store, engine=engine, safety=safety, monitor=monitor)
        return dp.get_overview()

    return app

def _create_mock_app():
    """Create a mock app when FastAPI is not available."""
    class MockApp:
        def __init__(self):
            self.routes = []
            self.state = type("State", (), {"store": None, "engine": None, "safety": None})()
        
        def run(self, host: str = "0.0.0.0", port: int = 8082, **kwargs):
            logger.info(f"Mock app would run on {host}:{port}")
        
        def __call__(self, *args, **kwargs):
            return self
    
    return MockApp()
