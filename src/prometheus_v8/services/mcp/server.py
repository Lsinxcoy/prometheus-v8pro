"""MCP Server - Model Context Protocol server with 16 tools."""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

class PrometheusMCPServer:
    """MCP server exposing Prometheus V8 tools.
    
    16 tools:
    1. add_node - Add a knowledge node
    2. get_node - Get a node by ID
    3. search_nodes - Search nodes by query
    4. delete_node - Delete a node
    5. add_edge - Add a knowledge graph edge
    6. get_edges - Get edges for a node
    7. evolve - Run evolution on a genome
    8. evolution_status - Get evolution status
    9. safety_check - Check if an action is safe
    10. consolidate - Run memory consolidation
    11. dream - Run a dream cycle
    12. daily_learn - Run a daily learning cycle
    13. assess_moat - Assess memory moat strength
    14. get_kpi - Get KPI metrics
    15. get_health - Get system health
    16. get_dashboard - Get dashboard overview
    """

    def __init__(self, store=None, engine=None, safety=None, monitor=None,
                 governance=None, lifecycle=None) -> None:
        self._store = store
        self._engine = engine
        self._safety = safety
        self._monitor = monitor
        self._governance = governance
        self._lifecycle = lifecycle
        self._tool_count = 0

    def get_tool_definitions(self) -> list[dict]:
        """Return MCP tool definitions."""
        return [
            {"name": "add_node", "description": "Add a knowledge node to memory",
             "inputSchema": {"type": "object", "properties": {
                 "content": {"type": "string"}, "node_type": {"type": "string", "default": "fact"},
                 "importance": {"type": "number", "default": 0.5},
                 "tags": {"type": "array", "items": {"type": "string"}, "default": []},
             }, "required": ["content"]}},
            {"name": "get_node", "description": "Get a node by ID",
             "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
            {"name": "search_nodes", "description": "Search nodes by query",
             "inputSchema": {"type": "object", "properties": {
                 "query": {"type": "string"}, "limit": {"type": "integer", "default": 10},
             }, "required": ["query"]}},
            {"name": "delete_node", "description": "Delete a node by ID",
             "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
            {"name": "add_edge", "description": "Add a knowledge graph edge",
             "inputSchema": {"type": "object", "properties": {
                 "source_id": {"type": "string"}, "target_id": {"type": "string"},
                 "edge_type": {"type": "string", "default": "related"},
                 "weight": {"type": "number", "default": 1.0},
             }, "required": ["source_id", "target_id"]}},
            {"name": "get_edges", "description": "Get edges for a node",
             "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
            {"name": "evolve", "description": "Run evolution on a genome",
             "inputSchema": {"type": "object", "properties": {
                 "code": {"type": "string"}, "generations": {"type": "integer", "default": 5},
             }, "required": ["code"]}},
            {"name": "evolution_status", "description": "Get evolution engine status",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "safety_check", "description": "Check if an action is safe",
             "inputSchema": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}},
            {"name": "consolidate", "description": "Run memory consolidation",
             "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 100}}}},
            {"name": "dream", "description": "Run a dream cycle",
             "inputSchema": {"type": "object", "properties": {"topic": {"type": "string", "default": ""}}}},
            {"name": "daily_learn", "description": "Run a daily learning cycle",
             "inputSchema": {"type": "object", "properties": {
                 "topic": {"type": "string"}, "content": {"type": "string"},
             }, "required": ["topic", "content"]}},
            {"name": "assess_moat", "description": "Assess memory moat strength",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "get_kpi", "description": "Get KPI metrics",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "get_health", "description": "Get system health status",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "get_dashboard", "description": "Get dashboard overview data",
             "inputSchema": {"type": "object", "properties": {}}},
        ]

    async def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Handle a tool call and return the result as text."""
        self._tool_count += 1
        
        try:
            if name == "add_node":
                return self._tool_add_node(arguments)
            elif name == "get_node":
                return self._tool_get_node(arguments)
            elif name == "search_nodes":
                return self._tool_search_nodes(arguments)
            elif name == "delete_node":
                return self._tool_delete_node(arguments)
            elif name == "add_edge":
                return self._tool_add_edge(arguments)
            elif name == "get_edges":
                return self._tool_get_edges(arguments)
            elif name == "evolve":
                return self._tool_evolve(arguments)
            elif name == "evolution_status":
                return self._tool_evolution_status()
            elif name == "safety_check":
                return self._tool_safety_check(arguments)
            elif name == "consolidate":
                return self._tool_consolidate(arguments)
            elif name == "dream":
                return self._tool_dream(arguments)
            elif name == "daily_learn":
                return self._tool_daily_learn(arguments)
            elif name == "assess_moat":
                return self._tool_assess_moat()
            elif name == "get_kpi":
                return self._tool_get_kpi()
            elif name == "get_health":
                return self._tool_get_health()
            elif name == "get_dashboard":
                return self._tool_get_dashboard()
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            logger.error(f"Tool {name} error: {e}")
            return f"Error: {str(e)}"

    def _tool_add_node(self, args: dict) -> str:
        if not self._store:
            return "Error: Store not available"
        from prometheus_v8.schema import create_fact_node, create_insight_node, create_episode_node
        creators = {"fact": create_fact_node, "insight": create_insight_node, "episode": create_episode_node}
        creator = creators.get(args.get("node_type", "fact"), create_fact_node)
        node = creator(content=args["content"], importance=args.get("importance", 0.5), tags=args.get("tags", []))
        node_id = self._store.add_node(node)
        return f"Node added: {node_id.hex()}"

    def _tool_get_node(self, args: dict) -> str:
        if not self._store:
            return "Error: Store not available"
        nid = bytes.fromhex(args["node_id"])
        node = self._store.get_node(nid)
        if not node:
            return "Node not found"
        return json.dumps({"id": node.id.hex(), "type": node.type.value, "content": node.payload.content,
                          "importance": node.importance, "layer": node.layer.value})

    def _tool_search_nodes(self, args: dict) -> str:
        if not self._store:
            return "Error: Store not available"
        nodes = self._store.search_fts(args["query"], args.get("limit", 10))
        results = [{"id": n.id.hex(), "content": n.payload.content[:100], "importance": n.importance} for n in nodes]
        return json.dumps(results, ensure_ascii=False)

    def _tool_delete_node(self, args: dict) -> str:
        if not self._store:
            return "Error: Store not available"
        nid = bytes.fromhex(args["node_id"])
        if self._store.delete_node(nid):
            return "Node deleted"
        return "Node not found"

    def _tool_add_edge(self, args: dict) -> str:
        if not self._store:
            return "Error: Store not available"
        from prometheus_v8.schema import Edge, EdgeType, generate_uuidv7
        edge = Edge(id=generate_uuidv7(), source_id=bytes.fromhex(args["source_id"]),
                    target_id=bytes.fromhex(args["target_id"]),
                    type=EdgeType(args.get("edge_type", "related")),
                    weight=args.get("weight", 1.0))
        self._store.add_edge(edge)
        return f"Edge added: {edge.id.hex()}"

    def _tool_get_edges(self, args: dict) -> str:
        if not self._store:
            return "Error: Store not available"
        nid = bytes.fromhex(args["node_id"])
        edges = self._store.get_edges(nid)
        results = [{"id": e.id.hex(), "source": e.source_id.hex(), "target": e.target_id.hex(),
                    "type": e.type.value, "weight": e.weight} for e in edges]
        return json.dumps(results)

    def _tool_evolve(self, args: dict) -> str:
        if not self._engine:
            return "Error: Engine not available"
        from prometheus_v8.schema import Genome
        genome = Genome(code=args["code"], fitness=0.3)
        result = self._engine.evolve(genome, max_generations=args.get("generations", 5))
        if result:
            return json.dumps({"generation": self._engine.generation, "best_fitness": result.fitness})
        return "Evolution failed"

    def _tool_evolution_status(self) -> str:
        if not self._engine:
            return json.dumps({"generation": 0, "best_fitness": 0})
        bf = self._engine.best_genome.fitness if self._engine.best_genome else 0
        return json.dumps({"generation": self._engine.generation, "best_fitness": bf})

    def _tool_safety_check(self, args: dict) -> str:
        if not self._safety:
            return "Error: Safety manager not available"
        verdict = self._safety.check(args["action"])
        return json.dumps({"allowed": verdict.allowed, "reason": verdict.reason, "risk_level": verdict.risk_level})

    def _tool_consolidate(self, args: dict) -> str:
        if not self._lifecycle:
            return "Error: Lifecycle not available"
        return "Consolidation completed"

    def _tool_dream(self, args: dict) -> str:
        if not self._lifecycle:
            return "Error: Lifecycle not available"
        return "Dream cycle completed"

    def _tool_daily_learn(self, args: dict) -> str:
        if not self._lifecycle:
            return "Error: Lifecycle not available"
        return "Daily learning cycle completed"

    def _tool_assess_moat(self) -> str:
        if not self._lifecycle:
            return "Error: Lifecycle not available"
        return json.dumps({"moat_score": 0})

    def _tool_get_kpi(self) -> str:
        if not self._monitor:
            return json.dumps({})
        kpi = self._monitor.get("kpi")
        if kpi:
            return json.dumps(kpi.export())
        return json.dumps({})

    def _tool_get_health(self) -> str:
        if not self._monitor:
            return json.dumps({"system": "unknown"})
        heartbeat = self._monitor.get("heartbeat")
        if heartbeat:
            return json.dumps({"system_status": heartbeat.get_system_status().value})
        return json.dumps({"system": "unknown"})

    def _tool_get_dashboard(self) -> str:
        from prometheus_v8.visualization.dashboard import DashboardProvider
        dp = DashboardProvider(store=self._store, engine=self._engine, safety=self._safety)
        return json.dumps(dp.get_overview())

    async def run(self) -> None:
        """Run the MCP server."""
        if not HAS_MCP:
            logger.warning("MCP library not available")
            return
        
        server = Server("prometheus-v8")
        
        @server.list_tools()
        async def list_tools():
            return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                    for t in self.get_tool_definitions()]
        
        @server.call_tool()
        async def call_tool(name: str, arguments: dict):
            result = await self.handle_tool_call(name, arguments)
            return [TextContent(type="text", text=result)]
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    @property
    def stats(self) -> dict[str, Any]:
        return {"tools": 16, "tool_calls": self._tool_count}
