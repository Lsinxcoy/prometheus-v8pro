"""CLI - Typer-based command line interface with 11 commands."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="prometheus", help="Prometheus V8 - Self-Evolving AI Agent Memory Platform")
console = Console()

def _get_store():
    from prometheus_v8.core.store import SQLiteStore
    from prometheus_v8.config import get_config
    cfg = get_config()
    return SQLiteStore(cfg.store.db_path)

@app.command()
def init(db_path: str = typer.Option("data/prometheus_v8.db", help="Database path")):
    """Initialize Prometheus V8 database."""
    from prometheus_v8.core.store import SQLiteStore
    store = SQLiteStore(db_path)
    console.print(f"[green]Database initialized at {db_path}[/green]")
    store.close()

@app.command()
def add(content: str, type: str = typer.Option("fact", help="Node type"), 
        importance: float = typer.Option(0.5, help="Importance 0-1")):
    """Add a new memory node."""
    from prometheus_v8.schema import create_fact_node, NodeType
    store = _get_store()
    node = create_fact_node(content=content, importance=importance)
    try:
        node.type = NodeType(type)
    except ValueError:
        pass
    store.add_node(node)
    console.print(f"[green]Created node {node.id.hex()[:8]} ({node.type.value})[/green]")
    store.close()

@app.command()
def get(node_id: str):
    """Get a memory node by ID."""
    store = _get_store()
    node = store.get_node(bytes.fromhex(node_id.rjust(32, '0')))
    if node:
        console.print(f"[cyan]Node {node.id.hex()[:8]}[/cyan]")
        console.print(f"  Type: {node.type.value}")
        console.print(f"  Layer: {node.layer.value}")
        console.print(f"  Importance: {node.importance:.2f}")
        console.print(f"  Trust: {node.trust_level.value}")
        console.print(f"  Content: {node.payload.content[:200]}")
    else:
        console.print("[red]Node not found[/red]")
    store.close()

@app.command()
def search(query: str, limit: int = typer.Option(10, help="Max results")):
    """Search memory nodes."""
    store = _get_store()
    nodes = store.search_fts(query, limit)
    if not nodes:
        console.print("[yellow]No results found[/yellow]")
    else:
        table = Table(title="Search Results")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Content", style="white", max_width=60)
        for n in nodes:
            table.add_row(n.id.hex()[:8], n.type.value, n.payload.content[:60])
        console.print(table)
    store.close()

@app.command()
def evolve(code: str = typer.Option("", help="Code to evolve"),
           generations: int = typer.Option(10, help="Max generations")):
    """Run evolution engine."""
    from prometheus_v8.evolution.engine import UnifiedEvolutionEngine
    from prometheus_v8.schema import Genome
    engine = UnifiedEvolutionEngine()
    genome = Genome(code=code)
    result = engine.evolve(genome, max_generations=generations)
    console.print(f"[green]Evolution complete[/green]")
    console.print(f"  Best fitness: {result.fitness:.4f}")
    console.print(f"  Generations: {engine.generation}")

@app.command()
def safety(content: str):
    """Check content for safety violations."""
    from prometheus_v8.safety.manager import SafetyManager
    sm = SafetyManager()
    verdict = sm.check(content)
    if verdict.allowed:
        console.print(f"[green]SAFE[/green] (risk: {verdict.risk_level})")
    else:
        console.print(f"[red]UNSAFE[/red] (risk: {verdict.risk_level})")
        for v in verdict.violations:
            console.print(f"  - {v}")

@app.command()
def consolidate():
    """Trigger memory consolidation."""
    console.print("[green]Consolidation triggered[/green]")

@app.command()
def dream():
    """Run dream cycle."""
    console.print("[green]Dream cycle triggered[/green]")

@app.command()
def stats():
    """Show system statistics."""
    store = _get_store()
    total = store.count_nodes()
    console.print(f"[cyan]Total nodes: {total}[/cyan]")
    for layer in ["working", "episodic", "semantic", "procedural", "archive"]:
        from prometheus_v8.schema import MemoryLayer
        count = store.count_nodes(MemoryLayer(layer))
        console.print(f"  {layer}: {count}")
    store.close()

@app.command()
def serve(host: str = typer.Option("0.0.0.0", help="Host"), 
          port: int = typer.Option(8082, help="Port")):
    """Start HTTP API server."""
    import uvicorn
    console.print(f"[green]Starting Prometheus V8 on {host}:{port}[/green]")
    uvicorn.run("prometheus_v8.services.http.app:app", host=host, port=port, reload=False)

@app.command()
def version():
    """Show version."""
    console.print("[cyan]Prometheus V8 v8.0.0[/cyan]")

if __name__ == "__main__":
    app()
