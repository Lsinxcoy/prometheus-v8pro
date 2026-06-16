"""CLI Interface - Typer-based command line with 11 commands."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import typer

    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False


def create_cli() -> Any:
    """Create the CLI application.

    11 commands:
    1. init - Initialize a new Prometheus database
    2. add - Add a knowledge node
    3. get - Get a node by ID
    4. search - Search nodes
    5. delete - Delete a node
    6. evolve - Run evolution
    7. safety-check - Check if an action is safe
    8. consolidate - Run memory consolidation
    9. dream - Run a dream cycle
    10. stats - Show system statistics
    11. serve - Start the HTTP server
    """
    if not HAS_TYPER:
        return _create_mock_cli()

    app = typer.Typer(
        name="prometheus",
        help="Prometheus V8 - Self-Evolving AI Agent Memory Platform",
        no_args_is_help=True,
    )

    @app.command()
    def init(db_path: str = typer.Option("data/prometheus_v8.db", help="Database path")):
        """Initialize a new Prometheus database."""
        from prometheus_v8.core.store import SQLiteStore

        store = SQLiteStore(db_path)
        count = store.count_nodes()
        store.close()
        typer.echo(f"Database initialized at {db_path} ({count} existing nodes)")

    @app.command()
    def add(
        content: str = typer.Argument(..., help="Node content"),
        node_type: str = typer.Option(
            "fact", help="Node type (fact/insight/episode/mutation/dream/curiosity/learning)"
        ),
        importance: float = typer.Option(0.5, help="Importance (0-1)"),
        tags: str = typer.Option("", help="Comma-separated tags"),
        db_path: str = typer.Option("data/prometheus_v8.db", help="Database path"),
    ):
        """Add a knowledge node to memory."""
        from prometheus_v8.core.store import SQLiteStore
        from prometheus_v8.schema import (
            create_curiosity_node,
            create_dream_node,
            create_episode_node,
            create_fact_node,
            create_insight_node,
            create_learning_node,
            create_mutation_node,
        )

        creators = {
            "fact": create_fact_node,
            "insight": create_insight_node,
            "episode": create_episode_node,
            "mutation": create_mutation_node,
            "dream": create_dream_node,
            "curiosity": create_curiosity_node,
            "learning": create_learning_node,
        }
        creator = creators.get(node_type, create_fact_node)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        node = creator(content=content, importance=importance, tags=tag_list)

        store = SQLiteStore(db_path)
        node_id = store.add_node(node)
        store.close()
        typer.echo(f"Added {node_type} node: {node_id.hex()}")

    @app.command()
    def get(
        node_id: str = typer.Argument(..., help="Node ID (hex)"),
        db_path: str = typer.Option("data/prometheus_v8.db", help="Database path"),
    ):
        """Get a node by ID."""
        from prometheus_v8.core.store import SQLiteStore

        store = SQLiteStore(db_path)
        nid = bytes.fromhex(node_id)
        node = store.get_node(nid)
        store.close()
        if not node:
            typer.echo("Node not found", err=True)
            raise typer.Exit(1)
        typer.echo(
            json.dumps(
                {
                    "id": node.id.hex(),
                    "type": node.type.value,
                    "content": node.payload.content,
                    "importance": node.importance,
                    "layer": node.layer.value,
                    "trust_level": node.trust_level.value,
                    "access_count": node.access_count,
                    "created_at": node.created_at,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    @app.command()
    def search(
        query: str = typer.Argument(..., help="Search query"),
        limit: int = typer.Option(10, help="Max results"),
        db_path: str = typer.Option("data/prometheus_v8.db", help="Database path"),
    ):
        """Search nodes by content."""
        from prometheus_v8.core.store import SQLiteStore

        store = SQLiteStore(db_path)
        nodes = store.search_fts(query, limit)
        store.close()
        if not nodes:
            typer.echo("No results found")
            return
        for node in nodes:
            typer.echo(f"  [{node.type.value}] {node.payload.content[:80]} (importance={node.importance:.2f})")

    @app.command()
    def delete(
        node_id: str = typer.Argument(..., help="Node ID (hex)"),
        db_path: str = typer.Option("data/prometheus_v8.db", help="Database path"),
    ):
        """Delete a node by ID."""
        from prometheus_v8.core.store import SQLiteStore

        store = SQLiteStore(db_path)
        nid = bytes.fromhex(node_id)
        if store.delete_node(nid):
            typer.echo(f"Deleted node {node_id}")
        else:
            typer.echo("Node not found", err=True)
        store.close()

    @app.command()
    def evolve(
        code: str = typer.Argument(..., help="Code to evolve"),
        generations: int = typer.Option(5, help="Max generations"),
        fitness_threshold: float = typer.Option(0.99, help="Fitness threshold"),
    ):
        """Run evolution on a code genome."""
        from prometheus_v8.evolution.engine import UnifiedEvolutionEngine
        from prometheus_v8.schema import Genome

        engine = UnifiedEvolutionEngine()
        genome = Genome(code=code, fitness=0.3)
        result = engine.evolve(genome, max_generations=generations, fitness_threshold=fitness_threshold)
        if result:
            typer.echo(f"Evolution complete: generation={engine.generation}, fitness={result.fitness:.4f}")
        else:
            typer.echo("Evolution failed")

    @app.command()
    def safety_check(action: str = typer.Argument(..., help="Action to check")):
        """Check if an action is safe."""
        from prometheus_v8.safety.manager import SafetyManager

        sm = SafetyManager()
        verdict = sm.check(action)
        if verdict.allowed:
            typer.echo(f"SAFE: {verdict.reason}")
        else:
            typer.echo(f"UNSAFE: {verdict.reason} (risk={verdict.risk_level})", err=True)

    @app.command()
    def consolidate(
        db_path: str = typer.Option("data/prometheus_v8.db", help="Database path"),
        limit: int = typer.Option(100, help="Max nodes to process"),
    ):
        """Run memory consolidation."""
        from prometheus_v8.core.store import SQLiteStore
        from prometheus_v8.lifecycle.consolidation import ConsolidationPipeline

        store = SQLiteStore(db_path)
        engine = ConsolidationPipeline(store=store)
        nodes = store.search_fts("", limit=limit) or []
        result = engine.consolidate(nodes)
        store.close()
        typer.echo(f"Consolidation complete: {len(result)} nodes processed")

    @app.command()
    def dream(
        topic: str = typer.Option("", help="Dream topic"),
        db_path: str = typer.Option("data/prometheus_v8.db", help="Database path"),
    ):
        """Run a dream cycle."""
        from prometheus_v8.core.store import SQLiteStore
        from prometheus_v8.lifecycle.dream import DreamCycle

        store = SQLiteStore(db_path)
        engine = DreamCycle(store=store)
        result = engine.dream()
        store.close()
        typer.echo(f"Dream cycle complete: {len(result)} insights generated")

    @app.command()
    def stats(db_path: str = typer.Option("data/prometheus_v8.db", help="Database path")):
        """Show system statistics."""
        from prometheus_v8.core.store import SQLiteStore

        store = SQLiteStore(db_path)
        from prometheus_v8.schema import MemoryLayer

        typer.echo(f"Total nodes: {store.count_nodes()}")
        for layer in MemoryLayer:
            count = store.count_nodes(layer)
            if count > 0:
                typer.echo(f"  {layer.value}: {count}")
        store.close()

    @app.command()
    def serve(
        host: str = typer.Option("0.0.0.0", help="Host"),
        port: int = typer.Option(8082, help="Port"),
        db_path: str = typer.Option("data/prometheus_v8.db", help="Database path"),
    ):
        """Start the HTTP server."""
        from prometheus_v8.core.store import SQLiteStore
        from prometheus_v8.services.http.app import create_app

        store = SQLiteStore(db_path)
        app = create_app(store=store)

        if hasattr(app, "run"):
            typer.echo(f"Starting Prometheus V8 server on {host}:{port}")
            import uvicorn

            uvicorn.run(app, host=host, port=port)
        else:
            typer.echo("FastAPI not available, cannot start server")

    return app


def _create_mock_cli():
    """Create a mock CLI when Typer is not available."""

    class MockCLI:
        def __call__(self, *args, **kwargs):
            print("Typer not available. Install with: pip install typer")

    return MockCLI()


# Create default CLI instance
if HAS_TYPER:
    cli = create_cli()
else:
    cli = _create_mock_cli()
