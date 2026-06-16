"""Tests for services: CLI, MCP server, HTTP app."""

import json
import pytest

from prometheus_v8.services.cli.main import create_cli, HAS_TYPER
from prometheus_v8.services.mcp.server import PrometheusMCPServer, HAS_MCP
from prometheus_v8.services.http.app import create_app, HAS_FASTAPI


# ── CLI Tests ─────────────────────────────────────────────────


class TestCLI:
    """Tests for CLI creation and command availability."""

    def test_create_cli(self):
        cli = create_cli()
        assert cli is not None

    def test_cli_has_commands(self):
        """If typer is available, check registered commands."""
        if not HAS_TYPER:
            pytest.skip("typer not installed")

        # Typer app should have registered commands
        app = create_cli()
        # The typer app has a registered_commands or click app
        assert hasattr(app, "typer") or hasattr(app, "info") or callable(app)


# ── MCP Server Tests ──────────────────────────────────────────


class TestMCPServer:
    """Tests for MCP tool definitions and handlers."""

    def test_tool_definitions_count(self):
        server = PrometheusMCPServer()
        tools = server.get_tool_definitions()
        assert len(tools) == 16

    def test_tool_definitions_structure(self):
        server = PrometheusMCPServer()
        tools = server.get_tool_definitions()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_tool_names(self):
        server = PrometheusMCPServer()
        tools = server.get_tool_definitions()
        names = [t["name"] for t in tools]
        assert "add_node" in names
        assert "search_nodes" in names
        assert "evolve" in names
        assert "safety_check" in names
        assert "get_dashboard" in names

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self):
        server = PrometheusMCPServer()
        result = await server.handle_tool_call("nonexistent_tool", {})
        assert "Unknown tool" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_handle_add_node_no_store(self):
        server = PrometheusMCPServer()
        result = await server.handle_tool_call("add_node", {"content": "test"})
        assert "Error" in result or "not available" in result

    @pytest.mark.asyncio
    async def test_handle_search_no_store(self):
        server = PrometheusMCPServer()
        result = await server.handle_tool_call("search_nodes", {"query": "test"})
        assert "Error" in result or "not available" in result

    @pytest.mark.asyncio
    async def test_handle_evolution_status_no_engine(self):
        server = PrometheusMCPServer()
        result = await server.handle_tool_call("evolution_status", {})
        data = json.loads(result)
        assert data["generation"] == 0

    def test_server_stats(self):
        server = PrometheusMCPServer()
        stats = server.stats
        assert stats["tools"] == 16
        assert stats["tool_calls"] == 0


# ── HTTP App Tests ────────────────────────────────────────────


class TestHTTPApp:
    """Tests for FastAPI app creation and structure."""

    def test_create_app(self):
        app = create_app()
        assert app is not None

    def test_app_has_state(self):
        """If FastAPI available, check state; if mock, check mock structure."""
        app = create_app()
        if HAS_FASTAPI:
            assert hasattr(app, "state")
        else:
            # MockApp should have routes
            assert hasattr(app, "routes") or hasattr(app, "state")

    def test_mock_app_run(self):
        if HAS_FASTAPI:
            pytest.skip("FastAPI available, mock not used")
        app = create_app()
        # Should not raise
        app.run(host="localhost", port=9999)
