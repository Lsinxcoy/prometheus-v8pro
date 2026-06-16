"""Services module - CLI, HTTP API, and MCP server."""

from .mcp.server import PrometheusMCPServer as PrometheusMCPServer

__all__ = [
    "PrometheusMCPServer",
]
