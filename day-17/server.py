"""A local MCP server exposing the Datex documentation snapshot."""

from __future__ import annotations

import sqlite3

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import corpus


server = MCPServer(
    "day-17-datex-docs",
    instructions="Search and read a fixed Datex documentation snapshot. Results are source evidence, not runtime verification.",
)


@server.tool(name="datex_search")
def datex_search(query: str, limit: int = 3) -> dict:
    """Find Datex documentation by API name or phrase. limit: 1–3."""
    try:
        return corpus.search(query, limit)
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ToolError(str(error)) from error


@server.tool(name="datex_read")
def datex_read(document_id: str) -> dict:
    """Read a document found by datex_search, including its source URL."""
    try:
        return corpus.read(document_id)
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ToolError(str(error)) from error


@server.tool(name="datex_status")
def datex_status() -> dict:
    """Report snapshot ID, document count and known coverage gaps."""
    try:
        return corpus.status()
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ToolError(str(error)) from error


if __name__ == "__main__":
    server.run(transport="stdio")
