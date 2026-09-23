"""Read-only MCP server for scheduled SSH login records."""

from __future__ import annotations

import sqlite3

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import login_store


server = MCPServer("day-18-ssh-logins", instructions="Read saved SSH login records. Returned times are Europe/Kirov (UTC+03:00). Never infer that a missing record proves no login occurred when collection is stale.")


@server.tool(name="ssh_recent_logins")
def ssh_recent_logins(hours: int = 1, limit: int = 50) -> dict:
    """List successful SSH authentications in the last 1–168 hours, newest first. Includes username, source IP and time. limit: 1–100."""
    try:
        return login_store.recent_logins(login_store.database_path(), hours, limit)
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ToolError(str(error)) from error


@server.tool(name="ssh_login_summary")
def ssh_login_summary(hours: int = 24) -> dict:
    """Count saved successful SSH authentications in the last 1–168 hours by user and Europe/Kirov hour."""
    try:
        return login_store.login_summary(login_store.database_path(), hours)
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ToolError(str(error)) from error


@server.tool(name="ssh_collector_status")
def ssh_collector_status() -> dict:
    """Show collection freshness and saved record coverage."""
    try:
        return login_store.collector_status(login_store.database_path())
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        raise ToolError(str(error)) from error


if __name__ == "__main__":
    server.run(transport="stdio")
