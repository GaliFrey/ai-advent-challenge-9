"""MCP servers available in the TUI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Server:
    name: str
    address: str
    local: bool = False


SERVERS = (
    Server("Datex · локальный", "day-17/server.py · STDIO", local=True),
    Server("Microsoft Learn", "https://learn.microsoft.com/api/mcp"),
    Server("DeepWiki", "https://mcp.deepwiki.com/mcp"),
    Server("Cloudflare Docs", "https://docs.mcp.cloudflare.com/mcp"),
)
