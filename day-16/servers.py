"""Public MCP endpoints; no user configuration or API keys required."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Server:
    name: str
    url: str


SERVERS = (
    Server("Microsoft Learn", "https://learn.microsoft.com/api/mcp"),
    Server("DeepWiki", "https://mcp.deepwiki.com/mcp"),
    Server("Cloudflare Docs", "https://docs.mcp.cloudflare.com/mcp"),
)
