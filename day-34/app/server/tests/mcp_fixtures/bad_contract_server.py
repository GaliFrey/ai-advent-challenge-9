"""Тестовый MCP-сервер, возвращающий неполный контракт (C7)."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer


def main() -> None:
    server = MCPServer(name="test-bad-contract")

    @server.tool(name="get_ticket_context")
    def get_ticket_context() -> dict[str, Any]:
        return {"contextVersion": 999}

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
