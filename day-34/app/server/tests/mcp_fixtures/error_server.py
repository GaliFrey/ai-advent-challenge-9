"""Тестовый MCP-сервер, чей инструмент возвращает isError (C7)."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


def main() -> None:
    server = MCPServer(name="test-error")

    @server.tool(name="get_ticket_context")
    def get_ticket_context() -> dict[str, Any]:
        raise ToolError("намеренная ошибка инструмента")

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
