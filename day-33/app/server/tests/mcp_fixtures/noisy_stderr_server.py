"""Тестовый MCP-сервер, пишущий диагностику в stderr (C9).

Диагностика в stderr не мешает MCP-протоколу: клиент должен успешно завершить
цикл. Это проверяет, что ``stdout`` отделён от диагностического канала.
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server import MCPServer

_CONTEXT: dict[str, Any] = {
    "contextVersion": 1,
    "conversationId": 1,
    "messageId": 1,
    "snapshotSchemaVersion": 1,
    "snapshot": {},
    "snapshotReceivedAt": "2026-10-14T12:00:00.000Z",
    "history": [],
    "historyIncludedCount": 0,
    "historyTruncated": False,
    "historyOmittedCount": 0,
    "limits": {"maxMessages": 20, "maxTextChars": 16000},
}


def main() -> None:
    sys.stderr.write("test noisy server starting\n")
    server = MCPServer(name="test-noisy")

    @server.tool(name="get_ticket_context")
    def get_ticket_context() -> dict[str, Any]:
        sys.stderr.write("test noisy server handling tool call\n")
        return _CONTEXT

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
