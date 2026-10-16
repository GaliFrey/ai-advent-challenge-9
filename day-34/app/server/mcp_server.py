"""Read-only MCP-сервер контекста обращения (официальный Python SDK, stdio).

Запускается backend-ом дочерним процессом тем же интерпретатором uv-окружения.
Транспорт — stdio; ``stdout`` используется только для MCP-протокола. Область
доступа (путь к БД, обращение, сообщение) задаётся исключительно доверенными
параметрами запуска и не может быть расширена аргументами инструмента: у
единственного инструмента ``get_ticket_context`` нет параметров. Cookie и токен
браузерной сессии дочернему процессу не передаются.

БД открывается в режиме ``mode=ro`` и не копируется ради каждого запроса.
Отсутствие сообщения или снимка в разрешённой области — явная ошибка инструмента
(``ToolError`` → ``isError``), а не пустой или частичный успех.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import config
from .db import open_readonly

TOOL_NAME = "get_ticket_context"
CONTEXT_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", required=True, dest="db_path")
    parser.add_argument("--conversation-id", type=int, required=True)
    parser.add_argument("--message-id", type=int, required=True)
    parser.add_argument("--max-messages", type=int, default=config.MCP_MAX_MESSAGES)
    parser.add_argument("--max-text-chars", type=int, default=config.MCP_MAX_TEXT_CHARS)
    return parser.parse_args(argv)


def _build_history(
    rows: list[Any],
    current_message_id: int,
    max_messages: int,
    max_text_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    """Собирает ограниченный хвост истории в порядке возрастания времени.

    ``assistant`` включается только в состоянии ``answered``. ``user``
    включается в любом состоянии со своим ``state``/``errorCode``. Лимит
    совместный: по числу сообщений и по суммарному объёму текста; отдельные
    сообщения не режутся посередине, поэтому при превышении объёма более старые
    сообщения целиком исключаются.
    """

    candidates = [
        row for row in rows if row["role"] == "user" or row["state"] == "answered"
    ]

    selected: list[Any] = []
    total_chars = 0
    for row in reversed(candidates):
        if len(selected) >= max_messages:
            break
        text = row["text"] or ""
        if total_chars + len(text) > max_text_chars:
            break
        selected.append(row)
        total_chars += len(text)

    selected.reverse()
    omitted = len(candidates) - len(selected)
    history = [
        {
            "id": row["id"],
            "role": row["role"],
            "text": row["text"],
            "state": row["state"],
            "createdAt": row["created_at"],
            "errorCode": row["error_code"],
        }
        for row in selected
    ]
    return history, omitted


def load_context(
    *,
    db_path: str,
    conversation_id: int,
    message_id: int,
    max_messages: int,
    max_text_chars: int,
) -> dict[str, Any]:
    """Читает контекст разрешённого обращения из БД только для чтения."""

    conn = open_readonly(db_path)
    try:
        message = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND conversation_id = ?",
            (message_id, conversation_id),
        ).fetchone()
        if message is None or message["role"] != "user":
            raise ToolError("Обрабатываемое сообщение не найдено в разрешённой области.")

        snapshot_row = conn.execute(
            "SELECT payload, created_at FROM snapshots WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if snapshot_row is None:
            raise ToolError("Снимок обрабатываемого сообщения не найден.")
        try:
            snapshot = json.loads(snapshot_row["payload"])
        except (TypeError, ValueError):
            raise ToolError("Снимок обрабатываемого сообщения повреждён.")

        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? AND id <= ? "
            "ORDER BY created_at ASC, id ASC",
            (conversation_id, message_id),
        ).fetchall()
        history, omitted = _build_history(
            rows, message_id, max_messages, max_text_chars
        )
        if not any(item["id"] == message_id for item in history):
            # Текущий вопрос обязан входить целиком; молчаливое усечение запрещено.
            raise ToolError("Текущий вопрос не помещается в лимиты истории.")

        return {
            "contextVersion": CONTEXT_VERSION,
            "conversationId": conversation_id,
            "messageId": message_id,
            "snapshotSchemaVersion": SNAPSHOT_SCHEMA_VERSION,
            "snapshot": snapshot,
            "snapshotReceivedAt": snapshot_row["created_at"],
            "history": history,
            "historyIncludedCount": len(history),
            "historyTruncated": omitted > 0,
            "historyOmittedCount": omitted,
            "limits": {"maxMessages": max_messages, "maxTextChars": max_text_chars},
        }
    finally:
        conn.close()


def create_server(
    *,
    db_path: str,
    conversation_id: int,
    message_id: int,
    max_messages: int,
    max_text_chars: int,
) -> MCPServer:
    """Собирает MCP-сервер с единственным read-only инструментом."""

    server = MCPServer(name="guitar-trainer-context", version="0.1.0")

    @server.tool(
        name=TOOL_NAME,
        description=(
            "Read-only контекст разрешённого обращения: текущий вопрос, "
            "ограниченная история и снимок именно этого сообщения."
        ),
    )
    def get_ticket_context() -> dict[str, Any]:
        return load_context(
            db_path=db_path,
            conversation_id=conversation_id,
            message_id=message_id,
            max_messages=max_messages,
            max_text_chars=max_text_chars,
        )

    return server


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    server = create_server(
        db_path=args.db_path,
        conversation_id=args.conversation_id,
        message_id=args.message_id,
        max_messages=args.max_messages,
        max_text_chars=args.max_text_chars,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
