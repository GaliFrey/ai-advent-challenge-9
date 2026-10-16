"""Короткоживущий stdio-клиент MCP, который запускает backend.

Один процесс на обработку сообщения, без пулов и демонов. Клиент реально
выполняет цикл ``initialize`` → ``tools/list`` → ``tools/call`` с дочерним
MCP-сервером на официальном Python SDK, проверяет ожидаемый инструмент и
контракт ответа и всегда освобождает ресурсы: транспорт закрывает stdin и гасит
дерево процессов при успехе, ошибке, таймауте и отмене.

``stdout`` дочернего процесса несёт только MCP-протокол; диагностика идёт в
``stderr``. Запуск идёт без shell и командной интерполяции, с минимальным
окружением без облачных ключей.
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

from . import config

EXPECTED_TOOL = "get_ticket_context"
CONTEXT_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1

_REQUIRED_KEYS = frozenset(
    {
        "contextVersion",
        "conversationId",
        "messageId",
        "snapshotSchemaVersion",
        "snapshot",
        "snapshotReceivedAt",
        "history",
        "historyIncludedCount",
        "historyTruncated",
        "historyOmittedCount",
        "limits",
    }
)


class McpError(Exception):
    """Сбой получения контекста через MCP; публичный API отвечает ``mcp_error``."""


@dataclass
class McpContextResult:
    """Успешный результат цикла MCP и наблюдаемые им факты рукопожатия."""

    context: dict[str, Any]
    protocol_version: str
    tool_names: list[str]


def _validate_context(context: Any) -> dict[str, Any]:
    """Проверяет контракт результата; нарушение — явная ошибка, а не успех."""

    if not isinstance(context, dict):
        raise McpError("MCP-инструмент вернул не объект.")
    missing = _REQUIRED_KEYS - context.keys()
    if missing:
        raise McpError("MCP-инструмент вернул неполный контракт.")
    if context["contextVersion"] != CONTEXT_VERSION:
        raise McpError("Неизвестная версия контекста MCP.")
    if context["snapshotSchemaVersion"] != SNAPSHOT_SCHEMA_VERSION:
        raise McpError("Неизвестная версия снимка MCP.")
    if not isinstance(context["snapshot"], dict):
        raise McpError("Снимок MCP должен быть объектом.")
    if not isinstance(context["history"], list):
        raise McpError("История MCP должна быть массивом.")
    if not isinstance(context["limits"], dict):
        raise McpError("limits MCP должен быть объектом.")
    return context


def _extract_context(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            return json.loads(text)
        except ValueError:
            continue
    raise McpError("MCP-инструмент не вернул контекст.")


async def fetch_ticket_context_result(
    *,
    db_path: str | Path,
    conversation_id: int,
    message_id: int,
    server_module: str = "server.mcp_server",
    extra_env: dict[str, str] | None = None,
    max_messages: int = config.MCP_MAX_MESSAGES,
    max_text_chars: int = config.MCP_MAX_TEXT_CHARS,
    initialize_timeout: float = config.MCP_INITIALIZE_TIMEOUT_SECONDS,
    list_tools_timeout: float = config.MCP_LIST_TOOLS_TIMEOUT_SECONDS,
    call_tool_timeout: float = config.MCP_CALL_TOOL_TIMEOUT_SECONDS,
    total_timeout: float = config.MCP_TOTAL_TIMEOUT_SECONDS,
) -> McpContextResult:
    """Выполняет полный stdio-цикл MCP и возвращает проверенный контекст."""

    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-B",
            "-m",
            server_module,
            "--db",
            str(db_path),
            "--conversation-id",
            str(conversation_id),
            "--message-id",
            str(message_id),
            "--max-messages",
            str(max_messages),
            "--max-text-chars",
            str(max_text_chars),
        ],
        cwd=str(config.PROJECT_ROOT),
        env={key: value for key, value in (extra_env or {}).items() if not key.startswith('DEEPSEEK_')},
    )

    async def run_cycle() -> McpContextResult:
        # `Client.__aenter__` оставляет активный task group сессии, поэтому шаг
        # рукопожатия ограничивается клиентским `read_timeout_seconds`, а не
        # внешним `fail_after` (иначе cancel scope закрывается не в том порядке).
        async with AsyncExitStack() as stack:
            client = Client(
                params, mode="legacy", read_timeout_seconds=initialize_timeout
            )
            await stack.enter_async_context(client)
            protocol_version = client.protocol_version
            with anyio.fail_after(list_tools_timeout):
                listing = await client.list_tools()
            tool_names = [tool.name for tool in listing.tools]
            if tool_names != [EXPECTED_TOOL]:
                raise McpError("MCP-сервер предоставил не ожидаемый набор инструментов.")
            with anyio.fail_after(call_tool_timeout):
                result = await client.call_tool(EXPECTED_TOOL, {})
            if getattr(result, "is_error", False):
                raise McpError("MCP-инструмент вернул ошибку.")
            context = _validate_context(_extract_context(result))
            return McpContextResult(
                context=context,
                protocol_version=protocol_version,
                tool_names=tool_names,
            )

    try:
        with anyio.fail_after(total_timeout):
            return await run_cycle()
    except McpError:
        raise
    except TimeoutError as exc:
        raise McpError("Превышено время ожидания MCP.") from exc
    except Exception as exc:  # noqa: BLE001 - любой сбой MCP становится mcp_error
        raise McpError(f"Сбой MCP: {type(exc).__name__}") from exc


async def fetch_ticket_context(**kwargs: Any) -> dict[str, Any]:
    """Возвращает только проверенный контекст (без диагностики рукопожатия)."""

    result = await fetch_ticket_context_result(**kwargs)
    return result.context
