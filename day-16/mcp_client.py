"""Discover tools through a real MCP handshake and tools/list requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from servers import Server


class DiscoveryError(Exception):
    """An actionable connection or discovery error for the UI."""


@dataclass(frozen=True)
class Discovery:
    server_name: str
    server_version: str
    protocol_version: str
    tools: tuple[types.Tool, ...]


def error_detail(error: Exception) -> str:
    """Expose the underlying cause of SDK/AnyIO exception groups."""
    if isinstance(error, ExceptionGroup):
        return "; ".join(dict.fromkeys(error_detail(item) for item in error.exceptions))
    return str(error) or type(error).__name__


async def discover_tools(server: Server, *, timeout: float = 30) -> Discovery:
    """Open one session, read all pages, then close it. Never call tools."""
    try:
        # One deadline covers connecting, handshake, pagination and cleanup.
        async with asyncio.timeout(timeout):
            async with streamable_http_client(server.url) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timeout) as session:
                    initialized = await session.initialize()
                    if initialized.capabilities.tools is None:
                        raise DiscoveryError("Сервер не объявил поддержку tools.")
                    tools: list[types.Tool] = []
                    cursor: str | None = None
                    seen_cursors: set[str] = set()
                    while True:
                        params = types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
                        page = await session.list_tools(params=params)
                        tools.extend(page.tools)
                        cursor = page.next_cursor
                        if cursor is None:
                            break
                        if cursor in seen_cursors:
                            raise DiscoveryError("Сервер повторил курсор списка инструментов.")
                        seen_cursors.add(cursor)
                    result = Discovery(
                        server_name=initialized.server_info.name,
                        server_version=initialized.server_info.version,
                        protocol_version=initialized.protocol_version,
                        tools=tuple(tools),
                    )
        return result
    except TimeoutError as error:
        raise DiscoveryError(f"Сервер не завершил запрос за {timeout:g} секунд. Повторите запрос.") from error
    except DiscoveryError:
        raise
    except Exception as error:
        raise DiscoveryError(f"Не удалось получить инструменты: {error_detail(error)}") from error
