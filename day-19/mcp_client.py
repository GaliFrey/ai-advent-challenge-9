"""MCP client for the SSH-hosted day 19 server."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from mcp import Client, StdioServerParameters, types

from servers import Server


@dataclass(frozen=True)
class Discovery:
    server_name: str
    server_version: str
    protocol_version: str
    tools: tuple[types.Tool, ...]


def error_detail(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(dict.fromkeys(error_detail(item) for item in error.exceptions))
    return str(error) or type(error).__name__


@asynccontextmanager
async def connect(server: Server) -> AsyncIterator[Client]:
    agent_socket = os.environ.get("SSH_AUTH_SOCK")
    target = StdioServerParameters(
        command=server.command[0],
        args=list(server.command[1:]),
        env={"SSH_AUTH_SOCK": agent_socket} if agent_socket else None,
    )
    async with Client(target, mode="legacy") as client:
        yield client


async def discover_tools(server: Server, *, timeout: float = 30) -> Discovery:
    try:
        async with asyncio.timeout(timeout):
            async with connect(server) as client:
                if client.server_capabilities.tools is None:
                    raise RuntimeError("Сервер не объявил поддержку tools")
                tools: list[types.Tool] = []
                cursor: str | None = None
                seen: set[str] = set()
                while True:
                    page = await client.list_tools(cursor=cursor)
                    tools.extend(page.tools)
                    cursor = page.next_cursor
                    if cursor is None:
                        break
                    if cursor in seen:
                        raise RuntimeError("Сервер повторил курсор списка инструментов")
                    seen.add(cursor)
                info = client.server_info
                return Discovery(
                    server_name=info.name if info else server.name,
                    server_version=info.version if info else "?",
                    protocol_version=client.protocol_version,
                    tools=tuple(tools),
                )
    except TimeoutError as error:
        raise RuntimeError(f"MCP server did not respond within {timeout:g} seconds") from error
    except Exception as error:
        raise RuntimeError(f"MCP connection failed: {error_detail(error)}") from error
