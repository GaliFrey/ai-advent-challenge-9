from __future__ import annotations

import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp import types

from mcp_client import DiscoveryError, discover_tools
from servers import SERVERS


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.events = []
        self.session = SimpleNamespace(
            initialize=AsyncMock(return_value=SimpleNamespace(
                capabilities=SimpleNamespace(tools=types.ToolsCapability()),
                server_info=SimpleNamespace(name="Test server", version="1.0"),
                protocol_version="2025-11-25",
            )),
            list_tools=AsyncMock(),
            call_tool=AsyncMock(),
        )

        @asynccontextmanager
        async def transport(url):
            self.assertEqual(url, SERVERS[0].url)
            self.events.append("transport opened")
            try:
                yield object(), object()
            finally:
                self.events.append("transport closed")

        @asynccontextmanager
        async def session(*args, **kwargs):
            self.events.append("session opened")
            try:
                yield self.session
            finally:
                self.events.append("session closed")

        self.enterContext(patch("mcp_client.streamable_http_client", transport))
        self.enterContext(patch("mcp_client.ClientSession", session))

    def assert_closed(self):
        self.assertEqual(self.events[-2:], ["session closed", "transport closed"])
        self.session.call_tool.assert_not_called()

    async def test_handshake_all_pages_and_cleanup(self):
        first = types.Tool(name="search", description="Поиск", input_schema={"type": "object"})
        second = types.Tool(name="fetch", input_schema={"type": "object"})
        self.session.list_tools.side_effect = [
            types.ListToolsResult(tools=[first], next_cursor="page-2"),
            types.ListToolsResult(tools=[second]),
        ]
        result = await discover_tools(SERVERS[0])
        self.assertEqual(result.tools, (first, second))
        self.assertEqual(result.server_name, "Test server")
        self.session.initialize.assert_awaited_once()
        calls = self.session.list_tools.await_args_list
        self.assertIsNone(calls[0].kwargs["params"])
        self.assertEqual(calls[1].kwargs["params"].cursor, "page-2")
        self.assert_closed()

    async def test_empty_list_is_success(self):
        self.session.list_tools.return_value = types.ListToolsResult(tools=[])
        self.assertEqual((await discover_tools(SERVERS[0])).tools, ())
        self.assert_closed()

    async def test_missing_tools_capability(self):
        self.session.initialize.return_value.capabilities.tools = None
        with self.assertRaisesRegex(DiscoveryError, "не объявил поддержку tools"):
            await discover_tools(SERVERS[0])
        self.session.list_tools.assert_not_called()
        self.assert_closed()

    async def test_repeated_cursor_stops_pagination(self):
        self.session.list_tools.return_value = types.ListToolsResult(tools=[], next_cursor="again")
        with self.assertRaisesRegex(DiscoveryError, "повторил курсор"):
            await discover_tools(SERVERS[0])
        self.assertEqual(self.session.list_tools.await_count, 2)
        self.assert_closed()

    async def test_timeout_closes_connection(self):
        async def hang(**kwargs):
            await asyncio.Event().wait()
        self.session.list_tools.side_effect = hang
        with self.assertRaisesRegex(DiscoveryError, "не завершил запрос"):
            await discover_tools(SERVERS[0], timeout=0.01)
        self.assert_closed()

    async def test_nested_transport_error_is_readable(self):
        self.session.initialize.side_effect = ExceptionGroup(
            "task group", [ExceptionGroup("nested", [OSError("Connection refused")])]
        )
        with self.assertRaisesRegex(DiscoveryError, "Connection refused"):
            await discover_tools(SERVERS[0])
        self.assert_closed()

    async def test_cancellation_propagates_and_closes_connection(self):
        self.session.list_tools.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await discover_tools(SERVERS[0])
        self.assert_closed()


if __name__ == "__main__":
    unittest.main()
