from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from mcp import types
from textual.widgets import Button, OptionList, RichLog, Select, Static

from main import MCPBrowser
from mcp_client import Discovery, DiscoveryError
from servers import SERVERS


def discovery(*tools):
    return Discovery("Test MCP", "1.0", "2025-11-25", tools)


def details_text(app):
    return "\n".join(line.text for line in app.query_one("#details", RichLog).lines)


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_browse_and_switch_server(self):
        tools = (
            types.Tool(name="search", description="Описание поиска", input_schema={"type": "object"}),
            types.Tool(name="fetch", description="Описание загрузки", input_schema={
                "type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"],
            }),
        )
        fetch = AsyncMock(return_value=discovery(*tools))
        with patch("main.discover_tools", fetch):
            app = MCPBrowser()
            async with app.run_test(size=(110, 32)) as pilot:
                fetch.assert_not_called()
                await pilot.click("#fetch")
                await app.workers.wait_for_complete()
                await pilot.pause()
                fetch.assert_awaited_once_with(SERVERS[0])
                self.assertEqual(app.query_one("#tools", OptionList).option_count, 2)
                self.assertIn("Описание поиска", details_text(app))
                self.assertIn("Объём списка:", str(app.query_one("#metrics", Static).render()))
                self.assertIn("≈", str(app.query_one("#metrics", Static).render()))
                self.assertIn("Объём инструмента:", details_text(app))
                await pilot.press("down")
                await pilot.pause()
                self.assertIn("Описание загрузки", details_text(app))
                self.assertIn('"required"', details_text(app))
                self.assertIn('"url"', details_text(app))
                app.query_one("#server", Select).value = 1
                await pilot.pause()
                self.assertIsNone(app.result)
                self.assertEqual(app.query_one("#tools", OptionList).option_count, 0)
                self.assertNotIn("Описание загрузки", details_text(app))
                self.assertIn("появится после запроса", str(app.query_one("#metrics", Static).render()))
                self.assertIn(SERVERS[1].url, str(app.query_one("#url", Static).render()))
                await pilot.click("#fetch")
                await app.workers.wait_for_complete()
                fetch.assert_awaited_with(SERVERS[1])

    async def test_busy_state_is_responsive_and_blocks_duplicate_requests(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def delayed(server):
            started.set()
            await finish.wait()
            return discovery()

        with patch("main.discover_tools", side_effect=delayed) as fetch:
            app = MCPBrowser()
            async with app.run_test(size=(100, 28)) as pilot:
                await pilot.click("#fetch")
                await asyncio.wait_for(started.wait(), 2)
                self.assertTrue(app.busy)
                self.assertTrue(app.query_one("#server", Select).disabled)
                self.assertTrue(app.query_one("#fetch", Button).disabled)
                self.assertTrue(app.query_one("#loading").display)
                app.fetch_pressed()
                await pilot.press("tab")
                self.assertEqual(fetch.await_count, 1)
                finish.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(app.busy)
                self.assertFalse(app.query_one("#server", Select).disabled)
                self.assertFalse(app.query_one("#loading").display)
                self.assertIn("пустой список", details_text(app))

    async def test_error_clears_previous_result_and_allows_retry(self):
        tool = types.Tool(name="[red]literal[/red]", input_schema={"type": "object"})
        with patch("main.discover_tools", new=AsyncMock(side_effect=[
            discovery(tool), DiscoveryError("Сеть недоступна"), discovery(tool),
        ])):
            app = MCPBrowser()
            async with app.run_test(size=(100, 28)) as pilot:
                for attempt in range(3):
                    await pilot.click("#fetch")
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    self.assertFalse(app.query_one("#fetch", Button).disabled)
                    if attempt == 1:
                        self.assertIn("Сеть недоступна", str(app.query_one("#status", Static).render()))
                        self.assertIsNone(app.result)
                        self.assertEqual(app.query_one("#tools", OptionList).option_count, 0)
                    else:
                        self.assertIn("[red]literal[/red]", details_text(app))
                        self.assertIn("Описание отсутствует", details_text(app))


if __name__ == "__main__":
    unittest.main()
