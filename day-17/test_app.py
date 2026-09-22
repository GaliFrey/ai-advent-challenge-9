from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import types
from textual.widgets import Button, OptionList, RichLog, Select, Static

from main import MCPAgentApp
from mcp_client import Discovery


def log_text(app: MCPAgentApp, name: str) -> str:
    return "\n".join(line.text for line in app.query_one(name, RichLog).lines)


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_chat_and_visible_trace(self) -> None:
        tools = (
            types.Tool(name="datex_search", description="Поиск", input_schema={
                "type": "object", "properties": {"query": {"type": "string"}},
            }),
            types.Tool(name="datex_read", description="Чтение", input_schema={"type": "object"}),
        )
        discovery = Discovery("Test MCP", "1.0", "2025-11-25", tools)
        histories = []

        async def fake_turn(server, available, history, question, trace):
            self.assertEqual(available, tools)
            histories.append(list(history))
            trace("LLM REQUEST 1", {"messages": history + [{"role": "user", "content": question}]})
            trace("LLM RESPONSE 1", {"message": {"tool_calls": [{"function": {"name": "datex_search"}}]}})
            trace("MCP CALL", {"name": "datex_search", "arguments": {"query": "ArraySort"}})
            trace("MCP RESULT", {"name": "datex_search", "data": {"results": ["ArraySort()"]}})
            answer = "ArraySort() найден в документации."
            return answer, history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

        with tempfile.TemporaryDirectory() as directory, patch("main.discover_tools", new=AsyncMock(return_value=discovery)), patch("main.run_turn", side_effect=fake_turn):
            app = MCPAgentApp(log_dir=Path(directory))
            async with app.run_test(size=(120, 40)) as pilot:
                self.assertIn(str(app.session_log.path), str(app.query_one("#log-path", Static).render()))
                self.assertTrue(app.query_one("#send", Button).disabled)
                await pilot.click("#fetch")
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(app.query_one("#tools", OptionList).option_count, 2)
                self.assertIn("Поиск", log_text(app, "#details"))
                self.assertIn("Объём списка:", str(app.query_one("#metrics", Static).render()))
                self.assertFalse(app.query_one("#send", Button).disabled)
                app.query_one("#question").value = "Что делает ArraySort?"
                await pilot.click("#send")
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertIn("ArraySort() найден", log_text(app, "#chat"))
                trace = log_text(app, "#trace")
                for item in ("LLM REQUEST 1", "LLM RESPONSE 1", "MCP CALL", "MCP RESULT", "ArraySort"):
                    self.assertIn(item, trace)
                app.query_one("#question").value = "Покажи пример"
                await pilot.click("#send")
                await app.workers.wait_for_complete()
                self.assertEqual(histories[1][-1]["content"], "ArraySort() найден в документации.")
                self.assertEqual(log_text(app, "#trace").count("MCP CALL"), 2)
                app.query_one("#server", Select).value = 1
                await pilot.pause()
                self.assertIsNone(app.result)
                self.assertEqual(app.query_one("#tools", OptionList).option_count, 0)
                self.assertTrue(app.query_one("#send", Button).disabled)
                self.assertNotIn("MCP CALL", log_text(app, "#trace"))
            events = [json.loads(line) for line in app.session_log.path.read_text(encoding="utf-8").splitlines()]
            labels = [item["event"] for item in events]
            self.assertEqual(labels.count("MCP CALL"), 2)
            self.assertEqual(labels.count("MCP RESULT"), 2)
            self.assertIn("SERVER SELECTED", labels)
            self.assertEqual(events[-1]["server"], "Microsoft Learn")
            self.assertEqual(app.session_log.path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
