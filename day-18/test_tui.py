from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import types
from textual.widgets import Button, OptionList, RichLog, Static

from main import MCPAgentApp
from mcp_client import Discovery


def log_text(app: MCPAgentApp, selector: str) -> str:
    return "\n".join(line.text for line in app.query_one(selector, RichLog).lines)


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_and_agent_answer_are_visible(self) -> None:
        tools = (types.Tool(name="ssh_recent_logins", description="Successful logins", input_schema={
            "type": "object", "properties": {"hours": {"type": "integer"}},
        }),)
        discovery = Discovery("day-18-ssh-logins", "1.0", "2025-11-25", tools)

        async def fake_turn(server, available, history, question, trace):
            self.assertEqual(available, tools)
            trace("LLM REQUEST 1", {"messages": history + [{"role": "user", "content": question}]})
            trace("LLM RESPONSE 1", {"choices": [{"message": {"tool_calls": ["ssh_recent_logins"]}}]})
            trace("MCP CALL", {"name": "ssh_recent_logins", "arguments": {"hours": 1}})
            trace("MCP RESULT", {"name": "ssh_recent_logins", "data": {"total": 1}})
            answer = "За час был один успешный вход."
            return answer, history + [{"role": "assistant", "content": answer}]

        with tempfile.TemporaryDirectory() as directory, patch(
            "main.discover_tools", new=AsyncMock(return_value=discovery)
        ), patch("main.run_turn", side_effect=fake_turn):
            app = MCPAgentApp(log_dir=Path(directory))
            async with app.run_test(size=(120, 40)) as pilot:
                self.assertTrue(app.query_one("#send", Button).disabled)
                await pilot.click("#fetch")
                await app.workers.wait_for_complete()
                self.assertEqual(app.query_one("#tools", OptionList).option_count, 1)
                self.assertIn("ssh_recent_logins", log_text(app, "#details"))
                self.assertFalse(app.query_one("#send", Button).disabled)
                app.query_one("#question").value = "Кто входил за час?"
                await pilot.click("#send")
                await app.workers.wait_for_complete()
                self.assertIn("один успешный вход", log_text(app, "#chat"))
                self.assertIn("MCP RESULT", log_text(app, "#trace"))
                self.assertIn(str(app.session_log.path), str(app.query_one("#log-path", Static).render()))
            self.assertEqual(app.session_log.path.stat().st_mode & 0o777, 0o600)
            events = [json.loads(line) for line in app.session_log.path.read_text(encoding="utf-8").splitlines()]
            self.assertIn("LLM REQUEST 1", {event["event"] for event in events})
            self.assertIn("LLM RESPONSE 1", {event["event"] for event in events})
            self.assertIn("MCP RESULT", {event["event"] for event in events})
            self.assertIn("ANSWER DISPLAYED", {event["event"] for event in events})


if __name__ == "__main__":
    unittest.main()
