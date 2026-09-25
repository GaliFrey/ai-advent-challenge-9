from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import types
from textual.widgets import Button, OptionList, RichLog

import agent
import report_download
from main import MCPAgentApp
from mcp_client import Discovery
from servers import SERVERS


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_routes_exact_data_to_three_servers(self) -> None:
        snapshot = {"since": "start", "until": "end", "timezone": "Europe/Kirov", "warnings": [],
                    "events": [{"occurred_at": "now", "username": "alice", "ip": "192.0.2.1", "method": "publickey"}]}
        analysis = {"snapshot_sha256": agent.digest(snapshot), "since": "start", "until": "end",
                    "timezone": "Europe/Kirov", "total": 1, "unique_ips": 1,
                    "by_user": {"alice": 1}, "by_ip": {"192.0.2.1": 1}, "warnings": []}
        returned = [
            {"snapshot": snapshot, "snapshot_sha256": agent.digest(snapshot), "event_count": 1},
            {"analysis": analysis, "analysis_sha256": agent.digest(analysis)},
            {"path": "/home/heimdall/ai-advent-day20/reports/ssh-test.md", "bytes": 4,
             "sha256": "abc", "analysis_sha256": agent.digest(analysis)},
        ]
        seen = []
        completions = []

        class Client:
            def __init__(self, server):
                self.server = server

            async def call_tool(self, name, args):
                seen.append((self.server.role, name, args))
                value = returned[len(seen) - 1]
                return types.CallToolResult(is_error=False, content=[types.TextContent(type="text", text=json.dumps(value))])

        @asynccontextmanager
        async def connect(server):
            yield Client(server)

        async def complete(_http, _key, messages, _tools):
            completions.append(1)
            self.assertNotIn("192.0.2.1", json.dumps(messages))
            self.assertNotIn("alice", json.dumps(messages))
            number = len([message for message in messages if message["role"] == "tool"])
            if number == 3:
                return {"choices": [{"message": {"role": "assistant", "content": "Готово."}}]}
            alias = tuple(agent.BY_ALIAS)[number]
            args = ({"hours": 24}, {"source_ref": agent.digest(snapshot)},
                    {"analysis_ref": agent.digest(analysis)})[number]
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": str(number), "type": "function", "function": {"name": alias, "arguments": json.dumps(args)}}]}}]}

        trace = []
        with patch.object(agent, "connect", connect), patch.object(agent, "download_report", new=AsyncMock(return_value={
            "local_path": "/tmp/report.md", "bytes": 4, "sha256": "abc"})), patch.object(agent, "api_key", return_value="test"):
            answer = await agent.run_turn("Сохрани отчёт", lambda label, data: trace.append((label, data)), complete=complete)
        self.assertEqual([item[0] for item in seen], ["source", "analyze", "report"])
        self.assertEqual(len(completions), 3)
        self.assertEqual(seen[1][2]["snapshot"], snapshot)
        self.assertEqual(seen[2][2]["analysis"], analysis)
        self.assertIn("/tmp/report.md", answer)
        self.assertIn("ВМ REPORT", answer)
        self.assertIn("REPORT RECEIVED", [label for label, _ in trace])

    async def test_wrong_route_is_rejected_before_mcp_call(self) -> None:
        async def complete(*_args):
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "report__save_report", "arguments": "{}"}}]}}]}
        events = []
        with self.assertRaisesRegex(RuntimeError, "Неверный маршрут"):
            await agent.run_turn("Отчёт", lambda label, data: events.append(label), complete=complete, key="test")
        self.assertIn("ROUTE REJECTED", events)
        self.assertNotIn("MCP CALL", events)

    async def test_client_rejects_modified_report(self) -> None:
        saved = {"path": "/home/heimdall/ai-advent-day20/reports/ssh-2026-09-25_20-25-58-ab8998.md",
                 "bytes": 4, "sha256": "0" * 64}
        with patch.object(report_download, "read_remote", new=AsyncMock(return_value=b"test")):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                await report_download.download_report(saved)

    async def test_tui_shows_short_route_and_client_result(self) -> None:
        async def fake_discover(server):
            tool = types.Tool(name=server.tool, input_schema={"type": "object"})
            return Discovery(server.role, "1", "2025-11-25", (tool,))

        async def fake_turn(question, trace):
            trace("USER", question)
            for server in SERVERS:
                trace("MCP CALL", {"server": server.host, "tool": server.tool,
                                   "model_arguments": {}, "forwarded_sha256": None})
                summary = ({"source_ref": "a" * 64, "event_count": 1},
                           {"analysis_ref": "b" * 64, "total": 1, "unique_ips": 1},
                           {"bytes": 4})[SERVERS.index(server)]
                trace("MCP RESULT", {"server": server.host, "tool": server.tool, "summary": summary})
            trace("REPORT RECEIVED", {"local_path": "/tmp/report.md", "bytes": 4, "sha256": "abc"})
            return "Один вход. Локальный отчёт: /tmp/report.md"

        with tempfile.TemporaryDirectory() as directory, patch("main.discover_tools", side_effect=fake_discover), patch("main.run_turn", side_effect=fake_turn):
            app = MCPAgentApp(log_dir=Path(directory))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.workers.wait_for_complete()
                self.assertFalse(app.query_one("#send", Button).disabled)
                await pilot.click("#send")
                await app.workers.wait_for_complete()
                titles = [title for title, _ in app.events]
                self.assertEqual(sum("LLM →" in title for title in titles), 3)
                self.assertTrue(any("ОТЧЁТ ПОЛУЧЕН" in title for title in titles))
                self.assertNotIn("messages", "\n".join(titles))
                self.assertGreater(app.query_one("#steps", OptionList).option_count, 5)
                text = "\n".join(line.text for line in app.query_one("#answer", RichLog).lines)
                self.assertIn("/tmp/report.md", text)


if __name__ == "__main__":
    unittest.main()
