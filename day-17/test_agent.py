from __future__ import annotations

import asyncio
import copy
import json
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

from mcp import Client

from agent import initial_history, run_turn
from server import server as datex_server
from servers import SERVERS


class AgentMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_receives_search_and_read_results(self) -> None:
        document_id = "public-docs:article:5620250451197911709"
        responses = [
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "search", "type": "function", "function": {
                    "name": "datex_search", "arguments": json.dumps({"query": "ArraySort", "limit": 2}),
                },
            }]}, "finish_reason": "tool_calls"}]},
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "read", "type": "function", "function": {
                    "name": "datex_read", "arguments": json.dumps({"document_id": document_id}),
                },
            }]}, "finish_reason": "tool_calls"}]},
            {"choices": [{"message": {"role": "assistant", "content": "ArraySort возвращает новый массив. Источник: http://docs.datex.ru/article.htm?id=5620250451197911709"}, "finish_reason": "stop"}]},
        ]
        requests: list[list[dict]] = []
        events: list[str] = []

        async def complete(_http, _key, messages, _tools):
            requests.append(copy.deepcopy(messages))
            return responses[len(requests) - 1]

        @asynccontextmanager
        async def connect(_server):
            async with Client(datex_server) as client:
                yield client

        async with Client(datex_server) as client:
            tools = tuple((await client.list_tools()).tools)
        with patch("agent.api_key", return_value="test-key"), patch("agent.connect", connect), patch("agent.completion", complete):
            answer, history = await run_turn(
                SERVERS[0], tools, initial_history(SERVERS[0]), "Что делает ArraySort?",
                lambda label, data: events.append(label),
            )
        self.assertIn("ArraySort возвращает", answer)
        self.assertEqual(len(requests), 3)
        self.assertEqual(json.loads(requests[1][-1]["content"])["results"][0]["document_id"], document_id)
        self.assertEqual(json.loads(requests[2][-1]["content"])["document_id"], document_id)
        self.assertEqual(history[-1]["content"], answer)
        self.assertEqual(events.count("MCP CALL"), 2)
        self.assertEqual(events.count("MCP RESULT"), 2)

    async def test_search_results_and_tool_calls_are_bounded(self) -> None:
        documents = (
            "public-docs:article:5620250451197911709",
            "public-docs:article:7562996252397277187",
        )

        def call(identifier: str, name: str, arguments: dict) -> dict:
            return {"id": identifier, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}

        responses = [
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                call("search-1", "datex_search", {"query": "ArraySort", "limit": 10}),
                call("search-2", "datex_search", {"query": "сортировка массива", "limit": 3}),
                call("search-3", "datex_search", {"query": "лишний поиск", "limit": 3}),
            ]}}]},
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                call("read-1", "datex_read", {"document_id": documents[0]}),
                call("read-2", "datex_read", {"document_id": documents[1]}),
            ]}}]},
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                call("search-4", "datex_search", {"query": "ArraySortExt", "limit": 3}),
            ]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "<｜｜DSML｜｜ calls>datex_read</｜｜DSML｜｜ calls>"}}]},
            {"choices": [{"message": {"role": "assistant", "content": "Найдены ArraySort и ArraySortExt."}}]},
        ]
        requests: list[tuple[list[dict], list[dict]]] = []
        trace: list[tuple[str, object]] = []

        async def complete(_http, _key, messages, offered):
            requests.append((copy.deepcopy(messages), copy.deepcopy(offered)))
            return responses[len(requests) - 1]

        @asynccontextmanager
        async def connect(_server):
            async with Client(datex_server) as client:
                yield client

        async with Client(datex_server) as client:
            tools = tuple((await client.list_tools()).tools)
        with patch("agent.api_key", return_value="test-key"), patch("agent.connect", connect), patch("agent.completion", complete):
            answer, _history = await run_turn(
                SERVERS[0], tools, initial_history(SERVERS[0]), "Какие есть сортировки массивов?",
                lambda label, data: trace.append((label, data)),
            )
        self.assertIn("ArraySortExt", answer)
        calls = [data for label, data in trace if label == "MCP CALL"]
        self.assertEqual(len(calls), 4)
        self.assertNotIn("datex_status", {tool["function"]["name"] for tool in requests[0][1]})
        self.assertEqual(calls[0]["arguments"]["limit"], 3)
        first_search_result = next(item for item in requests[1][0] if item.get("tool_call_id") == "search-1")
        self.assertEqual(len(json.loads(first_search_result["content"])["results"]), 3)
        self.assertEqual(sum(label == "MCP SKIPPED" for label, _ in trace), 2)
        self.assertEqual(requests[3][1], [])
        self.assertEqual(requests[3][0][-1]["role"], "user")
        self.assertIn("Лимит инструментов", requests[3][0][-1]["content"])
        self.assertEqual(requests[4][1], [])
        self.assertIn("Это служебный вызов", requests[4][0][-1]["content"])
        self.assertIn("MODEL OUTPUT INVALID", [label for label, _ in trace])


if __name__ == "__main__":
    unittest.main()
