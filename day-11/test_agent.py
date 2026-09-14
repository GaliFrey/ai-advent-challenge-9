"""Проверки prompt и транзакционного сохранения диалога."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from agent import Agent, AgentConfig, AgentError
from memory import MemoryLayers


CONFIG = AgentConfig(api_key="fake-key")


def response(text: str = "Ответ", *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        },
    )


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_contains_all_layers_in_explicit_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryLayers(Path(directory))
            memory.add_long_term("Python и uv")
            memory.add_working("Offline CLI")
            memory.commit_exchange("Старый вопрос", "Старый ответ")
            seen: dict = {}

            def handler(request: httpx.Request) -> httpx.Response:
                seen.update(json.loads(request.content))
                return response()

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = Agent(CONFIG, client, memory)
                await agent.ask("Новый вопрос")

            messages = seen["messages"]
            self.assertIn("ДОЛГОВРЕМЕННАЯ ПАМЯТЬ", messages[0]["content"])
            self.assertIn("Python и uv", messages[0]["content"])
            self.assertIn("РАБОЧАЯ ПАМЯТЬ", messages[0]["content"])
            self.assertIn("Offline CLI", messages[0]["content"])
            self.assertEqual(messages[1:3], list(memory.short_term[:2]))
            self.assertEqual(messages[-1], {"role": "user", "content": "Новый вопрос"})
            reopened = MemoryLayers(Path(directory))
            self.assertEqual(list(reopened.last_prompt), messages)

    async def test_api_failure_does_not_commit_short_term(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryLayers(Path(directory))
            transport = httpx.MockTransport(lambda request: httpx.Response(500))
            async with httpx.AsyncClient(transport=transport) as client:
                agent = Agent(CONFIG, client, memory)
                with self.assertRaisesRegex(AgentError, "HTTP 500"):
                    await agent.ask("Не сохраняй")
            self.assertEqual(memory.short_term, ())

    async def test_unfinished_response_does_not_commit_short_term(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryLayers(Path(directory))
            transport = httpx.MockTransport(lambda request: response(finish_reason="length"))
            async with httpx.AsyncClient(transport=transport) as client:
                agent = Agent(CONFIG, client, memory)
                with self.assertRaisesRegex(AgentError, "не завершён"):
                    await agent.ask("Не сохраняй")
            self.assertEqual(memory.short_term, ())
