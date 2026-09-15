"""Проверки system-профиля и транзакционного сохранения ответа."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from agent import Agent, AgentConfig, AgentError
from user_profile import DEFAULT_PROFILES
from session import SessionStore


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
    async def test_profile_is_in_system_of_every_request(self):
        seen: list[list[dict[str, str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content)["messages"])
            return response(f"Ответ {len(seen)}")

        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = store.create("session-01", "tech-lead")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = Agent(CONFIG, client, session, store, DEFAULT_PROFILES[0])
                await agent.ask("Первый вопрос")
                await agent.ask("Второй вопрос")

            self.assertEqual(len(seen), 2)
            for messages in seen:
                self.assertEqual(messages[0]["role"], "system")
                self.assertIn("Краткий техлид", messages[0]["content"])
                self.assertIn("Ровно три", messages[0]["content"])
                self.assertEqual(sum(item["role"] == "system" for item in messages), 1)
            self.assertNotIn("Краткий техлид", str(seen[1][1:]))
            self.assertEqual(store.load("session-01").last_request_messages, seen[-1])

    async def test_same_question_gets_different_profile_system(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            first_session = store.create("session-01", "tech-lead")
            second_session = store.create("session-02", "mentor")
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response())) as client:
                first = Agent(CONFIG, client, first_session, store, DEFAULT_PROFILES[0])
                second = Agent(CONFIG, client, second_session, store, DEFAULT_PROFILES[1])
                first_prompt = first.build_messages("Одинаковый вопрос")
                second_prompt = second.build_messages("Одинаковый вопрос")

            self.assertEqual(first_prompt[-1], second_prompt[-1])
            self.assertIn("Краткий техлид", first_prompt[0]["content"])
            self.assertIn("Обучающий наставник", second_prompt[0]["content"])
            self.assertNotEqual(first_prompt[0], second_prompt[0])

    async def test_api_failure_does_not_change_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = store.create("session-01", "tech-lead")
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
                agent = Agent(CONFIG, client, session, store, DEFAULT_PROFILES[0])
                with self.assertRaisesRegex(AgentError, "HTTP 500"):
                    await agent.ask("Не сохраняй")

            self.assertEqual(store.load("session-01").messages, [])

    async def test_unfinished_response_does_not_change_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = store.create("session-01", "tech-lead")
            transport = httpx.MockTransport(lambda request: response(finish_reason="length"))
            async with httpx.AsyncClient(transport=transport) as client:
                agent = Agent(CONFIG, client, session, store, DEFAULT_PROFILES[0])
                with self.assertRaisesRegex(AgentError, "не завершён"):
                    await agent.ask("Не сохраняй")

            self.assertEqual(store.load("session-01").messages, [])
