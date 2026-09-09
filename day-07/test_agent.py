"""Локальные проверки агента без реальных API-вызовов."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from agent import Agent, AgentConfig, AgentError
from storage import HistoryStorageError, JsonHistoryStore


CONFIG = AgentConfig(api_key="fake-key", model="fake-model")


def response(
    answer: str,
    input_tokens: int,
    output_tokens: int,
    *,
    finish_reason: str = "stop",
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    )


class AgentPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_is_loaded_after_recreation_and_sent_in_full(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                return response("Запомнил: вас зовут Влад.", 20, 7)
            return response("Вас зовут Влад.", 35, 5)

        with tempfile.TemporaryDirectory() as directory:
            store = JsonHistoryStore(Path(directory), "left")
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as first_client:
                first_agent = Agent(CONFIG, first_client, store)
                await first_agent.ask("Меня зовут Влад")

            async with httpx.AsyncClient(transport=transport) as second_client:
                restored_agent = Agent(CONFIG, second_client, store)
                self.assertEqual(restored_agent.stats.history_messages, 2)
                self.assertEqual(restored_agent.stats.successful_requests, 0)
                self.assertEqual(restored_agent.stats.session_total_tokens, 0)
                reply = await restored_agent.ask("Как меня зовут?")

            self.assertEqual(reply.text, "Вас зовут Влад.")
            self.assertEqual(
                [message["role"] for message in requests[1]["messages"]],
                ["system", "user", "assistant", "user"],
            )
            self.assertEqual(
                [message["content"] for message in requests[1]["messages"]][1:],
                ["Меня зовут Влад", "Запомнил: вас зовут Влад.", "Как меня зовут?"],
            )

    async def test_two_agents_keep_separate_files_and_statistics(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            contains_name = any(
                message["content"] == "Меня зовут Влад" for message in payload["messages"]
            )
            answer = "Вас зовут Влад." if contains_name else "Имя не сообщалось."
            return response(answer, 20, 5)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                left = Agent(CONFIG, client, JsonHistoryStore(root, "left"))
                right = Agent(CONFIG, client, JsonHistoryStore(root, "right"))
                await left.ask("Меня зовут Влад")
                await right.ask("Как меня зовут?")

            self.assertNotIn("Влад", json.dumps(requests[1], ensure_ascii=False))
            self.assertEqual(left.stats.history_messages, 2)
            self.assertEqual(right.stats.history_messages, 2)
            self.assertTrue((root / "left.json").exists())
            self.assertTrue((root / "right.json").exists())

    async def test_failed_and_incomplete_requests_are_not_persisted(self):
        call = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call
            call += 1
            if call == 1:
                return httpx.Response(429, json={"error": {"message": "fake-key"}})
            return response("Обрезанный ответ", 12, 4, finish_reason="length")

        with tempfile.TemporaryDirectory() as directory:
            store = JsonHistoryStore(Path(directory), "left")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = Agent(CONFIG, client, store)
                with self.assertRaisesRegex(AgentError, "HTTP 429"):
                    await agent.ask("Ошибочный запрос")
                with self.assertRaisesRegex(AgentError, "не завершён"):
                    await agent.ask("Незавершённый запрос")

            self.assertEqual(len(agent.messages), 1)
            self.assertEqual(agent.stats.attempts, 2)
            self.assertEqual(agent.stats.successful_requests, 0)
            self.assertEqual(store.load(), [])
            self.assertNotIn("fake-key", agent.stats.last_error or "")

    async def test_storage_failure_does_not_commit_answer_to_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonHistoryStore(Path(directory), "left")
            store.save([
                {"role": "user", "content": "Старый вопрос"},
                {"role": "assistant", "content": "Старый ответ"},
            ])
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: response("Новый ответ", 8, 3))
            ) as client:
                agent = Agent(CONFIG, client, store)
                with patch.object(
                    store,
                    "save",
                    side_effect=HistoryStorageError("тестовый сбой"),
                ):
                    with self.assertRaisesRegex(AgentError, "сохранить не удалось"):
                        await agent.ask("Новый вопрос")

            self.assertEqual(len(agent.messages), 3)
            self.assertEqual(agent.stats.history_messages, 2)
            self.assertEqual(agent.stats.successful_requests, 0)
            self.assertEqual(len(store.load()), 2)

    async def test_clear_removes_only_selected_history_and_resets_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: response("Ответ", 5, 2))
            ) as client:
                left = Agent(CONFIG, client, JsonHistoryStore(root, "left"))
                right = Agent(CONFIG, client, JsonHistoryStore(root, "right"))
                await left.ask("Слева")
                await right.ask("Справа")
                left.clear_history()

            self.assertEqual(JsonHistoryStore(root, "left").load(), [])
            self.assertEqual(len(JsonHistoryStore(root, "right").load()), 2)
            self.assertEqual(len(left.messages), 1)
            self.assertEqual(left.stats.attempts, 0)
            self.assertEqual(right.stats.history_messages, 2)

    async def test_empty_message_does_not_call_api(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response("Не должно вызываться", 1, 1)

        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = Agent(CONFIG, client, JsonHistoryStore(Path(directory), "left"))
                with self.assertRaisesRegex(AgentError, "не должно быть пустым"):
                    await agent.ask("   ")

        self.assertEqual(calls, 0)
        self.assertEqual(agent.stats.attempts, 0)


class ConfigTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            AgentConfig(api_key="").check()
        with self.assertRaisesRegex(ValueError, "temperature"):
            AgentConfig(api_key="fake", temperature=2.1).check()
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            AgentConfig(api_key="fake", max_tokens=0).check()


if __name__ == "__main__":
    unittest.main()
