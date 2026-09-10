"""Проверки агента и учёта summary без внешнего API."""

from __future__ import annotations

import json
import unittest

import httpx

from agent import Agent, AgentConfig, AgentError, DEFAULT_SYSTEM_PROMPT, parse_usage
from memory import FullHistoryMemory, SummaryMemory


CONFIG = AgentConfig(api_key="fake-key", model="deepseek-v4-flash")


def response(
    text: str,
    input_tokens: int,
    output_tokens: int,
    *,
    finish_reason: str = "stop",
) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    })


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_history_is_sent_and_counted(self):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return response("ПРИНЯТО", 20 + len(requests), 3)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client, FullHistoryMemory())
            await agent.ask("Первый факт")
            await agent.ask("Второй факт")

        self.assertEqual(
            [message["role"] for message in requests[1]["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(agent.memory.total_message_count, 4)
        self.assertEqual(agent.stats.chat_attempts, 2)
        self.assertEqual(agent.stats.successful_chat_requests, 2)
        self.assertEqual(agent.stats.chat_input_tokens, 43)
        self.assertEqual(agent.stats.chat_output_tokens, 6)
        self.assertEqual(agent.stats.total_tokens, 49)
        self.assertEqual(agent.stats.summary_total_tokens, 0)

    async def test_summary_replaces_old_messages_and_its_usage_is_included(self):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if payload["messages"][0]["content"].startswith("Ты сжимаешь"):
                return response("Проект ORBITA-7429; дата 26.10.2026.", 40, 8)
            return response("ПРИНЯТО", 20, 2)

        memory = SummaryMemory(keep_recent=4, interval=10)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client, memory)
            for index in range(5):
                reply = await agent.ask(f"Факт {index}")

            self.assertIsNotNone(reply.compression)
            await agent.ask("Что ты помнишь?")

        self.assertEqual(len(requests), 7)
        summary_request = requests[5]
        summary_source = json.loads(summary_request["messages"][1]["content"])
        self.assertIsNone(summary_source["previous_summary"])
        self.assertEqual(len(summary_source["messages"]), 6)
        self.assertEqual(memory.raw_message_count, 6)
        self.assertEqual(memory.summarized_message_count, 6)
        self.assertIn("ORBITA-7429", requests[6]["messages"][0]["content"])
        self.assertEqual(
            [message["role"] for message in requests[6]["messages"]],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        self.assertEqual(agent.stats.successful_chat_requests, 6)
        self.assertEqual(agent.stats.successful_summary_requests, 1)
        self.assertEqual(agent.stats.chat_total_tokens, 132)
        self.assertEqual(agent.stats.summary_total_tokens, 48)
        self.assertEqual(agent.stats.total_tokens, 180)

    async def test_summary_failure_keeps_raw_history_and_returns_chat_answer(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 6:
                return httpx.Response(429, json={"error": {"message": "secret"}})
            return response("ПРИНЯТО", 10, 2)

        memory = SummaryMemory(keep_recent=4, interval=10)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client, memory)
            for index in range(5):
                reply = await agent.ask(f"Факт {index}")

        self.assertEqual(reply.text, "ПРИНЯТО")
        self.assertIn("HTTP 429", reply.compression_error or "")
        self.assertEqual(memory.raw_message_count, 10)
        self.assertEqual(memory.summarized_message_count, 0)
        self.assertIsNone(memory.summary)
        self.assertEqual(agent.stats.successful_chat_requests, 5)
        self.assertEqual(agent.stats.summary_attempts, 1)
        self.assertEqual(agent.stats.successful_summary_requests, 0)

    async def test_failed_chat_does_not_change_history(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: response("Обрезано", 8, 2, finish_reason="length")
            )
        ) as client:
            agent = Agent(CONFIG, client, FullHistoryMemory())
            with self.assertRaisesRegex(AgentError, "не завершён"):
                await agent.ask("Вопрос")

        self.assertEqual(agent.memory.total_message_count, 0)
        self.assertEqual(agent.stats.chat_attempts, 1)
        self.assertEqual(agent.stats.successful_chat_requests, 0)

    async def test_empty_message_does_not_call_api(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response("Ответ", 1, 1)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client, FullHistoryMemory())
            with self.assertRaisesRegex(AgentError, "пустым"):
                await agent.ask("  ")

        self.assertEqual(calls, 0)


class ParsingAndConfigTests(unittest.TestCase):
    def test_default_system_prompt_allows_new_coding_requests(self):
        self.assertIn("используй общие знания и пиши код", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Ограничения формата из старых реплик", DEFAULT_SYSTEM_PROMPT)
        self.assertNotIn("Используй сведения только из переданного контекста", DEFAULT_SYSTEM_PROMPT)

    def test_usage_must_be_consistent(self):
        with self.assertRaisesRegex(ValueError, "совпадает"):
            parse_usage({
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 99,
                }
            })

    def test_invalid_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            AgentConfig(api_key="").check()
        with self.assertRaisesRegex(ValueError, "temperature"):
            AgentConfig(api_key="fake", temperature=2.1).check()
        with self.assertRaisesRegex(ValueError, "Лимиты"):
            AgentConfig(api_key="fake", summary_max_tokens=0).check()


if __name__ == "__main__":
    unittest.main()
