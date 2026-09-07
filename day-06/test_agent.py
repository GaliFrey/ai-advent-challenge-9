"""Локальные проверки агента без реальных API-вызовов."""

from __future__ import annotations

import json
import unittest

import httpx

from agent import Agent, AgentConfig, AgentError


CONFIG = AgentConfig(api_key="fake-key", model="fake-model")


def response(answer: str, input_tokens: int, output_tokens: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    )


class AgentIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_agents_keep_separate_histories_and_statistics(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            user_messages = [
                message["content"] for message in payload["messages"] if message["role"] == "user"
            ]
            if user_messages[-1] == "Меня зовут Влад":
                return response("Запомнил: вас зовут Влад.", 20, 7)
            if "Меня зовут Влад" in user_messages:
                return response("Вас зовут Влад.", 35, 5)
            return response("В этой сессии имя не сообщалось.", 18, 8)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            left = Agent(CONFIG, client)
            right = Agent(CONFIG, client)
            await left.ask("Меня зовут Влад")
            left_reply = await left.ask("Как меня зовут?")
            right_reply = await right.ask("Как меня зовут?")

        self.assertEqual(left_reply.text, "Вас зовут Влад.")
        self.assertEqual(right_reply.text, "В этой сессии имя не сообщалось.")
        self.assertEqual([message["role"] for message in requests[1]["messages"]], [
            "system", "user", "assistant", "user"
        ])
        self.assertEqual([message["role"] for message in requests[2]["messages"]], [
            "system", "user"
        ])
        self.assertNotIn("Влад", json.dumps(requests[2], ensure_ascii=False))
        self.assertEqual(left.stats.history_messages, 4)
        self.assertEqual(left.stats.successful_requests, 2)
        self.assertEqual(left.stats.session_total_tokens, 67)
        self.assertEqual(right.stats.history_messages, 2)
        self.assertEqual(right.stats.successful_requests, 1)
        self.assertEqual(right.stats.session_total_tokens, 26)

    async def test_failed_request_is_not_committed_to_history(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "fake-key"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client)
            with self.assertRaisesRegex(AgentError, "HTTP 429"):
                await agent.ask("Секретное сообщение")

        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.stats.attempts, 1)
        self.assertEqual(agent.stats.successful_requests, 0)
        self.assertEqual(agent.stats.session_total_tokens, 0)
        self.assertNotIn("fake-key", agent.stats.last_error or "")

    async def test_empty_message_does_not_call_api(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response("Не должно вызываться", 1, 1)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client)
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
