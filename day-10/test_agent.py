"""Проверки API-контрактов и учёта токенов без внешних вызовов."""

from __future__ import annotations

import json
import unittest

import httpx

from agent import Agent, AgentConfig, AgentError, FACTS_CONTEXT_PREFIX, parse_facts, parse_usage
from memory import BranchingMemory, FactsMemory, SlidingWindowMemory


CONFIG = AgentConfig(api_key="fake-key", model="deepseek-v4-flash")


def response(text: str, input_tokens: int = 12, output_tokens: int = 3) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    )


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_sliding_prompt_contains_only_last_n_completed_messages(self):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return response("ПРИНЯТО")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client, SlidingWindowMemory(4))
            for index in range(4):
                await agent.ask(f"Вопрос {index}")

        contents = [item["content"] for item in requests[-1]["messages"]]
        self.assertNotIn("Вопрос 0", contents)
        self.assertIn("Вопрос 1", contents)
        self.assertEqual(contents[-1], "Вопрос 3")
        self.assertEqual(agent.memory.total_message_count, 8)
        self.assertEqual(agent.stats.chat_attempts, 4)
        self.assertEqual(agent.stats.total_tokens, 60)

    async def test_facts_are_updated_before_chat_and_service_usage_is_counted(self):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if payload["messages"][0]["content"].startswith("Ты обновляешь"):
                return response('{"project":"FORUM-731"}', 20, 5)
            return response("ПРИНЯТО", 30, 2)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            memory = FactsMemory(4)
            agent = Agent(CONFIG, client, memory)
            reply = await agent.ask("Код проекта FORUM-731")

        self.assertEqual(len(requests), 2)
        self.assertIn(FACTS_CONTEXT_PREFIX, requests[1]["messages"][0]["content"])
        self.assertIn("FORUM-731", requests[1]["messages"][0]["content"])
        self.assertEqual(memory.facts, {"project": "FORUM-731"})
        self.assertEqual(reply.facts_usage.total_tokens, 25)
        self.assertEqual(agent.stats.chat_total_tokens, 32)
        self.assertEqual(agent.stats.facts_total_tokens, 25)
        self.assertEqual(agent.stats.total_tokens, 57)

    async def test_invalid_facts_stop_turn_without_committing_history(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response("это не json")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            memory = FactsMemory(4)
            agent = Agent(CONFIG, client, memory)
            with self.assertRaisesRegex(AgentError, "невалидный JSON"):
                await agent.ask("Факт")

        self.assertEqual(calls, 1)
        self.assertEqual(memory.total_message_count, 0)
        self.assertEqual(agent.stats.facts_attempts, 1)
        self.assertEqual(agent.stats.facts_total_tokens, 15)
        self.assertEqual(agent.stats.chat_attempts, 0)

    async def test_truncated_chat_is_counted_but_not_committed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Обрезано"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            memory = SlidingWindowMemory(4)
            agent = Agent(CONFIG, client, memory)
            with self.assertRaisesRegex(AgentError, "не завершён"):
                await agent.ask("Вопрос")

        self.assertEqual(memory.total_message_count, 0)
        self.assertEqual(agent.stats.chat_attempts, 1)
        self.assertEqual(agent.stats.successful_chat_requests, 0)
        self.assertEqual(agent.stats.chat_total_tokens, 13)

    async def test_branch_prompt_does_not_contain_sibling_messages(self):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return response("ПРИНЯТО")

        memory = BranchingMemory()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client, memory)
            await agent.ask("Общее")
            memory.create_checkpoint()
            await agent.ask("Только A")
            memory.switch_branch("B")
            await agent.ask("Только B")

        final_contents = [item["content"] for item in requests[-1]["messages"]]
        self.assertIn("Общее", final_contents)
        self.assertIn("Только B", final_contents)
        self.assertNotIn("Только A", final_contents)


class ParsingTests(unittest.TestCase):
    def test_parse_facts_accepts_json_fence_but_requires_object(self):
        self.assertEqual(parse_facts('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaisesRegex(AgentError, "JSON-объекта"):
            parse_facts("[]")

    def test_usage_must_be_consistent(self):
        with self.assertRaisesRegex(ValueError, "совпадает"):
            parse_usage(
                {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 99}}
            )


if __name__ == "__main__":
    unittest.main()
