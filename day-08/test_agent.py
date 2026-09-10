"""Локальные проверки агента без внешних запросов."""

from __future__ import annotations

import json
import unittest

import httpx

from agent import (
    Agent,
    AgentConfig,
    AgentError,
    parse_usage,
    provider_error,
    used_context_compression,
)
from token_counter import SUPPORTED_MODEL, TokenCounter, UnsupportedTokenizerError


CONFIG = AgentConfig(api_key="fake-key")


def response(
    answer: str = "OK",
    *,
    prompt_tokens: int = 20,
    completion_tokens: int = 1,
    cost: float | None = 0.000022,
    finish_reason: str = "stop",
    compressed: bool = False,
) -> httpx.Response:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    if cost is not None:
        usage["cost"] = cost
    payload = {
        "choices": [{"message": {"content": answer}, "finish_reason": finish_reason}],
        "usage": usage,
        "provider": "OpenAI",
    }
    if compressed:
        payload["openrouter_metadata"] = {
            "pipeline": [{"type": "context_compression", "name": "context-compression"}]
        }
    return httpx.Response(200, json=payload)


class TokenCounterTests(unittest.TestCase):
    def test_counter_supports_only_pinned_model(self):
        counter = TokenCounter(SUPPORTED_MODEL)
        self.assertGreater(counter.count_text("Привет, мир!"), 0)
        self.assertGreater(
            counter.count_messages([{"role": "user", "content": "Привет"}]),
            counter.count_text("Привет"),
        )
        with self.assertRaises(UnsupportedTokenizerError):
            TokenCounter("some/other-model")


class ParsingTests(unittest.TestCase):
    def test_usage_prefers_api_cost_and_validates_totals(self):
        usage = parse_usage(response().json())
        self.assertEqual(usage.prompt_tokens, 20)
        self.assertEqual(usage.completion_tokens, 1)
        self.assertEqual(usage.cost_usd, 0.000022)
        self.assertEqual(usage.cost_source, "usage.cost OpenRouter")

        payload = response(cost=None).json()
        estimated = parse_usage(payload)
        self.assertEqual(estimated.cost_source, "оценка по тарифу")
        self.assertAlmostEqual(estimated.cost_usd, 0.000064)

        payload["usage"]["total_tokens"] = 999
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            parse_usage(payload)

    def test_error_and_compression_metadata_are_parsed_safely(self):
        message, code = provider_error({
            "error": {"message": "  maximum\n context   length ", "code": 400},
        })
        self.assertEqual(message, "maximum context length")
        self.assertEqual(code, "400")
        nested_message, nested_code = provider_error({
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "raw": json.dumps({
                        "error": {
                            "message": "maximum context length exceeded",
                            "code": "context_length_exceeded",
                        }
                    })
                },
            }
        })
        self.assertIn("maximum context length exceeded", nested_message or "")
        self.assertEqual(nested_code, "400")
        self.assertTrue(used_context_compression(response(compressed=True).json()))
        self.assertFalse(used_context_compression(response().json()))


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_uses_required_router_settings_and_accumulates_usage(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return response("OK-1")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client)
            reply = await agent.ask("Ответь ровно: OK-1")

        body = requests[0]
        self.assertEqual(body["model"], SUPPORTED_MODEL)
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["plugins"], [{"id": "context-compression", "enabled": False}])
        self.assertEqual(body["provider"]["order"], ["openai"])
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(reply.text, "OK-1")
        self.assertEqual(agent.stats.attempts, 1)
        self.assertEqual(agent.stats.successful_requests, 1)
        self.assertEqual(agent.stats.cumulative_total_tokens, 21)
        self.assertEqual(len(agent.messages), 3)

    async def test_http_error_does_not_change_history(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={
                "error": {
                    "message": "Maximum context length is 16385 tokens",
                    "code": "context_length_exceeded",
                }
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client)
            before = agent.messages
            with self.assertRaises(AgentError) as caught:
                await agent.ask("Слишком длинный запрос")

        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.provider_code, "context_length_exceeded")
        self.assertIn("Maximum context length", str(caught.exception))
        self.assertNotIn("fake-key", str(caught.exception))
        self.assertEqual(agent.messages, before)
        self.assertEqual(agent.stats.attempts, 1)
        self.assertEqual(agent.stats.successful_requests, 0)

    async def test_incomplete_or_compressed_response_is_not_committed(self):
        responses = [response(finish_reason="length"), response(compressed=True)]

        def handler(request: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client)
            for message in ("Первый", "Второй"):
                with self.assertRaises(AgentError):
                    await agent.ask(message)

        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.stats.successful_requests, 0)


class ConfigTests(unittest.TestCase):
    def test_experiment_configuration_is_pinned(self):
        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            AgentConfig(api_key="").check()
        with self.assertRaisesRegex(ValueError, "воспроизводимости"):
            AgentConfig(api_key="x", model="other/model").check()
        with self.assertRaisesRegex(ValueError, "Контекстное окно"):
            AgentConfig(api_key="x", context_limit=8_000).check()


if __name__ == "__main__":
    unittest.main()
