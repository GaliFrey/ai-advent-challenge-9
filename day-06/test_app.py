"""Проверка структуры TUI без сети и внешнего API."""

from __future__ import annotations

import json
import unittest

import httpx
from textual.widgets import Input, RichLog, Static

from agent import AgentConfig
from main import DualAgentApp


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_creates_two_independent_agents(self):
        app = DualAgentApp(AgentConfig(api_key="fake-key", model="fake-model"))
        async with app.run_test(size=(140, 40)):
            self.assertEqual(set(app.agents), {"left", "right"})
            self.assertIsNot(app.agents["left"], app.agents["right"])
            self.assertIsNot(app.agents["left"].stats, app.agents["right"].stats)
            self.assertEqual(len(app.query(RichLog)), 2)
            self.assertEqual(len(app.query(Input)), 2)
            self.assertIn("Модель: fake-model", app.query_one("#stats-left", Static).content)

    async def test_messages_sent_from_panels_use_only_their_agent_history(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            answer = "Вас зовут Влад." if any(
                message["content"] == "Меня зовут Влад" for message in payload["messages"]
            ) else "Имя не сообщалось."
            return httpx.Response(200, json={
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            })

        app = DualAgentApp(
            AgentConfig(api_key="fake-key", model="fake-model"),
            transport=httpx.MockTransport(handler),
        )
        async with app.run_test(size=(140, 40)):
            app.query_one("#input-left", Input).value = "Меня зовут Влад"
            app.start_request("left")
            await app.workers.wait_for_complete()
            app.query_one("#input-left", Input).value = "Как меня зовут?"
            app.start_request("left")
            await app.workers.wait_for_complete()
            app.query_one("#input-right", Input).value = "Как меня зовут?"
            app.start_request("right")
            await app.workers.wait_for_complete()

            self.assertEqual(len(requests), 3)
            self.assertEqual(len(app.agents["left"].messages), 5)
            self.assertEqual(len(app.agents["right"].messages), 3)
            self.assertNotIn("Влад", json.dumps(requests[2], ensure_ascii=False))
            self.assertIn("всего 32", app.query_one("#stats-left", Static).content)
            self.assertIn("всего 16", app.query_one("#stats-right", Static).content)


if __name__ == "__main__":
    unittest.main()
