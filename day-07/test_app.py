"""Проверки TUI без сети и внешнего API."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import Button, Input, RichLog, Static

from agent import AgentConfig
from main import DualAgentApp
from storage import JsonHistoryStore


CONFIG = AgentConfig(api_key="fake-key", model="fake-model")


def response(answer: str) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    })


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_creates_two_agents_and_restores_both_chats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            JsonHistoryStore(root, "left").save([
                {"role": "user", "content": "Левый вопрос"},
                {"role": "assistant", "content": "Левый ответ"},
            ])
            JsonHistoryStore(root, "right").save([
                {"role": "user", "content": "Правый вопрос"},
                {"role": "assistant", "content": "Правый ответ"},
            ])
            app = DualAgentApp(CONFIG, history_dir=root)
            async with app.run_test(size=(140, 42)):
                self.assertEqual(set(app.agents), {"left", "right"})
                self.assertIsNot(app.agents["left"], app.agents["right"])
                self.assertEqual(len(app.query(RichLog)), 2)
                self.assertEqual(len(app.query(Input)), 2)
                self.assertEqual(len(app.query(Button)), 4)
                self.assertEqual(app.agents["left"].stats.history_messages, 2)
                self.assertEqual(app.agents["right"].stats.history_messages, 2)
                self.assertGreaterEqual(len(app.query_one("#chat-left", RichLog).lines), 3)
                stats = app.query_one("#stats-left", Static).content
                self.assertIn("История: 2", stats)
                self.assertIn("Запуск: 0/0", stats)

    async def test_messages_from_panels_use_only_their_persisted_history(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            answer = "Вас зовут Влад." if any(
                message["content"] == "Меня зовут Влад" for message in payload["messages"]
            ) else "Имя не сообщалось."
            return response(answer)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = DualAgentApp(
                CONFIG,
                history_dir=root,
                transport=httpx.MockTransport(handler),
            )
            async with app.run_test(size=(140, 42)):
                app.query_one("#input-left", Input).value = "Меня зовут Влад"
                app.start_request("left")
                await app.workers.wait_for_complete()
                app.query_one("#input-right", Input).value = "Как меня зовут?"
                app.start_request("right")
                await app.workers.wait_for_complete()

                self.assertEqual(len(requests), 2)
                self.assertNotIn("Влад", json.dumps(requests[1], ensure_ascii=False))
                self.assertEqual(len(JsonHistoryStore(root, "left").load()), 2)
                self.assertEqual(len(JsonHistoryStore(root, "right").load()), 2)

    async def test_clear_requires_confirmation_and_affects_one_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = [
                {"role": "user", "content": "Вопрос"},
                {"role": "assistant", "content": "Ответ"},
            ]
            JsonHistoryStore(root, "left").save(history)
            JsonHistoryStore(root, "right").save(history)
            app = DualAgentApp(CONFIG, history_dir=root)
            async with app.run_test(size=(140, 42)):
                app.confirm_or_clear_history("left")
                self.assertEqual(len(JsonHistoryStore(root, "left").load()), 2)
                self.assertTrue(app.clear_armed["left"])
                self.assertIn("ПОДТВЕРДИТЬ", str(app.query_one("#clear-left", Button).label))

                app.confirm_or_clear_history("left")
                self.assertEqual(JsonHistoryStore(root, "left").load(), [])
                self.assertEqual(len(JsonHistoryStore(root, "right").load()), 2)
                self.assertEqual(app.agents["left"].stats.history_messages, 0)
                self.assertEqual(app.agents["right"].stats.history_messages, 2)


if __name__ == "__main__":
    unittest.main()
