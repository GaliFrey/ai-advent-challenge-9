"""Headless-проверки TUI и автодемо."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import httpx
from textual.widgets import Button, Input, RichLog, Select, TabPane

from agent import AgentConfig
from main import (
    CONTROL_QUESTION,
    LONG_TERM_DEMO_NOTE,
    SHORT_TERM_DEMO_MESSAGE,
    WORKING_DEMO_NOTE,
    MemoryLayersApp,
)
from memory import MemoryLayers


CONFIG = AgentConfig(api_key="fake-key")


def response(text: str = "Краткий ответ") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        },
    )


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_layout_exposes_chat_memory_prompt_and_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MemoryLayersApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(180, 52)):
                self.assertEqual(len(app.query(RichLog)), 3)
                self.assertEqual(len(app.query(TabPane)), 2)
                self.assertEqual(len(app.query(Input)), 1)
                self.assertEqual(len(app.query(Select)), 3)
                ids = {button.id for button in app.query(Button)}
                self.assertTrue({"ask", "save-working", "save-long", "new-session", "new-task", "demo"} <= ids)

    async def test_manual_routing_does_not_call_api(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response()

        with tempfile.TemporaryDirectory() as directory:
            app = MemoryLayersApp(
                CONFIG,
                data_dir=Path(directory),
                transport=httpx.MockTransport(handler),
            )
            async with app.run_test(size=(180, 52)):
                field = app.query_one("#input", Input)
                field.value = "Только текущая задача"
                app.save_working()
                field.value = "Предпочтение пользователя"
                app.save_long()

                self.assertEqual(calls, 0)
                self.assertEqual(app.memory.working, ("Только текущая задача",))
                self.assertEqual(app.memory.long_term, ("Предпочтение пользователя",))
                self.assertEqual(app.memory.short_term, ())

    async def test_successful_question_enters_only_short_term(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MemoryLayersApp(
                CONFIG,
                data_dir=Path(directory),
                transport=httpx.MockTransport(lambda request: response()),
            )
            async with app.run_test(size=(180, 52)):
                app.query_one("#input", Input).value = "Вопрос"
                app.start_ask()
                await app.workers.wait_for_complete()

                self.assertEqual(len(app.memory.short_term), 2)
                self.assertEqual(app.memory.working, ())
                self.assertEqual(app.memory.long_term, ())

    async def test_new_session_and_new_task_apply_correct_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MemoryLayersApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(180, 52)):
                app.memory.commit_exchange("Вопрос", "Ответ")
                app.memory.add_working("Рабочая")
                app.memory.add_long_term("Долгая")
                old_task = app.memory.task_id
                old_session = app.memory.session_id

                app.new_session()
                self.assertNotEqual(app.memory.session_id, old_session)
                self.assertEqual(app.memory.short_term, ())
                self.assertEqual(app.memory.working, ("Рабочая",))
                self.assertEqual(app.memory.long_term, ("Долгая",))

                app.new_task()
                self.assertNotEqual(app.memory.task_id, old_task)
                self.assertEqual(app.memory.working, ())
                self.assertEqual(app.memory.long_term, ("Долгая",))

    async def test_demo_uses_six_calls_and_finishes_with_long_term_only(self):
        calls = 0
        prompts: list[list[dict[str, str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            prompts.append(json.loads(request.content)["messages"])
            return response(f"Ответ {calls}")

        with tempfile.TemporaryDirectory() as directory:
            app = MemoryLayersApp(
                CONFIG,
                data_dir=Path(directory),
                transport=httpx.MockTransport(handler),
            )
            async with app.run_test(size=(180, 52)):
                app.action_start_demo()
                await app.workers.wait_for_complete()

                self.assertEqual(calls, 6)
                self.assertEqual(app.memory.short_term, ())
                self.assertEqual(app.memory.working, ())
                self.assertEqual(len(app.memory.long_term), 1)
                self.assertEqual(app.memory.task_id, "task-02")
                self.assertEqual(app.total_tokens, 150)
                self.assertEqual(prompts[0][-1]["content"], CONTROL_QUESTION)
                self.assertNotIn(LONG_TERM_DEMO_NOTE, prompts[0][0]["content"])
                self.assertIn(LONG_TERM_DEMO_NOTE, prompts[1][0]["content"])
                self.assertNotIn(WORKING_DEMO_NOTE, prompts[1][0]["content"])
                self.assertIn(WORKING_DEMO_NOTE, prompts[2][0]["content"])
                self.assertEqual(prompts[3][-1]["content"], SHORT_TERM_DEMO_MESSAGE)
                self.assertTrue(
                    any(
                        message["content"] == SHORT_TERM_DEMO_MESSAGE
                        for message in prompts[4][1:-1]
                    )
                )
                self.assertFalse(
                    any(
                        message["content"] == SHORT_TERM_DEMO_MESSAGE
                        for message in prompts[5][1:-1]
                    )
                )

    async def test_saved_session_can_be_selected_and_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = MemoryLayers(root)
            first_prompt = [
                {"role": "system", "content": "Системная роль"},
                {"role": "user", "content": "Первый вопрос"},
            ]
            memory.commit_exchange(
                "Первый вопрос", "Первый ответ", request_messages=first_prompt
            )
            memory.create_session("session-02")
            memory.commit_exchange(
                "Второй вопрос",
                "Второй ответ",
                request_messages=[
                    {"role": "system", "content": "Другая роль"},
                    {"role": "user", "content": "Второй вопрос"},
                ],
            )

            app = MemoryLayersApp(CONFIG, data_dir=root)
            async with app.run_test(size=(180, 52)) as pilot:
                self.assertEqual(app.memory.session_id, "session-02")
                app.query_one("#session", Select).value = "session-01"
                await pilot.pause()
                self.assertEqual(app.memory.session_id, "session-01")
                self.assertEqual(app.memory.short_term[0]["content"], "Первый вопрос")
                self.assertEqual(list(app.memory.last_prompt), first_prompt)

    async def test_clear_requires_two_clicks(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MemoryLayersApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(180, 52)):
                app.memory.add_long_term("Не удалять сразу")
                app.clear_active()
                self.assertEqual(app.memory.long_term, ("Не удалять сразу",))
                app.clear_active()
                self.assertEqual(app.memory.long_term, ())
