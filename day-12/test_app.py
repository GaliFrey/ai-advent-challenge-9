"""Headless-проверки TUI, профилей и автодемо."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import Button, Input, RichLog, Select, TabPane

from agent import AgentConfig
from main import CONTROL_QUESTION, FOLLOW_UP_QUESTION, ProfilesApp


CONFIG = AgentConfig(api_key="fake-key")


def response(text: str = "Ответ") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        },
    )


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_layout_exposes_profile_editor_chat_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ProfilesApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(190, 58)):
                self.assertEqual(len(app.query(RichLog)), 2)
                self.assertEqual(len(app.query(TabPane)), 2)
                self.assertEqual(len(app.query(Select)), 3)
                self.assertEqual(len(app.query(Input)), 7)
                ids = {button.id for button in app.query(Button)}
                self.assertTrue(
                    {"ask", "new-session", "save-profile", "new-profile", "delete-profile", "demo"} <= ids
                )

    async def test_profile_can_be_created_edited_and_activated_without_api(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response()

        with tempfile.TemporaryDirectory() as directory:
            app = ProfilesApp(CONFIG, data_dir=Path(directory), transport=httpx.MockTransport(handler))
            async with app.run_test(size=(190, 58)) as pilot:
                await pilot.click("#new-profile")
                values = {
                    "#profile-name": "Редактор",
                    "#profile-address": "Автор",
                    "#profile-language": "Русский",
                    "#profile-style": "Строгий",
                    "#profile-format": "Список замечаний",
                    "#profile-constraints": "Без похвалы",
                }
                for selector, value in values.items():
                    app.query_one(selector, Input).value = value
                await pilot.click("#save-profile")
                await pilot.pause()

                self.assertEqual(calls, 0)
                self.assertEqual(app.active_profile.name, "Редактор")
                self.assertEqual(app.session.profile_id, "profile-01")
                self.assertEqual(len(app.profiles), 3)
                self.assertIn("Редактор", app.agent.build_messages("Проверка")[0]["content"])

    async def test_new_profile_is_valid_and_can_be_saved_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ProfilesApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(190, 58)) as pilot:
                await pilot.click("#new-profile")
                await pilot.click("#save-profile")
                await pilot.pause()

                self.assertEqual(len(app.profiles), 3)
                self.assertEqual(app.active_profile.profile_id, "profile-01")
                self.assertEqual(app.active_profile.name, "Новый профиль")
                self.assertEqual(len(app.profile_store.load()), 3)

    async def test_session_restores_its_selected_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ProfilesApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(190, 58)) as pilot:
                first_id = app.session.session_id
                app.query_one("#profile", Select).value = "mentor"
                await pilot.pause()
                app.new_session()
                second_id = app.session.session_id
                app.query_one("#profile", Select).value = "tech-lead"
                await pilot.pause()

                app.query_one("#session", Select).value = first_id
                await pilot.pause()
                self.assertEqual(app.active_profile.profile_id, "mentor")
                app.query_one("#session", Select).value = second_id
                await pilot.pause()
                self.assertEqual(app.active_profile.profile_id, "tech-lead")

    async def test_successful_question_saves_actual_messages(self):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return response()

        with tempfile.TemporaryDirectory() as directory:
            app = ProfilesApp(CONFIG, data_dir=Path(directory), transport=httpx.MockTransport(handler))
            async with app.run_test(size=(190, 58)):
                app.query_one("#message", Input).value = "Вопрос"
                app.start_ask()
                await app.workers.wait_for_complete()

                self.assertEqual(len(seen), 1)
                self.assertEqual(app.session.last_request_messages, seen[0]["messages"])
                self.assertEqual(len(app.session.messages), 2)
                self.assertIn(app.active_profile.name, seen[0]["messages"][0]["content"])

    async def test_demo_uses_two_profiles_and_applies_second_automatically(self):
        prompts: list[list[dict[str, str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            prompts.append(json.loads(request.content)["messages"])
            return response(f"Ответ {len(prompts)}")

        with tempfile.TemporaryDirectory() as directory:
            app = ProfilesApp(CONFIG, data_dir=Path(directory), transport=httpx.MockTransport(handler))
            async with app.run_test(size=(190, 58)):
                app.action_start_demo()
                await app.workers.wait_for_complete()

                self.assertEqual(len(prompts), 3)
                self.assertIn("Краткий техлид", prompts[0][0]["content"])
                self.assertIn("Обучающий наставник", prompts[1][0]["content"])
                self.assertIn("Обучающий наставник", prompts[2][0]["content"])
                self.assertEqual(prompts[0][-1]["content"], CONTROL_QUESTION)
                self.assertEqual(prompts[1][-1]["content"], CONTROL_QUESTION)
                self.assertEqual(prompts[2][-1]["content"], FOLLOW_UP_QUESTION)
                self.assertEqual(app.active_profile.profile_id, "mentor")
                self.assertEqual(app.total_tokens, 75)

    async def test_delete_requires_confirmation_and_preserves_other_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ProfilesApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(190, 58)):
                self.assertEqual(app.active_profile.profile_id, "tech-lead")
                app.delete_profile()
                self.assertEqual(len(app.profiles), 2)
                app.delete_profile()
                self.assertEqual(len(app.profiles), 1)
                self.assertEqual(app.active_profile.profile_id, "mentor")
