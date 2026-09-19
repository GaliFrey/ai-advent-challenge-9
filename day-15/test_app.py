from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import Button, Input, LoadingIndicator, RichLog, Select, TabbedContent

from agent import AgentConfig
from main import ControlledTransitionsApp
from task_state import Phase


def plan_response() -> httpx.Response:
    content = json.dumps(
        {
            "reply": "План готов к обсуждению",
            "steps": [
                {"title": "Анализ", "instruction": "Уточнить требования"},
                {"title": "Результат", "instruction": "Подготовить ответ"},
            ],
        },
        ensure_ascii=False,
    )
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def execution_response() -> httpx.Response:
    content = json.dumps(
        {"reply": "Результат готов", "artifact": "Полная реализация плана"},
        ensure_ascii=False,
    )
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        },
    )


def validation_response() -> httpx.Response:
    content = json.dumps(
        {"reply": "Проверка завершена", "summary": "Результат соответствует плану", "issues": []},
        ensure_ascii=False,
    )
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39},
        },
    )


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_selector_switches_between_saved_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
                app = ControlledTransitionsApp(Path(directory), AgentConfig(api_key="fake"), client)
                async with app.run_test(size=(170, 52)) as pilot:
                    goal = app.query_one("#goal", Input)
                    goal.value = "Первая задача"
                    await pilot.click("#new")
                    await pilot.pause()
                    self.assertEqual(app.task_state.task_id, "task-01")
                    app.new_task()
                    self.assertTrue(app.creating_task)
                    self.assertFalse(goal.disabled)
                    goal.value = "Вторая задача"
                    app.new_task()
                    await pilot.pause()
                    self.assertEqual(app.task_state.task_id, "task-02")
                    self.assertTrue(goal.disabled)
                    app.new_task()
                    goal.value = "Черновик третьей задачи"
                    app.cancel_new_task()
                    self.assertFalse(app.creating_task)
                    self.assertEqual(goal.value, "Вторая задача")
                    selector = app.query_one("#task-select", Select)
                    selector.value = "task-01"
                    await pilot.pause()
                    self.assertEqual(app.task_state.task_id, "task-01")
                    self.assertEqual(goal.value, "Первая задача")

    async def test_planning_reply_stays_in_stage_until_approval_button(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = iter((plan_response(),))
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: next(responses))) as client:
                app = ControlledTransitionsApp(Path(directory), AgentConfig(api_key="fake"), client)
                async with app.run_test(size=(170, 52)) as pilot:
                    await pilot.click("#new")
                    self.assertIsInstance(app.query_one("#state"), RichLog)
                    self.assertIsInstance(app.query_one("#artifact"), RichLog)
                    self.assertIsInstance(app.query_one("#prompt"), RichLog)
                    self.assertIsInstance(app.query_one("#transitions"), RichLog)
                    self.assertIsInstance(app.query_one("#run-route"), RichLog)
                    self.assertIsInstance(app.query_one("#details-tabs"), TabbedContent)
                    self.assertIsInstance(app.query_one("#api-loading"), LoadingIndicator)
                    self.assertTrue(app.query_one("#primary-transition", Button).disabled)
                    self.assertEqual(
                        app.query_one("#pause", Button).region.y,
                        app.query_one("#primary-transition", Button).region.y,
                    )
                    self.assertIn(("ctrl+q", "quit", "Quit"), app.BINDINGS)
                    self.assertIn(("ctrl+enter", "send_stage_message", "Send"), app.BINDINGS)
                    self.assertIn("план", app.query_one("#message", Input).placeholder.lower())
                    app.query_one("#message", Input).value = "Составь план"
                    await pilot.click("#send")
                    await pilot.pause()
                    self.assertEqual(app.task_state.phase, Phase.PLANNING)
                    self.assertEqual(app.task_state.active_run.turns, 1)
                    self.assertFalse(app.query_one("#primary-transition", Button).disabled)
                    await app.primary_transition()
                    self.assertEqual(app.task_state.phase, Phase.EXECUTION)
                    self.assertEqual(app.task_state.active_run.turns, 0)
                    self.assertEqual(app.task_state.execution_result, "")
                    self.assertTrue(app.query_one("#primary-transition", Button).disabled)

    async def test_enter_submits_and_pending_message_is_visible_before_response(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(_: httpx.Request) -> httpx.Response:
            started.set()
            await release.wait()
            return plan_response()

        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                app = ControlledTransitionsApp(Path(directory), AgentConfig(api_key="fake"), client)
                async with app.run_test(size=(170, 52)) as pilot:
                    await pilot.click("#new")
                    message = app.query_one("#message", Input)
                    message.value = "Сообщение через Enter"
                    submit = asyncio.create_task(app.send_message())
                    await asyncio.wait_for(started.wait(), timeout=1)
                    self.assertEqual(app.pending_message, "Сообщение через Enter")
                    rendered = "\n".join(line.text for line in app.query_one("#chat", RichLog).lines)
                    self.assertIn("Сообщение через Enter", rendered)
                    self.assertIn("Модель работает", rendered)
                    release.set()
                    await submit
                    self.assertEqual(app.task_state.active_run.turns, 1)
                    self.assertEqual(message.value, "")

                    message.value = "Второе сообщение через Enter"
                    message.focus()
                    await pilot.press("enter")
                    for _ in range(50):
                        if app.task_state.active_run.turns == 2:
                            break
                        await asyncio.sleep(0.01)
                    self.assertEqual(app.task_state.active_run.turns, 2)

    async def test_target_selector_records_rejections_and_preserves_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
                app = ControlledTransitionsApp(Path(directory), AgentConfig(api_key="fake"), client)
                async with app.run_test(size=(170, 52)) as pilot:
                    await pilot.click("#new")
                    selector = app.query_one("#target-phase", Select)
                    selector.value = Phase.VALIDATION.value
                    await pilot.pause()
                    await app.request_transition()
                    selector.value = Phase.DONE.value
                    await pilot.pause()
                    await app.request_transition()
                    self.assertEqual(app.task_state.phase, Phase.PLANNING)
                    rejected = [item for item in app.store.events(app.task_state.task_id) if item["event"] == "transition_rejected"]
                    self.assertEqual(len(rejected), 2)

    async def test_full_ui_route_renders_done_without_active_run(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = iter((plan_response(), execution_response(), validation_response()))
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: next(responses))) as client:
                app = ControlledTransitionsApp(Path(directory), AgentConfig(api_key="fake"), client)
                async with app.run_test(size=(170, 52)):
                    app.new_task()
                    app.query_one("#message", Input).value = "Составь план"
                    await app.send_message()
                    await app.primary_transition()
                    self.assertEqual(app.task_state.phase, Phase.EXECUTION)
                    app.query_one("#message", Input).value = "Выполни утверждённый план"
                    await app.send_message()
                    await app.primary_transition()
                    self.assertEqual(app.task_state.phase, Phase.VALIDATION)
                    app.query_one("#message", Input).value = "Проверь результат"
                    await app.send_message()
                    await app.primary_transition()
                    self.assertEqual(app.task_state.phase, Phase.DONE)
                    self.assertTrue(app.query_one("#primary-transition", Button).disabled)

if __name__ == "__main__":
    unittest.main()
