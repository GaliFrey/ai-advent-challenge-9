"""Headless-проверки интерфейса и демонстрационного checkpoint."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import Button, Input, RichLog, Select

from agent import AgentConfig
from main import TaskMachineApp
from test_agent import PLAN, response


CONFIG = AgentConfig(api_key="fake-key", openrouter_api_key="fake-openrouter-key")


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_layout_exposes_stage_chain_chat_and_technical_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            app = TaskMachineApp(CONFIG, data_dir=Path(directory))
            async with app.run_test(size=(190, 55)) as pilot:
                self.assertEqual(len(app.query(RichLog)), 2)
                self.assertEqual(len(app.query(Select)), 3)
                self.assertEqual(len(app.query(Input)), 1)
                self.assertEqual(len(app.query("#stats")), 0)
                self.assertIn("openai/gpt-5.6-sol", str(app.query_one("#model-map").content))
                self.assertIn("deepseek-v4-pro", str(app.query_one("#model-map").content))
                ids = {button.id for button in app.query(Button)}
                self.assertTrue({"new-task", "run", "pause", "resume", "demo"} <= ids)
                self.assertTrue(
                    {"stage-planning", "stage-execution", "stage-validation", "stage-revision", "stage-done"} <= ids
                )
                configuration_y = app.query_one("#model", Select).region.y
                self.assertEqual(app.query_one("#configuration").region.height, 3)
                self.assertEqual(app.query_one("#configuration").content_region.height, 1)
                for selector in ("#profile", "#task", "#new-task", "#demo"):
                    self.assertEqual(app.query_one(selector).region.y, configuration_y)
                    self.assertEqual(app.query_one(selector).region.height, 1)
                self.assertEqual(app.query_one("#model", Select).region.height, 1)
                goal_y = app.query_one("#goal", Input).region.y
                self.assertEqual(app.query_one("#run", Button).region.y, goal_y)
                self.assertEqual(app.query_one("#pause", Button).region.y, goal_y)
                self.assertEqual(app.query_one("#resume", Button).region.y, goal_y)
                self.assertGreaterEqual(
                    app.query_one("#goal-bar").content_region.height,
                    app.query_one("#goal", Input).region.height,
                )
                self.assertGreaterEqual(
                    app.query_one("#stage-chain").content_region.height,
                    app.query_one("#stage-planning", Button).region.height,
                )
                self.assertTrue(app.query_one("#stage-planning", Button).has_class("active"))
                self.assertTrue(app.query_one("#stage-planning", Button).has_class("viewing"))
                self.assertIn("ПЛАН", str(app.query_one("#stage-planning", Button).label))
                await pilot.click("#stage-execution")
                self.assertEqual(app.selected_phase, "execution")
                self.assertTrue(app.query_one("#stage-execution", Button).has_class("viewing"))

    async def test_inflight_request_is_visible_before_response(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(_: httpx.Request) -> httpx.Response:
            started.set()
            await release.wait()
            return response(PLAN)

        with tempfile.TemporaryDirectory() as directory:
            app = TaskMachineApp(CONFIG, data_dir=Path(directory), transport=httpx.MockTransport(handler))
            async with app.run_test(size=(190, 55)) as pilot:
                app.start_run()
                await asyncio.wait_for(started.wait(), timeout=2)
                await pilot.pause()

                chat_text = "\n".join(line.text for line in app.query_one("#stage-chat", RichLog).lines)
                self.assertIn("ЗАПРОС ОТПРАВЛЕН", chat_text)
                self.assertIn("работает", chat_text)
                self.assertEqual(str(app.query_one("#prompt-title").content), "PROMPT · ОТПРАВЛЕН")
                self.assertIn("API:          выполняется", str(app.query_one("#state").content))

                app.pause_task()
                release.set()
                await app.workers.wait_for_complete()
                self.assertEqual(app.task_state.status, "paused")

    async def test_demo_saves_pause_then_resume_finishes_without_new_goal(self):
        prompts: list[list[dict[str, str]]] = []
        replies = iter(
            (
                PLAN,
                "Часть 1",
                "Часть 2",
                "Часть 3",
                json.dumps(
                    {
                        "summary": "Проверено",
                        "issues": [],
                        "improvements": [{"area": "Подача", "suggestion": "Добавить вывод"}],
                        "revision_instruction": "",
                        "final_result": "Итог",
                    },
                    ensure_ascii=False,
                ),
            )
        )

        def handler(request: httpx.Request) -> httpx.Response:
            prompts.append(json.loads(request.content)["messages"])
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            app = TaskMachineApp(CONFIG, data_dir=Path(directory), transport=httpx.MockTransport(handler))
            async with app.run_test(size=(190, 55)):
                app.action_start_demo()
                await app.workers.wait_for_complete()
                self.assertEqual(app.task_state.status, "paused")
                self.assertEqual(app.task_state.phase, "execution")
                self.assertEqual(app.task_state.current_step, 1)

                app.resume_task()
                await app.workers.wait_for_complete()
                self.assertEqual(app.task_state.phase, "done")
                self.assertEqual(app.task_state.final_result, "Итог")
                self.assertEqual(len(prompts), 5)
                self.assertIn("Часть 1", prompts[2][-1]["content"])
                self.assertEqual(app.selected_phase, "done")
                self.assertTrue(app.query_one("#stage-done", Button).has_class("active"))
                state_text = str(app.query_one("#state").content)
                self.assertIn("Токенов:      125", state_text)
                self.assertGreaterEqual(app.query_one("#state").content_region.height, 11)
                chat_text = "\n".join(line.text for line in app.query_one("#stage-chat", RichLog).lines)
                self.assertIn("ИТОГОВЫЙ РЕЗУЛЬТАТ", chat_text)
                self.assertIn("Итог", chat_text)
