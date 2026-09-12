"""Headless-проверки четырёхвкладочного TUI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import Button, Input, RichLog, Select, Static, TabPane

from agent import AgentConfig
from demo import CHECK_MESSAGE
from main import ContextStrategiesApp
from memory import BranchingMemory


CONFIG = AgentConfig(api_key="fake-key", model="deepseek-v4-flash")
ANSWER_A = (
    "ПРОЕКТ=FORUM-731; ДАТА=26.10.2026; БЮДЖЕТ=510000; ПОДРЯДЧИК=Меридиан; "
    "ДОСТУПНОСТЬ=безбарьерный вход; ИНТЕРНЕТ=две независимые линии интернета; "
    "ОТКРЫТЫЙ_ВОПРОС=выбор кейтеринга"
)
ANSWER_B = (
    "ПРОЕКТ=FORUM-731; ДАТА=02.11.2026; БЮДЖЕТ=560000; ПОДРЯДЧИК=Вектор; "
    "ДОСТУПНОСТЬ=безбарьерный вход; ИНТЕРНЕТ=две независимые линии интернета; "
    "ОТКРЫТЫЙ_ВОПРОС=выбор ведущего"
)


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


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_layout_has_four_tabs_and_two_panels_per_strategy(self):
        app = ContextStrategiesApp(CONFIG)
        async with app.run_test(size=(180, 52)):
            self.assertEqual(set(app.agents), {"sliding", "facts", "branching"})
            self.assertEqual(len(app.query(TabPane)), 4)
            self.assertEqual(len(app.query(RichLog)), 6)
            self.assertEqual(len(app.query(Input)), 3)
            self.assertEqual(len(app.query(Select)), 2)
            self.assertIn(
                "Общий расход: 0",
                str(app.query_one("#stats-facts", Static).content),
            )

    async def test_each_tab_has_an_independent_session(self):
        app = ContextStrategiesApp(
            CONFIG,
            transport=httpx.MockTransport(lambda request: response("ПРИНЯТО")),
        )
        async with app.run_test(size=(180, 52)):
            app.query_one("#input-sliding", Input).value = "Только sliding"
            app.start_message("sliding")
            await app.workers.wait_for_complete()

            self.assertEqual(app.agents["sliding"].memory.total_message_count, 2)
            self.assertEqual(app.agents["facts"].memory.total_message_count, 0)
            self.assertEqual(app.agents["branching"].memory.total_message_count, 0)
            self.assertTrue(app.query_one("#model-select", Select).disabled)

    async def test_checkpoint_buttons_switch_independent_branches(self):
        app = ContextStrategiesApp(
            CONFIG,
            transport=httpx.MockTransport(lambda request: response("ПРИНЯТО")),
        )
        async with app.run_test(size=(180, 52)):
            app.query_one("#input-branching", Input).value = "Общее"
            app.start_message("branching")
            await app.workers.wait_for_complete()
            app.create_checkpoint()
            app.query_one("#input-branching", Input).value = "Только A"
            app.start_message("branching")
            await app.workers.wait_for_complete()
            app.switch_branch("B")
            app.query_one("#input-branching", Input).value = "Только B"
            app.start_message("branching")
            await app.workers.wait_for_complete()

            memory = app.agents["branching"].memory
            self.assertIsInstance(memory, BranchingMemory)
            self.assertEqual(memory.active_branch, "B")
            contents = [item["content"] for item in memory.context_messages]
            self.assertIn("Только B", contents)
            self.assertNotIn("Только A", contents)

    async def test_demo_uses_49_calls_and_saves_complete_report(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = json.loads(request.content)
            system = payload["messages"][0]["content"]
            if system.startswith("Ты обновляешь"):
                return response(
                    json.dumps(
                        {
                            "project": "FORUM-731",
                            "date": "26.10.2026",
                            "budget": 510000,
                            "contractor": "Меридиан",
                            "accessibility": "безбарьерный вход",
                            "internet": "две независимые линии интернета",
                            "open_question": "выбор кейтеринга",
                        },
                        ensure_ascii=False,
                    )
                )
            if payload["messages"][-1]["content"] == CHECK_MESSAGE:
                whole_prompt = json.dumps(payload["messages"], ensure_ascii=False)
                return response(ANSWER_B if "02.11.2026" in whole_prompt else ANSWER_A, 50, 15)
            return response("ПРИНЯТО")

        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "results"
            app = ContextStrategiesApp(
                CONFIG,
                transport=httpx.MockTransport(handler),
                report_dir=report_dir,
            )
            async with app.run_test(size=(180, 52)):
                app.action_start_demo()
                await app.workers.wait_for_complete()

                self.assertEqual(calls, 49)
                self.assertEqual(app.quality["facts"].score, 7)
                self.assertEqual(app.quality["branch_A"].score, 7)
                self.assertEqual(app.quality["branch_B"].score, 7)
                report_path = next(report_dir.glob("demo-*.json"))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["outcome"], "completed")
                self.assertEqual(report["actual_api_attempts"], 49)
                self.assertEqual(len(report["turns"]), 27)
                self.assertEqual(report["branch_tokens"], {"common": 90, "A": 125, "B": 125})
                self.assertNotIn("fake-key", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
