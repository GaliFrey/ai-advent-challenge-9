"""Headless-проверки двухпанельного TUI без внешнего API."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import Button, Input, RichLog, Select, Static

from agent import AgentConfig
from demo import DEMO_CHECK_MESSAGE
from main import ContextComparisonApp


CONFIG = AgentConfig(api_key="fake-key", model="deepseek-v4-flash")


def response(text: str, input_tokens: int = 12, output_tokens: int = 3) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    })


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_layout_has_shared_input_two_panels_and_configuration(self):
        app = ContextComparisonApp(CONFIG)
        async with app.run_test(size=(160, 46)):
            self.assertEqual(set(app.agents), {"full", "summary"})
            self.assertEqual(len(app.query(RichLog)), 2)
            self.assertEqual(len(app.query(Input)), 1)
            self.assertEqual(len(app.query(Select)), 2)
            self.assertEqual(len(app.query(Button)), 4)
            self.assertEqual(app.agents["summary"].memory.keep_recent, 4)
            self.assertIn(
                "Общий расход: 0",
                str(app.query_one("#stats-summary", Static).content),
            )

    async def test_one_input_is_sent_to_both_agents(self):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return response("Одинаковый ответ")

        app = ContextComparisonApp(CONFIG, transport=httpx.MockTransport(handler))
        async with app.run_test(size=(160, 46)):
            field = app.query_one("#message-input", Input)
            field.value = "Общий вопрос"
            app.start_message()
            await app.workers.wait_for_complete()

            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["messages"][-1]["content"], "Общий вопрос")
            self.assertEqual(requests[1]["messages"][-1]["content"], "Общий вопрос")
            self.assertEqual(app.agents["full"].memory.total_message_count, 2)
            self.assertEqual(app.agents["summary"].memory.total_message_count, 2)
            self.assertTrue(app.query_one("#model-select", Select).disabled)
            self.assertTrue(app.query_one("#recent-select", Select).disabled)

    async def test_fifth_exchange_creates_summary_only_on_right(self):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if payload["messages"][0]["content"].startswith("Ты сжимаешь"):
                return response("Сжатые факты", 30, 5)
            return response("ПРИНЯТО", 12, 3)

        app = ContextComparisonApp(CONFIG, transport=httpx.MockTransport(handler))
        async with app.run_test(size=(160, 46)):
            for index in range(5):
                app.query_one("#message-input", Input).value = f"Сообщение {index}"
                app.start_message()
                await app.workers.wait_for_complete()

            self.assertEqual(len(requests), 11)
            self.assertEqual(app.agents["full"].stats.successful_summary_requests, 0)
            self.assertEqual(app.agents["summary"].stats.successful_summary_requests, 1)
            self.assertEqual(app.agents["summary"].memory.raw_message_count, 4)
            self.assertEqual(app.agents["summary"].memory.summarized_message_count, 6)
            self.assertEqual(app.agents["summary"].memory.messages_until_summary, 10)
            stats = str(app.query_one("#stats-summary", Static).content)
            self.assertIn("summary v1", stats)
            self.assertIn("Суммаризация: 1/1", stats)
            self.assertIn("следующее через 10 сообщ.", stats)

    async def test_clear_requires_confirmation_and_unlocks_configuration(self):
        app = ContextComparisonApp(
            CONFIG,
            transport=httpx.MockTransport(lambda request: response("Ответ")),
        )
        async with app.run_test(size=(160, 46)):
            app.query_one("#message-input", Input).value = "Вопрос"
            app.start_message()
            await app.workers.wait_for_complete()

            app.confirm_or_clear()
            self.assertTrue(app.clear_armed)
            self.assertEqual(app.agents["full"].memory.total_message_count, 2)
            app.confirm_or_clear()

            self.assertEqual(app.agents["full"].memory.total_message_count, 0)
            self.assertEqual(app.agents["summary"].memory.total_message_count, 0)
            self.assertFalse(app.query_one("#model-select", Select).disabled)
            self.assertFalse(app.query_one("#recent-select", Select).disabled)

    async def test_demo_refuses_nonempty_history(self):
        app = ContextComparisonApp(
            CONFIG,
            transport=httpx.MockTransport(lambda request: response("Ответ")),
        )
        async with app.run_test(size=(160, 46)):
            app.query_one("#message-input", Input).value = "Вопрос"
            app.start_message()
            await app.workers.wait_for_complete()
            app.action_start_demo()

            status = str(app.query_one("#status", Static).content)
            self.assertIn("сначала очистите", status)

    async def test_demo_runs_same_scenario_and_reports_quality(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload = json.loads(request.content)
            system = payload["messages"][0]["content"]
            if system.startswith("Ты сжимаешь"):
                return response(
                    "Проект ORBITA-7429. Актуальны дата 26.10.2026, бюджет 510000, "
                    "подрядчик Меридиан. Открыт выбор кейтеринга.",
                    40,
                    10,
                )
            if payload["messages"][-1]["content"] == DEMO_CHECK_MESSAGE:
                return response(
                    "ПРОЕКТ=ORBITA-7429; ДАТА=26.10.2026; БЮДЖЕТ=510000; "
                    "ПОДРЯДЧИК=Меридиан; ОТКРЫТЫЙ_ВОПРОС=выбор кейтеринга",
                    50,
                    15,
                )
            return response("ПРИНЯТО")

        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "results"
            app = ContextComparisonApp(
                CONFIG,
                transport=httpx.MockTransport(handler),
                report_dir=report_dir,
            )
            async with app.run_test(size=(160, 46)):
                app.action_start_demo()
                await app.workers.wait_for_complete()

                self.assertEqual(calls, 35)
                self.assertEqual(app.quality["full"].score, 5)
                self.assertEqual(app.quality["summary"].score, 5)
                self.assertEqual(app.agents["summary"].stats.successful_summary_requests, 3)
                self.assertIn(
                    "Качество: 5/5",
                    str(app.query_one("#stats-summary", Static).content),
                )
                reports = list(report_dir.glob("demo-*.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                self.assertEqual(report["outcome"], "completed")
                self.assertEqual(report["actual_api_attempts"], 35)
                self.assertEqual(len(report["turns"]), 16)
                self.assertEqual(report["agents"]["summary"]["quality"]["score"], 5)
                self.assertNotIn("fake-key", reports[0].read_text(encoding="utf-8"))

    async def test_report_keeps_repeated_summary_errors(self):
        summary_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal summary_calls
            payload = json.loads(request.content)
            system = payload["messages"][0]["content"]
            if system.startswith("Ты сжимаешь"):
                summary_calls += 1
                if summary_calls >= 3:
                    return httpx.Response(200, json={
                        "choices": [{
                            "message": {"content": "Обрезанное summary"},
                            "finish_reason": "length",
                        }],
                        "usage": {
                            "prompt_tokens": 40,
                            "completion_tokens": 350,
                            "total_tokens": 390,
                        },
                    })
                return response("Краткое актуальное summary", 40, 8)
            if payload["messages"][-1]["content"] == DEMO_CHECK_MESSAGE:
                return response(
                    "ПРОЕКТ=ORBITA-7429; ДАТА=26.10.2026; БЮДЖЕТ=510000; "
                    "ПОДРЯДЧИК=Меридиан; ОТКРЫТЫЙ_ВОПРОС=выбор кейтеринга",
                    50,
                    15,
                )
            return response("ПРИНЯТО")

        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "results"
            app = ContextComparisonApp(
                CONFIG,
                transport=httpx.MockTransport(handler),
                report_dir=report_dir,
            )
            async with app.run_test(size=(160, 46)):
                app.action_start_demo()
                await app.workers.wait_for_complete()

            report_path = next(report_dir.glob("demo-*.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            errors = [
                result["compression_error"]
                for turn in report["turns"]
                for result in turn["results"].values()
                if result.get("compression_error")
            ]
            self.assertEqual(report["outcome"], "completed_with_summary_errors")
            self.assertEqual(report["actual_api_attempts"], 36)
            self.assertEqual(len(errors), 2)
            self.assertTrue(all("finish_reason: length" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
