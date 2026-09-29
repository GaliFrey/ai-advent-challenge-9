"""Headless checks of the live comparison and the full demo."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Select, Static

from tui import RagApp


QUESTIONS = [{"id": f"q{i:02}", "question": f"Question {i}",
              "expected": [f"Fact {i}"], "sources": [f"day-{i:02}/README.md"]}
             for i in range(1, 11)]


class FakeRetriever:
    metadata = {"document_count": "21", "corpus_sha256": "test"}


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def wait_finished(self, app):
        for _ in range(100):
            if app.report and app.report["status"] in ("complete", "failed") and not app.running:
                await app.workers.wait_for_complete()
                return
            await asyncio.sleep(0.02)
        self.fail("TUI run did not complete")

    def fake_answer(self, question, mode, *, key, model, retriever):
        chunks = [] if mode == "plain" else [{
            "source": "day-07/README.md", "section": "Проверка", "text": f"Exact context: {question}",
            "chunk_id": "fake-chunk", "score": 0.5}]
        return {"answer": f"{mode}: {question}" + (" [S1]" if chunks else ""),
                "usage": {"total_tokens": 10}, "llm_seconds": 0.1, "search_seconds": 0.01,
                "chunks": chunks, "citation_check": {"unknown_references": []},
                "review": None}

    async def test_selected_question_runs_live_and_saves_both_modes(self):
        calls = []

        def ask(*args, **kwargs):
            calls.append((args[0], args[1]))
            return self.fake_answer(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                patch("tui.DAY", Path(directory)), \
                patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network")):
            app = RagApp(questions=QUESTIONS, ask=ask, retriever_factory=FakeRetriever, key="very-secret-test-key")
            async with app.run_test(size=(120, 45)) as pilot:
                self.assertIn("Ответ появится", str(app.query_one("#plain-answer", Static).content))
                app.query_one("#question-select", Select).value = "q07"
                await pilot.pause()
                await pilot.click("#ask")
                await self.wait_finished(app)
                await pilot.pause()
                self.assertEqual(calls, [("Question 7", "plain"), ("Question 7", "rag")])
                self.assertIn("rag: Question 7", str(app.query_one("#rag-answer", Static).content))
                self.assertIn("Exact context: Question 7", str(app.query_one("#chunk-text", Static).content))
                self.assertNotIn("/2", str(app.query_one("#rag-meta", Static).content))
                app.query_one("#question-select", Select).value = "q08"
                await pilot.pause()
                self.assertIn("Ответ появится", str(app.query_one("#rag-answer", Static).content))
                app.query_one("#question-select", Select).value = "q07"
                await pilot.pause()
                self.assertIn("plain: Question 7", str(app.query_one("#plain-answer", Static).content))
            saved = json.loads(app.report_path.read_text())
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(len(saved["items"]), 1)
            self.assertEqual(saved["items"][0]["results"]["rag"]["chunks"][0]["text"],
                             "Exact context: Question 7")
            self.assertNotIn("very-secret-test-key", app.report_path.read_text())
            records = [json.loads(line) for line in app.session_log.path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in records if row["event"] == "request_ok"],
                             ["request_ok", "request_ok"])
            self.assertNotIn("very-secret-test-key", app.session_log.path.read_text())
            self.assertNotIn("Question 7", app.session_log.path.read_text())

    async def test_demo_runs_all_ten_questions_and_keeps_results(self):
        calls = []

        def ask(*args, **kwargs):
            calls.append((args[0], args[1]))
            return self.fake_answer(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch("tui.DAY", Path(directory)):
            app = RagApp(questions=QUESTIONS, ask=ask, retriever_factory=FakeRetriever, key="secret-key")
            async with app.run_test(size=(120, 45)) as pilot:
                await pilot.click("#demo")
                await self.wait_finished(app)
                await pilot.pause()
                self.assertEqual(len(calls), 20)
                self.assertEqual(calls[0], ("Question 1", "plain"))
                self.assertEqual(calls[-1], ("Question 10", "rag"))
                self.assertIn("20/20", str(app.query_one("#run-status", Static).content))
                app.query_one("#question-select", Select).value = "q03"
                await pilot.pause()
                self.assertIn("rag: Question 3", str(app.query_one("#rag-answer", Static).content))
            saved = json.loads(app.report_path.read_text())
            self.assertEqual(len(saved["items"]), 10)
            self.assertTrue(all(set(item["results"]) == {"plain", "rag"} for item in saved["items"]))
            self.assertNotIn("secret-key", app.report_path.read_text())

    async def test_failure_preserves_partial_run_and_reports_error(self):
        attempts = 0

        def ask(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError("Provider unavailable")
            return self.fake_answer(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch("tui.DAY", Path(directory)):
            app = RagApp(questions=QUESTIONS, ask=ask, retriever_factory=FakeRetriever, key="fake")
            async with app.run_test(size=(100, 35)) as pilot:
                await pilot.click("#ask")
                await self.wait_finished(app)
                await pilot.pause()
                self.assertIn("Остановлено", str(app.query_one("#run-status", Static).content))
                self.assertIn("plain: Question 1", str(app.query_one("#plain-answer", Static).content))
            saved = json.loads(app.report_path.read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertIn("RuntimeError на этапе request", saved["items"][0]["results"]["rag"]["error"])
            records = [json.loads(line) for line in app.session_log.path.read_text().splitlines()]
            self.assertEqual(records[-1]["event"], "run_failed")
            self.assertEqual(records[-1]["question_id"], "q01")
            self.assertEqual(records[-1]["mode"], "rag")

    async def test_model_setup_error_is_recorded_without_provider_text(self):
        def broken_model():
            raise RuntimeError("huggingface.co failure; secret=topsecret")

        with tempfile.TemporaryDirectory() as directory, patch("tui.DAY", Path(directory)):
            app = RagApp(questions=QUESTIONS, retriever_factory=broken_model, key="secret-key")
            async with app.run_test(size=(120, 45)) as pilot:
                await pilot.click("#ask")
                await self.wait_finished(app)
                await pilot.pause()
                self.assertIn("Журнал:", str(app.query_one("#run-status", Static).content))
                self.assertIn("Модель эмбеддингов недоступна", str(app.query_one("#run-status", Static).content))
            saved = json.loads(app.report_path.read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error"]["stage"], "model")
            records = [json.loads(line) for line in app.session_log.path.read_text().splitlines()]
            self.assertEqual(records[-1]["error_code"], "MODEL_NETWORK")
            self.assertTrue(records[-1]["frames"])
            self.assertNotIn("topsecret", app.session_log.path.read_text() + app.report_path.read_text())
            self.assertNotIn("secret-key", app.session_log.path.read_text() + app.report_path.read_text())


if __name__ == "__main__":
    unittest.main()
