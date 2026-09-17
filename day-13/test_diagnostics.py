"""Проверки append-only журнала и отчёта по прогону."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from diagnostics import DiagnosticError
from report import render
from task_state import TaskStore


class DiagnosticsTests(unittest.TestCase):
    def test_lifecycle_events_are_append_only_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель", "engineering")
            store.pause(task)
            store.resume(task)
            store.log_event(
                task,
                "stage_completed",
                model="test-model",
                request_messages=[{"role": "user", "content": "Запрос"}],
                response="Ответ",
                usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                elapsed_seconds=1.25,
            )

            events = store.run_log.read("task-01")
            self.assertEqual([item["sequence"] for item in events], [1, 2, 3, 4])
            self.assertEqual([item["event"] for item in events[:3]], ["task_created", "paused", "resumed"])
            self.assertEqual(events[0]["details"]["goal"], "Цель")
            self.assertEqual(events[-1]["request_messages"][0]["content"], "Запрос")
            self.assertEqual(events[-1]["response"], "Ответ")
            report = render("task-01", store.run_log)
            self.assertIn("API-вызовов: 1", report)
            self.assertIn("input 7 + output 3 = 10", report)
            self.assertIn("planning", report)

    def test_corrupted_event_log_is_reported_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            path = store.run_log.path("task-01")
            path.parent.mkdir(parents=True)
            path.write_text("{bad json\n", encoding="utf-8")
            with self.assertRaisesRegex(DiagnosticError, "Повреждён"):
                store.run_log.read("task-01")
            self.assertEqual(path.read_text(encoding="utf-8"), "{bad json\n")
