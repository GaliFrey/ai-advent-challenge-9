"""Проверки конечного автомата и его хранилища."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from task_state import TaskError, TaskState, TaskStep, TaskStore


def execution_task(*, completed: int = 0) -> TaskState:
    steps = [TaskStep(f"Шаг {index}", f"Сделать {index}") for index in range(1, 4)]
    for step in steps[:completed]:
        step.status = "done"
        step.result = f"Результат {step.title}"
    phase = "validation" if completed == len(steps) else "execution"
    task = TaskState("task-01", "Проверяемая цель", "engineering", phase, "ready", completed, steps=steps)
    task.expected_action = task.derived_action()
    return task.checked()


class TaskStateTests(unittest.TestCase):
    def test_pause_and_resume_preserve_planning_checkpoint(self):
        task = TaskState("task-01", "Цель", "engineering")
        task.pause()
        self.assertEqual((task.phase, task.current_step, task.status), ("planning", 0, "paused"))
        self.assertIn("продолжить", task.expected_action)
        task.resume()
        self.assertEqual(task.expected_action, "составить план")

    def test_pause_and_resume_preserve_execution_checkpoint(self):
        task = execution_task(completed=1)
        task.pause()
        task.resume()
        self.assertEqual(task.phase, "execution")
        self.assertEqual(task.current_step, 1)
        self.assertIn("шаг 2", task.expected_action)
        self.assertEqual(task.steps[0].result, "Результат Шаг 1")

    def test_pause_and_resume_preserve_validation_checkpoint(self):
        task = execution_task(completed=3)
        task.phase = "validation"
        task = task.checked()
        task.pause()
        task.resume()
        self.assertEqual(task.phase, "validation")
        self.assertEqual(task.current_step, 3)
        self.assertEqual(task.expected_action, "проверить и собрать итог")

    def test_pause_and_resume_preserve_revision_checkpoint(self):
        task = execution_task(completed=3)
        task.phase = "revision"
        task.validation = "Нужна доработка"
        task.final_result = "Предварительный результат"
        task.validation_passed = False
        task.validation_issues = ["Нет примера"]
        task.revision_instruction = "Добавить пример"
        task = task.checked()
        task.pause()
        task.resume()
        self.assertEqual(task.phase, "revision")
        self.assertEqual(task.current_step, 3)
        self.assertEqual(task.expected_action, "доработать результат по замечаниям")

    def test_second_revision_checkpoint_is_valid(self):
        task = execution_task(completed=3)
        task.phase = "revision"
        task.validation = "Первая доработка не решила проблему"
        task.final_result = "Первая исправленная версия"
        task.validation_passed = False
        task.validation_issues = ["Пример всё ещё неточен"]
        task.revision_instruction = "Исправить пример повторно"
        task.revision_result = "Первая исправленная версия"
        task.revision_count = 1
        checked = task.checked()
        self.assertEqual(checked.revision_count, 1)
        self.assertEqual(checked.phase, "revision")

    def test_inconsistent_checkpoint_is_rejected(self):
        task = execution_task(completed=1)
        task.current_step = 2
        with self.assertRaisesRegex(ValueError, "Текущий шаг"):
            task.checked()

    def test_done_task_cannot_be_paused(self):
        task = execution_task(completed=3)
        task.phase = "done"
        task.status = "done"
        task.final_result = "Итог"
        task.validation_passed = True
        task = task.checked()
        with self.assertRaisesRegex(TaskError, "нельзя"):
            task.pause()


class TaskStoreTests(unittest.TestCase):
    def test_round_trip_preserves_checkpoint_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = execution_task(completed=1)
            task.pause()
            store.save(task)
            loaded = store.load("task-01")
            self.assertEqual(loaded.profile_id, "engineering")
            self.assertEqual(loaded.status, "paused")
            self.assertEqual(loaded.current_step, 1)
            self.assertEqual(loaded.steps[0].result, "Результат Шаг 1")

    def test_corrupted_task_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            path = store.path("task-01")
            path.parent.mkdir(parents=True)
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaisesRegex(TaskError, "Повреждён"):
                store.load("task-01")
            self.assertEqual(path.read_text(encoding="utf-8"), "{bad json")

    def test_saved_expected_action_is_derived_from_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель", "explainer")
            raw = json.loads(store.path(task.task_id).read_text(encoding="utf-8"))
            self.assertEqual(raw["expected_action"], "составить план")
