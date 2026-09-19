from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from task_state import (
    ChatMessage,
    Event,
    Phase,
    PlanStep,
    Status,
    TaskState,
    TaskStore,
    TransitionController,
    TransitionError,
)


def planned_task() -> TaskState:
    task = TaskState("task-01", "Проверить управляемый процесс").checked()
    task.plan = [PlanStep("Анализ", "Уточнить требования"), PlanStep("Результат", "Подготовить ответ")]
    task.active_run.messages = [
        ChatMessage("user", "Составь план"),
        ChatMessage("assistant", "Предлагаю план из двух шагов"),
    ]
    task.active_run.turns = 1
    task.active_run.artifact = "1. Анализ\n2. Результат"
    return task.checked()


class TransitionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = TransitionController()

    def test_plan_artifact_does_not_advance_without_user_approval(self):
        task = planned_task()
        self.assertEqual(task.phase, Phase.PLANNING)
        self.assertEqual(task.active_run.turns, 1)
        self.controller.apply(task, Event.PLAN_APPROVED)
        self.assertEqual(task.phase, Phase.EXECUTION)

    def test_approval_without_stage_conversation_is_rejected_transactionally(self):
        task = TaskState("task-01", "Цель").checked()
        task.plan = [PlanStep("Шаг 1", "Сделать первое"), PlanStep("Шаг 2", "Сделать второе")]
        before = copy.deepcopy(task)
        with self.assertRaisesRegex(TransitionError, "сначала обсудите"):
            self.controller.apply(task, Event.PLAN_APPROVED)
        self.assertEqual(task, before)

    def test_full_route_requires_separate_user_events(self):
        task = planned_task()
        self.controller.apply(task, Event.PLAN_APPROVED)
        task.execution_result = "Готовый результат"
        task.active_run.messages = [
            ChatMessage("user", "Выполни план"),
            ChatMessage("assistant", "Результат подготовлен"),
        ]
        task.active_run.turns = 1
        self.assertEqual(task.phase, Phase.EXECUTION)
        self.controller.apply(task, Event.EXECUTION_SUBMITTED)
        self.assertEqual(task.phase, Phase.VALIDATION)
        task.validation_summary = "Нарушений нет"
        task.validation_passed = True
        self.assertEqual(task.phase, Phase.VALIDATION)
        self.controller.apply(task, Event.VALIDATION_ACCEPTED)
        self.assertEqual((task.phase, task.status, task.final_result), (Phase.DONE, Status.DONE, "Готовый результат"))

    def test_failed_validation_returns_through_revision(self):
        task = planned_task()
        self.controller.apply(task, Event.PLAN_APPROVED)
        task.execution_result = "Черновик"
        self.controller.apply(task, Event.EXECUTION_SUBMITTED)
        task.validation_summary = "Нужна доработка"
        task.validation_issues = ["Нет примера"]
        task.validation_passed = False
        before = copy.deepcopy(task)
        with self.assertRaisesRegex(TransitionError, "успешной validation"):
            self.controller.apply(task, Event.VALIDATION_ACCEPTED)
        self.assertEqual(task, before)
        self.controller.apply(task, Event.VALIDATION_SENT_TO_REVISION)
        self.assertEqual(task.phase, Phase.REVISION)
        task.execution_result = "Черновик с примером"
        task.active_run.messages = [
            ChatMessage("user", "Добавь пример"),
            ChatMessage("assistant", "Пример добавлен"),
        ]
        task.active_run.turns = 1
        self.controller.apply(task, Event.REVISION_SUBMITTED)
        self.assertEqual(task.phase, Phase.VALIDATION)
        self.assertEqual(task.revision_count, 1)
        self.assertIsNone(task.validation_passed)

    def test_return_creates_planning_visit_two_and_preserves_visit_one(self):
        task = planned_task()
        self.controller.apply(task, Event.PLAN_APPROVED)
        task.execution_result = "Устаревший результат"
        task.active_run.artifact = "Устаревший результат"
        self.controller.apply(task, Event.RETURNED_TO_PLANNING)
        self.assertEqual(task.phase, Phase.PLANNING)
        self.assertEqual([run.phase for run in task.runs], [Phase.PLANNING, Phase.EXECUTION, Phase.PLANNING])
        self.assertEqual(task.active_run.visit, 2)
        self.assertEqual(task.active_run.turns, 0)
        self.assertEqual(task.runs[0].messages[0].content, "Составь план")
        self.assertEqual(task.execution_result, "")
        self.assertEqual(task.runs[1].artifact, "Устаревший результат")

    def test_target_request_uses_graph_and_the_same_guards(self):
        empty = TaskState("task-02", "Цель").checked()
        with self.assertRaisesRegex(TransitionError, "сначала обсудите"):
            self.controller.request_target(empty, Phase.EXECUTION)
        self.assertEqual(empty.phase, Phase.PLANNING)
        task = planned_task()
        with self.assertRaisesRegex(TransitionError, "запрещён"):
            self.controller.request_target(task, Phase.DONE)
        self.assertEqual(task.phase, Phase.PLANNING)
        self.controller.request_target(task, Phase.EXECUTION)
        self.assertEqual(task.phase, Phase.EXECUTION)


class StoreTests(unittest.TestCase):
    def test_pause_restore_resume_preserves_stage_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = planned_task()
            store.save(task)
            store.pause(task)
            restored = store.load("task-01")
            self.assertEqual((restored.phase, restored.status), (Phase.PLANNING, Status.PAUSED))
            self.assertEqual(restored.active_run.turns, 1)
            self.assertEqual(len(restored.active_run.messages), 2)
            store.resume(restored)
            self.assertEqual((restored.phase, restored.status), (Phase.PLANNING, Status.READY))

    def test_corrupt_checkpoint_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            path = store.path("task-01")
            path.parent.mkdir(parents=True)
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaisesRegex(TransitionError, "повреждено"):
                store.load("task-01")
            self.assertEqual(path.read_text(encoding="utf-8"), "{bad json")


if __name__ == "__main__":
    unittest.main()
