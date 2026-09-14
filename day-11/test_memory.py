"""Проверки изоляции и жизненного цикла слоёв памяти."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memory import MemoryError, MemoryLayers


class MemoryLayersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.memory = MemoryLayers(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_three_layers_are_saved_to_separate_scopes(self):
        self.memory.commit_exchange("Вопрос", "Ответ")
        self.memory.add_working("Ограничение задачи")
        self.memory.add_long_term("Предпочтение пользователя")

        self.assertTrue((self.root / "short_term" / "task-01" / "session-01.json").exists())
        self.assertTrue((self.root / "working" / "task-01.json").exists())
        self.assertTrue((self.root / "long_term" / "demo.json").exists())
        self.assertEqual(len(self.memory.short_term), 2)
        self.assertEqual(self.memory.working, ("Ограничение задачи",))
        self.assertEqual(self.memory.long_term, ("Предпочтение пользователя",))

    def test_new_session_changes_only_short_term(self):
        self.memory.commit_exchange("Вопрос", "Ответ")
        self.memory.add_working("Рабочая")
        self.memory.add_long_term("Долгая")

        self.memory.new_session("session-02")

        self.assertEqual(self.memory.short_term, ())
        self.assertEqual(self.memory.working, ("Рабочая",))
        self.assertEqual(self.memory.long_term, ("Долгая",))

    def test_switch_task_gets_separate_working_and_keeps_long_term(self):
        self.memory.add_working("Только первая задача")
        self.memory.add_long_term("Общее предпочтение")

        self.memory.switch_task("task-02", session_id="session-02")

        self.assertEqual(self.memory.short_term, ())
        self.assertEqual(self.memory.working, ())
        self.assertEqual(self.memory.long_term, ("Общее предпочтение",))
        self.memory.add_working("Только вторая задача")
        self.memory.switch_task("task-01", session_id="session-03")
        self.assertEqual(self.memory.working, ("Только первая задача",))

    def test_saved_sessions_can_be_listed_and_reopened(self):
        self.memory.commit_exchange("Первая", "Ответ 1")
        self.memory.create_session("session-02")
        self.memory.commit_exchange("Вторая", "Ответ 2")

        self.assertEqual(self.memory.session_ids(), ("session-01", "session-02"))
        self.memory.switch_session("session-01")
        self.assertEqual(self.memory.short_term[0]["content"], "Первая")

    def test_empty_session_and_task_are_persisted_in_navigation(self):
        self.memory.create_session("session-02")
        self.assertIn("session-02", self.memory.session_ids())

        self.memory.create_task("task-02")
        self.assertIn("task-02", self.memory.task_ids())
        self.assertEqual(self.memory.session_ids("task-02"), ("session-01",))

    def test_same_session_id_is_isolated_between_tasks(self):
        self.memory.commit_exchange("Первая задача", "Ответ 1")
        self.memory.switch_task("task-02", session_id="session-01")
        self.memory.commit_exchange("Вторая задача", "Ответ 2")

        self.memory.switch_task("task-01", session_id="session-01")
        self.assertEqual(self.memory.short_term[0]["content"], "Первая задача")
        self.memory.switch_task("task-02", session_id="session-01")
        self.assertEqual(self.memory.short_term[0]["content"], "Вторая задача")

    def test_failed_save_does_not_change_memory(self):
        self.memory.add_working("Стабильное значение")
        original_save = self.memory._working_store().save

        class BrokenStore:
            def save(self, items):
                raise MemoryError("ошибка")

        self.memory._working_store = lambda: BrokenStore()  # type: ignore[method-assign]
        with self.assertRaises(MemoryError):
            self.memory.add_working("Не должно попасть")
        self.assertEqual(self.memory.working, ("Стабильное значение",))
        self.assertTrue(callable(original_save))

    def test_corrupt_json_is_not_silently_overwritten(self):
        path = self.root / "long_term" / "demo.json"
        path.parent.mkdir(parents=True)
        path.write_text("{broken", encoding="utf-8")

        with self.assertRaisesRegex(MemoryError, "Повреждён JSON"):
            MemoryLayers(self.root)
        self.assertEqual(path.read_text(encoding="utf-8"), "{broken")

    def test_short_term_rejects_unfinished_exchange(self):
        path = self.root / "short_term" / "task-01" / "session-01.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps([{"role": "user", "content": "Незавершённый"}]),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(MemoryError, "завершённых пар"):
            MemoryLayers(self.root)

    def test_failed_task_switch_preserves_current_scope(self):
        path = self.root / "working" / "task-02.json"
        path.parent.mkdir(parents=True)
        path.write_text("{broken", encoding="utf-8")

        with self.assertRaisesRegex(MemoryError, "Повреждён JSON"):
            self.memory.switch_task("task-02", session_id="session-02")

        self.assertEqual(self.memory.task_id, "task-01")
        self.assertEqual(self.memory.session_id, "session-01")

    def test_legacy_sessions_remain_visible_in_task_one(self):
        path = self.root / "short_term" / "session-07.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                [
                    {"role": "user", "content": "Старый вопрос"},
                    {"role": "assistant", "content": "Старый ответ"},
                ]
            ),
            encoding="utf-8",
        )

        memory = MemoryLayers(self.root, session_id="session-07")
        self.assertIn("session-07", memory.session_ids("task-01"))
        self.assertEqual(memory.short_term[0]["content"], "Старый вопрос")
