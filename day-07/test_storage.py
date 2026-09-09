"""Локальные проверки JSON-хранилища."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from storage import HistoryStorageError, JsonHistoryStore


HISTORY = [
    {"role": "user", "content": "Меня зовут Влад"},
    {"role": "assistant", "content": "Запомнил."},
]


class JsonHistoryStoreTests(unittest.TestCase):
    def test_save_load_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonHistoryStore(Path(directory), "left")

            self.assertEqual(store.load(), [])
            store.save(HISTORY)

            self.assertEqual(store.load(), HISTORY)
            self.assertEqual(
                json.loads(store.path.read_text(encoding="utf-8")),
                HISTORY,
            )
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

            store.clear()
            self.assertEqual(store.load(), [])

    def test_sessions_use_different_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = JsonHistoryStore(root, "left")
            right = JsonHistoryStore(root, "right")
            left.save(HISTORY)
            right.save([
                {"role": "user", "content": "Другая сессия"},
                {"role": "assistant", "content": "Вижу."},
            ])

            self.assertEqual(left.path.name, "left.json")
            self.assertEqual(right.path.name, "right.json")
            self.assertNotEqual(left.load(), right.load())

    def test_corrupted_and_invalid_history_are_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonHistoryStore(Path(directory), "left")
            store.path.write_text("{broken", encoding="utf-8")
            original = store.path.read_bytes()

            with self.assertRaisesRegex(HistoryStorageError, "повреждённый JSON"):
                store.load()
            self.assertEqual(store.path.read_bytes(), original)

            store.path.write_text(
                json.dumps([{"role": "user", "content": "незавершённая пара"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(HistoryStorageError, "завершённых пар"):
                store.load()

    def test_session_id_cannot_escape_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "недопустимые"):
                JsonHistoryStore(Path(directory), "../left")


if __name__ == "__main__":
    unittest.main()
