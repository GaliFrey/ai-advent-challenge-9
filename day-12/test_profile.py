"""Проверки профилей и привязки профиля к сессии."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from user_profile import DEFAULT_PROFILES, Profile, ProfileError, ProfileStore, render_profile
from session import Session, SessionStore


class ProfileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ProfileStore(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_defaults_are_persisted_and_structured(self):
        profiles = self.store.ensure_defaults()

        self.assertEqual(profiles, DEFAULT_PROFILES)
        self.assertTrue(self.store.path.exists())
        block = render_profile(profiles[0])
        self.assertIn("Стиль:", block)
        self.assertIn("Формат ответа:", block)
        self.assertIn("Ограничения:", block)

    def test_profile_can_be_created_and_updated(self):
        self.store.ensure_defaults()
        created = Profile("reviewer", "Ревьюер", "Автор", "Русский", "Строго", "Список", "Без похвалы")
        self.store.upsert(created)
        updated = Profile("reviewer", "Старший ревьюер", "Автор", "Русский", "Строго", "Список", "Без похвалы")
        profiles = self.store.upsert(updated)

        self.assertEqual(len(profiles), 3)
        self.assertEqual(next(item for item in profiles if item.profile_id == "reviewer").name, "Старший ревьюер")

    def test_invalid_profile_is_not_saved(self):
        self.store.ensure_defaults()
        before = self.store.path.read_bytes()

        with self.assertRaisesRegex(ValueError, "Стиль"):
            Profile("bad", "Плохой", "Друг", "Русский", "", "Текст", "Нет").normalized()

        self.assertEqual(self.store.path.read_bytes(), before)

    def test_corrupt_profiles_are_not_overwritten(self):
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_text("{broken", encoding="utf-8")

        with self.assertRaisesRegex(ProfileError, "Повреждён JSON"):
            self.store.ensure_defaults()

        self.assertEqual(self.store.path.read_text(encoding="utf-8"), "{broken")


class SessionStoreTests(unittest.TestCase):
    def test_session_restores_profile_history_and_actual_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = Session(
                "session-01",
                "mentor",
                [{"role": "user", "content": "Вопрос"}, {"role": "assistant", "content": "Ответ"}],
                [{"role": "system", "content": "Профиль mentor"}, {"role": "user", "content": "Вопрос"}],
            )
            store.save(session)

            self.assertEqual(store.load("session-01"), session)
            self.assertEqual(store.sessions_using("mentor"), ("session-01",))

    def test_unfinished_exchange_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            with self.assertRaisesRegex(ProfileError, "неверную сессию"):
                store.save(Session("session-01", "mentor", [{"role": "user", "content": "Один"}]))

    def test_corrupt_session_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            path = store.path("session-01")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"wrong": True}), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "неверную структуру"):
                store.load("session-01")
