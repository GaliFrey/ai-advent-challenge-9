from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import Input, RichLog, Select, TabPane

from agent import AgentConfig
from configuration import ConfigurationStore
from main import InvariantsApp, _chat_line, _session_violation_counts
from session import Session, SessionStore


def response(text: str = "A tuple is immutable; a list is mutable.", *, violation: str | None = None) -> httpx.Response:
    violations = [] if violation is None else [{
        "invariant_id": violation,
        "evidence": "request",
        "explanation": "The request conflicts with the invariant.",
    }]
    content = json.dumps({
        "decision": "allow" if violation is None else "refuse",
        "violations": violations,
        "response": text,
    })
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    })


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_report_counts_refusals_from_conversation(self):
        session = Session(
            "session-01", "tech-lead", "tech-lead-v2", 2, "0" * 64,
            messages=[
                {"role": "user", "content": "Go"},
                {"role": "assistant", "content": "ОТКАЗ · TECH-CODE-001\nПервый"},
                {"role": "user", "content": "Python 3.10"},
                {"role": "assistant", "content": "Не могу выполнить: конфликт с инвариантом [TECH-CODE-001]"},
            ],
        )
        self.assertEqual(_session_violation_counts(session)["TECH-CODE-001"], 2)

    async def test_session_report_uses_persisted_policy_metadata(self):
        session = Session(
            "session-01", "tech-lead", "tech-lead-v2", 2, "0" * 64,
            messages=[
                {"role": "user", "content": "Go"},
                {"role": "assistant", "content": "Без слова отказ и без ID"},
            ],
            policy_history=[{"decision": "REFUSED", "violations": ["TECH-CODE-001"]}],
        )
        self.assertEqual(_session_violation_counts(session)["TECH-CODE-001"], 1)

    async def test_session_report_combines_legacy_and_structured_turns(self):
        session = Session(
            "session-01", "tech-lead", "tech-lead-v2", 2, "0" * 64,
            messages=[
                {"role": "user", "content": "Старый запрос"},
                {"role": "assistant", "content": "ОТКАЗ · TECH-LANG-001"},
                {"role": "user", "content": "Новый запрос"},
                {"role": "assistant", "content": "Структурированный отказ"},
            ],
            policy_history=[{"decision": "REFUSED", "violations": ["TECH-CODE-001"]}],
        )
        counts = _session_violation_counts(session)
        self.assertEqual(counts["TECH-LANG-001"], 1)
        self.assertEqual(counts["TECH-CODE-001"], 1)

    async def test_restored_chat_uses_same_role_colors(self):
        user = _chat_line("user", "Вопрос")
        assistant = _chat_line("assistant", "Ответ")

        self.assertEqual(user.plain, "Вы: Вопрос")
        self.assertEqual(assistant.plain, "Агент: Ответ")
        self.assertEqual(str(user.style), "bold #8ed6dc")
        self.assertEqual(str(assistant.style), "bold #b8d982")

    async def test_layout_and_profile_bound_session(self):
        with tempfile.TemporaryDirectory() as directory:
            app = InvariantsApp(
                AgentConfig(api_key="fake"), data_dir=Path(directory),
                transport=httpx.MockTransport(lambda request: response()),
            )
            async with app.run_test(size=(190, 55)):
                self.assertEqual(len(app.query(RichLog)), 4)
                self.assertEqual(len(app.query(TabPane)), 3)
                self.assertEqual(len(app.query(Select)), 2)
                self.assertEqual(app.profile.profile_id, "tech-lead")
                self.assertEqual(app.rules.set_id, "tech-lead-v2")
                widgets = [
                    app.query_one("#profile-label"), app.query_one("#profile"),
                    app.query_one("#new-session"), app.query_one("#session-label"),
                    app.query_one("#session"), app.query_one("#demo"),
                ]
                self.assertEqual({widget.region.y for widget in widgets}, {4})
                self.assertEqual({widget.region.height for widget in widgets}, {3})
                self.assertLess(widgets[-1].region.right, app.size.width)

    async def test_new_session_uses_selected_profile_and_invariants(self):
        with tempfile.TemporaryDirectory() as directory:
            app = InvariantsApp(AgentConfig(api_key="fake"), data_dir=Path(directory))
            async with app.run_test(size=(190, 55)) as pilot:
                app.query_one("#profile", Select).value = "english-tutor"
                await pilot.click("#new-session")
                await pilot.pause()
                self.assertEqual(app.profile.profile_id, "english-tutor")
                self.assertEqual(app.session.invariant_set_id, "english-tutor-v1")

    async def test_profile_selection_rebinds_empty_session(self):
        with tempfile.TemporaryDirectory() as directory:
            app = InvariantsApp(AgentConfig(api_key="fake"), data_dir=Path(directory))
            async with app.run_test(size=(190, 55)) as pilot:
                session_id = app.session.session_id
                app.query_one("#profile", Select).value = "english-tutor"
                await pilot.pause()

                self.assertEqual(app.session.session_id, session_id)
                self.assertEqual(app.profile.profile_id, "english-tutor")
                self.assertEqual(app.rules.set_id, "english-tutor-v1")
                restored = app.session_store.load(session_id)
                self.assertEqual(restored.profile_id, "english-tutor")
                self.assertEqual(restored.invariant_set_id, "english-tutor-v1")

    async def test_old_session_restores_its_original_invariant_version(self):
        config_store = ConfigurationStore(Path(__file__).resolve().parent / "config")
        old_rules = config_store.invariant_set("tech-lead-v1")
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.create("session-01", "tech-lead", old_rules.set_id, old_rules.version, old_rules.content_hash)
            app = InvariantsApp(AgentConfig(api_key="fake"), data_dir=Path(directory))
            async with app.run_test(size=(190, 55)):
                self.assertEqual(app.profile.profile_id, "tech-lead")
                self.assertEqual(app.rules.set_id, "tech-lead-v1")
                self.assertIsNone(app.rules.by_id("TECH-CODE-001"))

    async def test_profile_selection_creates_new_session_after_dialog_started(self):
        with tempfile.TemporaryDirectory() as directory:
            app = InvariantsApp(
                AgentConfig(api_key="fake"), data_dir=Path(directory),
                transport=httpx.MockTransport(lambda request: response()),
            )
            async with app.run_test(size=(190, 55)) as pilot:
                first_id = app.session.session_id
                app.query_one("#message", Input).value = "Answer only in English."
                app.start_ask()
                await app.workers.wait_for_complete()
                app.query_one("#profile", Select).value = "english-tutor"
                await pilot.pause()

                self.assertNotEqual(app.session.session_id, first_id)
                self.assertEqual(app.profile.profile_id, "english-tutor")
                self.assertEqual(app.session.messages, [])
                first = app.session_store.load(first_id)
                self.assertEqual(first.profile_id, "tech-lead")
                self.assertEqual(len(first.messages), 2)

    async def test_explicit_conflict_is_visible_from_model_decision(self):
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            return response("Не могу ответить на английском. TECH-LANG-001", violation="TECH-LANG-001")
        with tempfile.TemporaryDirectory() as directory:
            app = InvariantsApp(AgentConfig(api_key="fake"), data_dir=Path(directory),
                                transport=httpx.MockTransport(handler))
            async with app.run_test(size=(190, 55)):
                app.query_one("#message", Input).value = "Answer only in English."
                app.start_ask()
                await app.workers.wait_for_complete()
                self.assertEqual(calls, 1)
                self.assertEqual(app.session.last_checks[0]["status"], "VIOLATION")
                self.assertIn("TECH-LANG-001", str(app.session.messages))
