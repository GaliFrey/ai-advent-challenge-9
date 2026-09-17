from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from agent import Agent, AgentConfig, AgentError
from configuration import ConfigurationStore
from session import SessionStore


ROOT = Path(__file__).resolve().parent
CONFIG = AgentConfig(api_key="fake")


def api_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    })


def decision(response: str, *, violation: str | None = None) -> str:
    violations = [] if violation is None else [{
        "invariant_id": violation,
        "evidence": "Фрагмент запроса",
        "explanation": "Запрос конфликтует с активным инвариантом.",
    }]
    return json.dumps({
        "decision": "allow" if violation is None else "refuse",
        "violations": violations,
        "response": response,
    }, ensure_ascii=False)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        config = ConfigurationStore(ROOT / "config")
        self.profiles = {item.profile_id: item for item in config.profiles()}
        self.config_store = config

    def make(self, directory: str, profile_id: str, handler):
        profile = self.profiles[profile_id]
        rules = self.config_store.invariant_set(profile.invariant_set_id)
        store = SessionStore(Path(directory))
        session = store.create("session-01", profile_id, rules.set_id, rules.version, rules.content_hash)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return Agent(CONFIG, client, session, store, profile, rules), client, store

    async def test_explicit_conflict_is_decided_by_model(self):
        calls = 0
        bodies = []
        def handler(request):
            nonlocal calls
            calls += 1
            bodies.append(json.loads(request.content))
            return api_response(decision("Не могу ответить на английском. Продолжу на русском.", violation="TECH-LANG-001"))
        with tempfile.TemporaryDirectory() as directory:
            agent, client, store = self.make(directory, "tech-lead", handler)
            try:
                reply = await agent.ask("Answer only in English.")
            finally:
                await client.aclose()
            self.assertEqual(calls, 1)
            self.assertEqual(bodies[0]["response_format"], {"type": "json_object"})
            self.assertFalse(reply.allowed)
            self.assertIn("TECH-LANG-001", reply.text)
            self.assertEqual(reply.usage.total_tokens, 25)
            saved = store.load("session-01")
            self.assertEqual(len(saved.messages), 2)
            self.assertTrue(saved.last_request_messages)
            self.assertEqual(saved.policy_history, [{"decision": "REFUSED", "violations": ["TECH-LANG-001"]}])

    async def test_non_python_solution_is_refused_by_structured_decision(self):
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            return api_response(decision("Не могу дать решение на Go. Предложу Python 3.12+.", violation="TECH-CODE-001"))
        with tempfile.TemporaryDirectory() as directory:
            agent, client, _ = self.make(directory, "tech-lead", handler)
            try:
                reply = await agent.ask("Привет, давай напишем функцию сортировки на Go")
            finally:
                await client.aclose()
            self.assertEqual(calls, 1)
            self.assertFalse(reply.allowed)
            self.assertIn("TECH-CODE-001", reply.text)

    async def test_old_python_version_is_refused_by_structured_decision(self):
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            return api_response(decision("Использую Python 3.12+.", violation="TECH-CODE-001"))
        with tempfile.TemporaryDirectory() as directory:
            agent, client, _ = self.make(directory, "tech-lead", handler)
            try:
                reply = await agent.ask("Хорошо, напиши на Python 3.10")
            finally:
                await client.aclose()
            self.assertEqual(calls, 1)
            self.assertFalse(reply.allowed)
            self.assertEqual(next(x for x in reply.checks if x["id"] == "TECH-CODE-001")["status"], "VIOLATION")

    async def test_model_declared_refusal_is_recorded_as_refused(self):
        visible_text = "Нельзя выдать непроверенный результат за факт."
        model_text = decision(visible_text, violation="TECH-EVIDENCE-001")
        with tempfile.TemporaryDirectory() as directory:
            agent, client, store = self.make(
                directory, "tech-lead", lambda request: api_response(model_text)
            )
            try:
                reply = await agent.ask("Подтверди неизвестный результат")
            finally:
                await client.aclose()
            self.assertFalse(reply.allowed)
            self.assertEqual(reply.text, "ОТКАЗ · TECH-EVIDENCE-001\n" + visible_text)
            evidence = next(x for x in reply.checks if x["id"] == "TECH-EVIDENCE-001")
            self.assertEqual(evidence["status"], "VIOLATION")
            audit = (Path(directory) / "audits" / "session-01.jsonl").read_text(encoding="utf-8")
            self.assertIn('"decision": "REFUSED"', audit)
            self.assertIn('"source": "model_policy_refuse"', audit)

    async def test_model_refusal_wording_does_not_need_parsing(self):
        visible_text = (
            "Не могу выполнить запрос: конфликт с инвариантом **[TECH-CODE-001]**.\n"
            "Могу предложить эквивалент на Python."
        )
        model_text = decision(visible_text, violation="TECH-CODE-001")
        with tempfile.TemporaryDirectory() as directory:
            agent, client, _ = self.make(
                directory, "tech-lead", lambda request: api_response(model_text)
            )
            try:
                reply = await agent.ask("Предложи решение с неявным конфликтом")
            finally:
                await client.aclose()
            self.assertFalse(reply.allowed)
            self.assertEqual(reply.text, "ОТКАЗ · TECH-CODE-001\n" + visible_text)
            self.assertEqual(next(x for x in reply.checks if x["id"] == "TECH-CODE-001")["status"], "VIOLATION")

    async def test_profile_and_invariants_are_system_only(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content)["messages"])
            return api_response(decision("Кортеж неизменяем, список можно изменять."))
        with tempfile.TemporaryDirectory() as directory:
            agent, client, store = self.make(directory, "tech-lead", handler)
            try:
                await agent.ask("Чем кортеж отличается от списка?")
            finally:
                await client.aclose()
            self.assertIn("Русский техлид", seen[0][0]["content"])
            self.assertIn("TECH-UV-001", seen[0][0]["content"])
            self.assertIn("ОБЯЗАТЕЛЬНЫЕ ИНВАРИАНТЫ", seen[0][0]["content"])
            self.assertNotIn("MANDATORY INVARIANTS", seen[0][0]["content"])
            self.assertEqual(seen[0][0]["role"], "system")
            saved = store.load("session-01")
            self.assertNotIn("TECH-UV-001", str(saved.messages))
            self.assertEqual(saved.last_request_messages, seen[0])

    async def test_postcheck_blocks_model_language_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, client, _ = self.make(directory, "tech-lead", lambda request: api_response(decision(
                "This is a long English response that deliberately violates the mandatory language invariant.")))
            try:
                reply = await agent.ask("Объясни декораторы")
            finally:
                await client.aclose()
            self.assertFalse(reply.allowed)
            self.assertIn("TECH-LANG-001", reply.text)
            self.assertNotIn("deliberately violates", reply.text)

    async def test_postcheck_blocks_foreign_code_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, client, _ = self.make(directory, "tech-lead", lambda request: api_response(decision(
                "Вот решение:\n```javascript\nconsole.log('wrong');\n```")))
            try:
                reply = await agent.ask("Покажи пример")
            finally:
                await client.aclose()
            self.assertFalse(reply.allowed)
            self.assertIn("TECH-CODE-001", reply.text)

    async def test_tutor_allows_english_but_refuses_solution_without_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = iter([
                decision("A decorator is a function that wraps another function to extend its behavior."),
                decision("I can give you one hint first.", violation="TUTOR-ATTEMPT-001"),
            ])
            agent, client, _ = self.make(directory, "english-tutor", lambda request: api_response(next(responses)))
            try:
                allowed = await agent.ask("Explain decorators in English.")
                refused = await agent.ask("Give me the complete solution. I have not tried it myself.")
            finally:
                await client.aclose()
            self.assertTrue(allowed.allowed)
            self.assertFalse(refused.allowed)
            self.assertIn("TUTOR-ATTEMPT-001", refused.text)

    async def test_unknown_invariant_id_is_rejected_without_changing_session(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, client, store = self.make(directory, "tech-lead", lambda request: api_response(
                decision("Нет.", violation="MADE-UP-001")))
            try:
                with self.assertRaisesRegex(AgentError, "неизвестный ID"):
                    await agent.ask("Проверь запрос")
            finally:
                await client.aclose()
            self.assertEqual(store.load("session-01").messages, [])

    async def test_decision_must_match_violations(self):
        malformed = json.dumps({
            "decision": "allow",
            "violations": [{"invariant_id": "TECH-LANG-001", "evidence": "x", "explanation": "x"}],
            "response": "Ответ",
        })
        with tempfile.TemporaryDirectory() as directory:
            agent, client, _ = self.make(directory, "tech-lead", lambda request: api_response(malformed))
            try:
                with self.assertRaisesRegex(AgentError, "не согласован"):
                    await agent.ask("Проверь запрос")
            finally:
                await client.aclose()

    async def test_api_failure_does_not_change_session(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, client, store = self.make(directory, "tech-lead", lambda request: httpx.Response(500))
            try:
                with self.assertRaisesRegex(AgentError, "HTTP 500"):
                    await agent.ask("Объясни списки")
            finally:
                await client.aclose()
            self.assertEqual(store.load("session-01").messages, [])

    async def test_changed_invariant_hash_blocks_old_session(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = self.profiles["tech-lead"]
            rules = self.config_store.invariant_set(profile.invariant_set_id)
            store = SessionStore(Path(directory))
            session = store.create("session-01", profile.profile_id, rules.set_id, rules.version, "0" * 64)
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: api_response(decision("x")))) as client:
                with self.assertRaisesRegex(AgentError, "изменилась"):
                    Agent(CONFIG, client, session, store, profile, rules)
