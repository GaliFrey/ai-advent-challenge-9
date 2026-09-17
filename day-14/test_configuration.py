from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from configuration import ConfigurationError, ConfigurationStore, render_invariants
from policy_engine import parse_model_decision
from session import SessionError, SessionStore


ROOT = Path(__file__).resolve().parent


class ConfigurationTests(unittest.TestCase):
    def test_profiles_have_separate_valid_invariant_sets(self):
        store = ConfigurationStore(ROOT / "config")
        profiles = store.profiles()
        self.assertEqual({item.profile_id for item in profiles}, {"tech-lead", "english-tutor"})
        sets = {item.invariant_set_id: store.invariant_set(item.invariant_set_id) for item in profiles}
        self.assertEqual(len(sets), 2)
        self.assertNotEqual(sets["tech-lead-v2"].content_hash, sets["english-tutor-v1"].content_hash)
        self.assertIn("TECH-CODE-001", render_invariants(sets["tech-lead-v2"]))
        self.assertNotIn("TECH-UV-001", render_invariants(sets["english-tutor-v1"]))

    def test_corrupt_configuration_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profiles.json").write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "JSON"):
                ConfigurationStore(root).profiles()

    def test_model_decisions_are_validated_against_profile_specific_ids(self):
        store = ConfigurationStore(ROOT / "config")
        payload = json.dumps({
            "decision": "refuse",
            "violations": [{"invariant_id": "TECH-LANG-001", "evidence": "in English", "explanation": "wrong language"}],
            "response": "Продолжу на русском.",
        })
        tech = parse_model_decision(store.invariant_set("tech-lead-v2"), payload).result
        self.assertFalse(tech.allowed)
        self.assertEqual(tech.violations[0].invariant_id, "TECH-LANG-001")
        with self.assertRaisesRegex(ValueError, "неизвестный ID"):
            parse_model_decision(store.invariant_set("english-tutor-v1"), payload)


class SessionTests(unittest.TestCase):
    def test_session_persists_profile_invariant_identity_and_audit(self):
        config = ConfigurationStore(ROOT / "config")
        rules = config.invariant_set("tech-lead-v1")
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = store.create("session-01", "tech-lead", rules.set_id, rules.version, rules.content_hash)
            store.append_audit(session, {"decision": "REFUSED"})
            restored = store.load("session-01")
            self.assertEqual(restored.invariant_hash, rules.content_hash)
            audit = (Path(directory) / "audits" / "session-01.jsonl").read_text(encoding="utf-8")
            self.assertIn('"decision": "REFUSED"', audit)

    def test_invalid_session_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            with self.assertRaises(SessionError):
                store.create("bad/id", "tech-lead", "tech-lead-v1", 1, "x" * 64)
