from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import collect
from login_store import collector_status, login_summary, recent_logins


def journal_entry(cursor: str, message: str) -> str:
    return json.dumps({
        "__CURSOR": cursor,
        "__REALTIME_TIMESTAMP": str(int(datetime.now(timezone.utc).timestamp() * 1_000_000)),
        "MESSAGE": message,
    })


class CollectionTests(unittest.TestCase):
    def test_incremental_collection_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logins.sqlite3"
            first = "\n".join([
                journal_entry("c1", "Accepted publickey for alice from 192.0.2.1 port 1234 ssh2: ED25519 SHA256:test"),
                journal_entry("c2", "Failed password for root from 192.0.2.2 port 1235 ssh2"),
            ])
            second = journal_entry("c3", "Accepted publickey for bob from 2001:db8::1 port 1236 ssh2")

            def fake_run(command, **kwargs):
                self.assertEqual(command[-2:], ["--since", "24 hours ago"] if len(calls) == 0 else ["--after-cursor", "c2"])
                result = type("Result", (), {"stdout": first if len(calls) == 0 else second})()
                calls.append(command)
                return result

            calls: list[list[str]] = []
            with patch.dict(os.environ, {"DAY18_DB_PATH": str(path)}), patch("collect.subprocess.run", side_effect=fake_run):
                self.assertEqual(collect.collect(), 1)
                self.assertEqual(collect.collect(), 1)

            recent = recent_logins(path)
            self.assertEqual(recent["total"], 2)
            self.assertEqual({item["username"] for item in recent["logins"]}, {"bob", "alice"})
            self.assertEqual({item["ip"] for item in recent["logins"]}, {"192.0.2.1", "2001:db8::1"})
            self.assertTrue(recent["last_collection"])
            summary = login_summary(path)
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["unique_ips"], 2)
            self.assertEqual(collector_status(path)["total_saved"], 2)

    def test_invalid_window_and_limit(self) -> None:
        with self.assertRaises(ValueError):
            login_summary(Path("/nonexistent"), hours=0)
        with self.assertRaises(ValueError):
            recent_logins(Path("/nonexistent"), limit=101)


if __name__ == "__main__":
    unittest.main()
