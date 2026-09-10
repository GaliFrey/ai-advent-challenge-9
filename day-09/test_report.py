"""Проверки JSON-отчёта демо."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from report import save_demo_report


class ReportTests(unittest.TestCase):
    def test_report_is_saved_outside_resources_with_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            path = save_demo_report({"outcome": "completed", "turns": []}, results)

            self.assertEqual(path.parent, results)
            self.assertTrue(path.name.startswith("demo-"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["outcome"], "completed")
            self.assertEqual(payload["turns"], [])
            self.assertIn("created_at", payload)
            self.assertFalse(any(item.name.endswith(".tmp") for item in results.iterdir()))


if __name__ == "__main__":
    unittest.main()
