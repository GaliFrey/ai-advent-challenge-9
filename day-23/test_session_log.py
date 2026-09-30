import json
import tempfile
import unittest
from pathlib import Path

from session_log import LOG_DIRECTORY, SessionLog


class SessionLogTests(unittest.TestCase):
    def test_stage_success_and_separate_session_files(self):
        self.assertEqual(LOG_DIRECTORY, Path(__file__).resolve().parent / "logs")
        with tempfile.TemporaryDirectory() as directory:
            log = SessionLog(Path(directory))
            other = SessionLog(Path(directory))
            self.assertNotEqual(log.path, other.path)
            with log.stage("search", mode="rewrite_filter", question_number=1, top_k_before=20):
                pass
            records = [json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in records],
                             ["session_started", "stage_started", "stage_finished"])
            self.assertEqual(records[-1]["top_k_before"], 20)
            self.assertGreaterEqual(records[-1]["seconds"], 0)
            self.assertEqual(log.path.stat().st_mode & 0o777, 0o600)

    def test_error_is_logged_and_propagated_without_sensitive_message(self):
        with tempfile.TemporaryDirectory() as directory:
            log = SessionLog(Path(directory))
            error = RuntimeError("secret-key request-body response-body")
            with self.assertRaises(RuntimeError) as caught:
                with log.stage("rewrite", mode="rewrite", question_number=2):
                    raise error
            self.assertIs(caught.exception, error)
            content = log.path.read_text()
            self.assertNotIn("secret-key", content)
            self.assertNotIn("request-body", content)
            self.assertNotIn("response-body", content)
            record = json.loads(content.splitlines()[-1])
            self.assertEqual(record["event"], "stage_failed")
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertEqual(record["question_number"], 2)
            self.assertTrue(record["frames"])

    def test_unknown_fields_and_text_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log = SessionLog(Path(directory))
            for fields in ({"prompt": "secret"}, {"tokens": "secret"}):
                with self.assertRaises(ValueError):
                    with log.stage("answer", **fields):
                        pass


if __name__ == "__main__":
    unittest.main()
