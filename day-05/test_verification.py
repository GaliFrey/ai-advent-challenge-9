import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx
from rich.markdown import Markdown
from textual.widgets import Static

from experiment import Model, Settings, ROOT, load_session, run_experiment
from main import CompareApp
from verification import FIXED, ORIGINAL, code_sections, release_reference, verify_answer, verify_release_answer, verify_session, verification_markdown

FIXTURE = json.loads((ROOT / "fixtures" / "python-build-review.json").read_text())


def sample():
    return {"prompts": [{"id": "task-1", "text": "Другая задача"}, {"id": "task-2", "text": FIXTURE["prompt"]}],
            "models": FIXTURE["models"], "results": {"task-1": {}, "task-2": {
                slot: {"status": "ok", "answer": answer} for slot, answer in FIXTURE["answers"].items()}}}


class VerificationTests(unittest.TestCase):
    def test_release_reference_and_structured_result(self):
        expected = release_reference()
        self.assertEqual(expected, {
            "best_15": {"tasks": ["A", "B", "D", "F"], "budget": 15, "days": 12, "value": 33},
            "second_15": {"tasks": ["A", "B", "C"], "budget": 15, "days": 12, "value": 31},
            "best_12": {"tasks": ["A", "B", "F"], "budget": 12, "days": 10, "value": 26},
        })
        result = {
            "budget_15": {"best": expected["best_15"], "second": expected["second_15"]},
            "budget_12": {"best": expected["best_12"]},
        }
        answer = "## Итог\n```json\n" + json.dumps(result, ensure_ascii=False) + "\n```"
        verified = verify_release_answer({"status": "ok", "answer": answer})
        self.assertEqual(verified["score"], 3)
        result["budget_15"]["second"]["value"] = 30
        wrong = "## Итог\n```json\n" + json.dumps(result, ensure_ascii=False) + "\n```"
        self.assertEqual(verify_release_answer({"status": "ok", "answer": wrong})["score"], 2)
        self.assertIsNone(verify_release_answer({"status": "ok", "answer": "Без раздела"})["score"])

    def test_both_default_prompts_have_independent_checkers(self):
        session = sample()
        session["prompts"][0]["text"] = (ROOT / "prompt.txt").read_text()
        reference = release_reference()
        result = {"budget_15": {"best": reference["best_15"], "second": reference["second_15"]}, "budget_12": {"best": reference["best_12"]}}
        answer = "## Итог\n```json\n" + json.dumps(result) + "\n```"
        session["results"]["task-1"] = {model["slot"]: {"status": "ok", "answer": answer} for model in session["models"]}
        verification = verify_session(session)
        self.assertEqual(verification["tasks"]["task-1"]["checker"], "release-plan-v1")
        self.assertEqual(verification["tasks"]["task-1"]["max_score"], 3)
        self.assertEqual(verification["tasks"]["task-2"]["checker"], "python-build-v1")

    def test_regression_three_real_answers_and_tied_leaders(self):
        session = sample()
        session["verification"] = verify_session(session)
        task = session["verification"]["tasks"]["task-2"]
        self.assertEqual([task["answers"][s]["score"] for s in ("weak", "medium", "strong")], [2, 3, 3])
        self.assertEqual(task["leaders"], ["medium", "strong"])
        for slot, passed, total in (("weak", 5, 7), ("medium", 7, 7), ("strong", 8, 8)):
            checks = task["answers"][slot]["checks"]
            self.assertEqual(checks["contract"]["status"], "pass")
            self.assertEqual((checks["asserts"]["passed"], checks["asserts"]["total"]), (passed, total))
        rendered = verification_markdown(session)
        self.assertIn("Общий победитель автоматически не определяется", rendered)
        self.assertIn("(5/7)", rendered)
        # Ensure rows form a real Markdown table, not disconnected paragraphs.
        tables = [token for token in Markdown(rendered).parsed if token.type == "table_open"]
        self.assertEqual(len(tables), 1)

    def test_reference_stdout_is_executed_independently(self):
        # Trusted fixture from the exercise, not generated model code.
        source = FIXTURE["prompt"].split("\n\n", 1)[1].split("\n\n1.", 1)[0]
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(source, {})
        self.assertEqual(out.getvalue().strip(), ORIGINAL)
        corrected = code_sections(FIXTURE["answers"]["strong"])["исправление"]
        # Worker verifies that this produces the stored FIXED reference.
        answer = FIXTURE["answers"]["strong"].replace("[(3, (1, 2, 3))]\n```\n\n### Проверки", "[(3, (3,))]\n```\n\n### Проверки")
        self.assertEqual(verify_answer({"status": "ok", "answer": answer})["score"], 4)
        self.assertIn("def build", corrected)
        self.assertIn("[(3, (3,))]", FIXED)

    def test_changed_prompt_and_incomplete_answer_do_not_get_a_rank(self):
        session = sample()
        session["prompts"][1]["text"] += "\nИзмени контракт."
        with patch("verification.subprocess.run") as runner:
            self.assertEqual(verify_session(session)["tasks"]["task-2"]["status"], "unsupported")
            runner.assert_not_called()
        session = sample()
        session["results"]["task-2"]["weak"]["status"] = "incomplete"
        task = verify_session(session)["tasks"]["task-2"]
        self.assertIsNone(task["answers"]["weak"]["score"])
        self.assertEqual(task["leaders"], [])

    def test_code_cannot_import_read_files_or_introspect(self):
        original = FIXTURE["answers"]["strong"]
        function = code_sections(original)["исправление"]
        bad_functions = [
            "import os\ndef build(values, history=None):\n    return []",
            "def build(values, history=None):\n    return open('/etc/passwd').read()",
            "def build(values, history=None):\n    return ().__class__.__base__.__subclasses__()",
            "def build(values, history=None):\n    while True:\n        pass",
        ]
        for bad in bad_functions:
            with self.subTest(bad=bad):
                result = verify_answer({"status": "ok", "answer": original.replace(function, bad)})
                self.assertEqual(result["checks"]["contract"]["status"], "unknown")
                self.assertIsNone(result["score"])

    def test_incorrect_function_and_asserts_are_not_trusted(self):
        original = FIXTURE["answers"]["medium"]
        function = code_sections(original)["исправление"]
        answer = original.replace(function, "def build(values, history=None):\n    return []")
        checks = verify_answer({"status": "ok", "answer": answer})["checks"]
        self.assertEqual(checks["contract"]["status"], "fail")
        self.assertEqual(checks["asserts"]["status"], "fail")

    def test_ambiguous_sections_and_worker_limits_are_unknown(self):
        original = FIXTURE["answers"]["strong"]
        result = verify_answer({"status": "ok", "answer": original + "\n### Исправление\n```python\ndef build(values, history=None):\n    return []\n```"})
        self.assertIsNone(result["score"])
        with patch("verification.subprocess.run", side_effect=subprocess.TimeoutExpired("checker", 2)):
            result = verify_answer({"status": "ok", "answer": original})
        self.assertEqual(result["checks"]["contract"]["status"], "unknown")
        self.assertIsNone(result["score"])


class VerificationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_checks_are_recomputed_and_comparison_is_local(self):
        models = tuple(Model(**m) for m in FIXTURE["models"])
        calls = []
        async def handler(request):
            data = json.loads(request.content)
            calls.append(data)
            slot = models[(len(calls) - 1) % 3].slot
            answer = FIXTURE["answers"][slot] if len(calls) > 3 else "Ответ"
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer}}]})
        with tempfile.TemporaryDirectory() as directory:
            session, path = await run_experiment(Settings(models, "fake", "fake"), ["Другая задача", FIXTURE["prompt"]], lambda *_: None, results_dir=Path(directory), transport=httpx.MockTransport(handler))
            original = path.read_bytes()
            self.assertEqual(load_session(path), session)
            stale = copy.deepcopy(session)
            stale["verification"]["tasks"]["task-2"]["answers"]["weak"]["score"] = 4
            path.write_text(json.dumps(stale))
            self.assertEqual(load_session(path)["verification"], session["verification"])
            path.write_bytes(original)
            def no_network(request):
                self.fail("Viewing saved data cannot call API")
            app = CompareApp(Settings(models), session_path=path, results_dir=Path(directory), transport=httpx.MockTransport(no_network))
            async with app.run_test(size=(110, 40)):
                conclusion = app.query_one("#conclusion-text", Static).content.markup
                self.assertIn("2/4", conclusion)
                self.assertIn("3/4", conclusion)
                self.assertIn("Исходный вывод", str(app.query_one("#compare-preview-2-weak", Static).content))
                self.assertIn("полный текст во вкладке модели", str(app.query_one("#compare-preview-2-weak", Static).content))
                self.assertIn("локальные проверки 2/4", str(app.query_one("#compare-meta-2-weak", Static).content))
                self.assertEqual(len(app.query("#review-text")), 0)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(calls), 6)
