"""Offline requirement checks, including an actual MCP stdio session."""
import asyncio
from contextlib import closing, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx

from assistant import ask, validate
from clients import complete, project_context
from index import build, documents, open_fresh, retrieve
from main import OVERVIEW, main, question_from, show
from project_mcp import context_for, files_for


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class Model:
    tokenizer = Tokenizer()

    def encode(self, texts, **kwargs):
        return [1.0, 0.0] if isinstance(texts, str) else [[1.0, 0.0] for _ in texts]


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "README.md").write_text("# Project\n\nApplication architecture and tests.")
        (self.root / "docs" / "guide.md").write_text("# Guide\n\nRun npm test for unit tests.")
        self.path = self.root / "index.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_only_readme_and_visible_docs(self):
        (self.root / ".env").write_text("not a document")
        (self.root / "src").mkdir()
        (self.root / "src" / "code.md").write_text("not a document")
        (self.root / "docs" / ".private.md").write_text("not a document")
        self.assertEqual([name for name, _ in documents(self.root)], ["README.md", "docs/guide.md"])

    def test_docs_required(self):
        (self.root / "docs" / "guide.md").unlink()
        with self.assertRaises(ValueError):
            documents(self.root)

    def test_symlink_rejected(self):
        (self.root / "docs" / "alias.md").symlink_to(self.root / "README.md")
        with self.assertRaisesRegex(ValueError, "символические"):
            documents(self.root)

    def test_build_and_search(self):
        info = build(self.root, self.path, Model())
        self.assertEqual(info["documents"], 2)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        hits = retrieve(self.root, self.path, "tests", Model())
        self.assertTrue(hits)
        self.assertEqual({hit["source"] for hit in hits}, {"README.md", "docs/guide.md"})

    def test_short_section_expansion_preserves_complete_source(self):
        # A short section longer than one embedding window must be returned
        # whole, once, without mixing fragments from different documents.
        text = "# Guide\n\n" + " ".join(f"component{i}" for i in range(130))
        (self.root / "docs" / "guide.md").write_text(text)
        build(self.root, self.path, Model())
        hits = retrieve(self.root, self.path, "components", Model())
        guide_hits = [hit for hit in hits if hit["source"] == "docs/guide.md"]
        self.assertEqual(len(guide_hits), 1)
        self.assertEqual(guide_hits[0]["text"], text)
        self.assertTrue(guide_hits[0]["context_expanded"])

    def test_changed_added_deleted_docs_make_index_stale(self):
        build(self.root, self.path, Model())
        guide = self.root / "docs" / "guide.md"
        original = guide.read_text()
        guide.write_text(original + " changed")
        with self.assertRaisesRegex(ValueError, "изменилась"):
            open_fresh(self.root, self.path)
        guide.write_text(original)
        extra = self.root / "docs" / "extra.md"
        extra.write_text("# Extra")
        with self.assertRaisesRegex(ValueError, "изменилась"):
            open_fresh(self.root, self.path)
        extra.unlink()
        guide.unlink()
        with self.assertRaises(ValueError):
            open_fresh(self.root, self.path)

    def test_index_identity_and_missing(self):
        with self.assertRaisesRegex(ValueError, "отсутствует"):
            open_fresh(self.root, self.path)
        build(self.root, self.path, Model())
        with self.assertRaisesRegex(ValueError, "другому"):
            open_fresh(self.root / "elsewhere", self.path)

    def test_corrupted_or_wrong_schema_index_has_recovery_message(self):
        self.path.write_bytes(b"not a SQLite database")
        with self.assertRaisesRegex(ValueError, "повреждён.*--index"):
            open_fresh(self.root, self.path)
        self.path.unlink()
        import sqlite3
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("CREATE TABLE unrelated (value TEXT)")
        with self.assertRaisesRegex(ValueError, "неверный формат.*--index"):
            open_fresh(self.root, self.path)

    def test_failed_build_preserves_previous_index(self):
        build(self.root, self.path, Model())
        original = self.path.read_bytes()
        with patch.object(Model, "encode", side_effect=RuntimeError("model failure")):
            with self.assertRaises(RuntimeError):
                build(self.root, self.path, Model())
        self.assertEqual(self.path.read_bytes(), original)

    def test_pipeline_fresh_mcp_and_stale_before_calls(self):
        build(self.root, self.path, Model())
        calls = []

        async def context(project):
            calls.append(project)
            return {"context": {"branch": f"branch-{len(calls)}"}, "files": {"files": []}}

        one = ask("tests", self.root, self.path, Model(), "", "fake", retrieve_only=True,
                  context_loader=context)
        two = ask("tests", self.root, self.path, Model(), "", "fake", retrieve_only=True,
                  context_loader=context)
        self.assertNotEqual(one["mcp"]["context"], two["mcp"]["context"])
        (self.root / "README.md").write_text("changed")
        with self.assertRaises(ValueError):
            ask("tests", self.root, self.path, Model(), "", "fake", context_loader=context)
        self.assertEqual(len(calls), 2)


class AnswerTests(unittest.TestCase):
    sources = [{"id": "S1", "text": "Run npm test for unit tests."},
               {"id": "G1", "text": '{"branch": "feature-demo"}'}]

    def test_valid_docs_and_git_quotes(self):
        response = {"known": True, "answer": "Run npm test [S1]. Branch [G1].",
                    "citations": [{"source": "S1", "quote": "Run npm test for unit tests."},
                                  {"source": "G1", "quote": '"branch": "feature-demo"'}]}
        self.assertTrue(validate(json.dumps(response), self.sources)["known"])

    def test_unknown_reference_not_masked_by_valid_quote(self):
        response = {"known": True, "answer": "tests [S1] extra [S99]",
                    "citations": [{"source": "S1", "quote": "Run npm test for unit tests."}]}
        with self.assertRaises(ValueError):
            validate(json.dumps(response), self.sources)

    def test_invented_or_wrong_source_quote_rejected(self):
        for source, quote in [("S1", "Run yarn test for unit tests."),
                              ("G1", "Run npm test for unit tests.")]:
            response = {"known": True, "answer": f"tests [{source}]",
                        "citations": [{"source": source, "quote": quote}]}
            with self.assertRaises(ValueError):
                validate(json.dumps(response), self.sources)

    def test_missing_citations_and_non_json_rejected(self):
        for raw in ["plain text", json.dumps({"known": True, "answer": "tests [S1]", "citations": []})]:
            with self.assertRaises(ValueError):
                validate(raw, self.sources)

    def test_unknown_cannot_smuggle_facts(self):
        answer = validate('{"known": false, "answer": "", "citations": []}', self.sources)
        self.assertFalse(answer["known"])
        with self.assertRaises(ValueError):
            validate('{"known": false, "answer": "invented fact", "citations": []}', self.sources)

    def test_commands(self):
        self.assertEqual(question_from("/help"), OVERVIEW)
        self.assertEqual(question_from("/help tests?"), "tests?")
        with self.assertRaises(ValueError):
            question_from("/helpful")
        self.assertEqual(main(["/exit"]), 0)


class OutputTests(unittest.TestCase):
    def result(self):
        return {"status": "accepted", "answer": {
            "known": True, "answer": "Запустите npm test [S1].",
            "citations": [{"source": "S1", "quote": "Run npm test for unit tests."}]},
            "sources": [{"id": "S1", "source": "docs/development.md", "section": "Tests",
                         "text": "Run npm test for unit tests.", "score": 0.8},
                        {"id": "S2", "source": "docs/audio.md", "section": "Audio",
                         "text": "Unused audio details.", "score": 0.7}],
            "mcp": {"context": {"branch": "main", "head": "123456789abcdef",
                                 "project": "app", "repository": "repo", "project_path": "app"},
                    "files": {"files": ["src/internal-module.js"], "total": 1, "truncated": False},
                    "trace": {"server": "day31-project"}}}

    def test_default_shows_answer_and_used_sources_without_context_dump(self):
        output = io.StringIO()
        with redirect_stdout(output):
            show(self.result())
        text = output.getvalue()
        self.assertIn("Запустите npm test", text)
        self.assertIn("docs/development.md", text)
        self.assertIn("ветка: main", text)
        self.assertIn("Фрагмент чанка: Run npm test for unit tests.", text)
        for hidden in ("src/internal-module.js", "Unused audio details.", "docs/audio.md"):
            self.assertNotIn(hidden, text)

    def test_excerpt_is_short_and_source_is_not_duplicated(self):
        result = self.result()
        quote = "long excerpt " * 40
        result["answer"]["citations"] = [
            {"source": "S1", "quote": quote}, {"source": "S1", "quote": "Another quote."}]
        output = io.StringIO()
        with redirect_stdout(output):
            show(result)
        text = output.getvalue()
        self.assertEqual(text.count("Источник:"), 1)
        excerpt = next(line.split(": ", 1)[1] for line in text.splitlines()
                       if line.startswith("Фрагмент чанка:"))
        self.assertLessEqual(len(excerpt), 160)
        self.assertTrue(excerpt.endswith("…"))

    def test_details_preserves_evidence_and_prints_answer_once(self):
        output = io.StringIO()
        with redirect_stdout(output):
            show(self.result(), details=True)
        text = output.getvalue()
        self.assertIn("src/internal-module.js", text)
        self.assertIn("Run npm test for unit tests.", text)
        self.assertIn("Unused audio details.", text)
        self.assertEqual(text.count("Запустите npm test"), 1)


class ProviderTests(unittest.TestCase):
    def test_response_contract_and_payload(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            return httpx.Response(200, json={"model": "fake", "choices": [
                {"finish_reason": "stop", "message": {"content": "{}"}}]})
        self.assertEqual(complete([], "test-key", "fake", httpx.MockTransport(handler))["content"], "{}")

    def test_api_errors_do_not_expose_response_body(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as failure:
            complete([], "test-key", "fake", httpx.MockTransport(
                lambda request: httpx.Response(401, text="sensitive-body")))
        self.assertNotIn("sensitive-body", str(failure.exception))

    def test_truncated_and_malformed_responses_rejected(self):
        for body in [{"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}, {}]:
            with self.assertRaises(RuntimeError):
                complete([], "test-key", "fake", httpx.MockTransport(
                    lambda request: httpx.Response(200, json=body)))


class McpTests(unittest.TestCase):
    def test_real_stdio_nested_project_branch_change_and_detached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "app"
            project.mkdir()
            (project / "README.md").write_text("# App")
            (root / "outside.md").write_text("not part of app")

            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)

            git("init", "-b", "main")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial")
            (project / "untracked.txt").write_text("not tracked")
            result = asyncio.run(project_context(project))
            self.assertEqual(result["context"]["branch"], "main")
            self.assertEqual(result["context"]["project_path"], "app")
            self.assertEqual(result["files"]["files"], ["README.md"])
            self.assertEqual(result["trace"]["server"], "day31-project")
            git("checkout", "-b", "feature-test")
            self.assertEqual(asyncio.run(project_context(project))["context"]["branch"], "feature-test")
            git("checkout", "--detach")
            self.assertTrue(context_for(project)["detached"])
            self.assertEqual(files_for(project)["files"], ["README.md"])

    def test_missing_repo_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                context_for(directory)
            with self.assertRaises(RuntimeError):
                asyncio.run(project_context(directory))


if __name__ == "__main__":
    unittest.main()
