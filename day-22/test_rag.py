import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from rich.console import Console

import main
import rag
from retrieval import validate_index


CHUNK = {"chunk_id": "test-1", "source": "day-20/README.md", "title": "День 20",
         "section": "Решение", "text": "Размер и SHA256", "score": 0.8}


def response(answer="Размер и SHA256 [S1]."):
    return {"answer": answer, "usage": {"total_tokens": 30}, "llm_seconds": 0.1,
            "model_returned": "fake", "finish_reason": "stop"}


class RagTests(unittest.TestCase):
    def test_plain_never_searches_and_rag_contains_exact_evidence(self):
        retriever = Mock()
        retriever.search.return_value = [CHUNK]
        complete = Mock(return_value=response())
        plain = rag.answer_question("Проверки?", "plain", key="test", model="fake",
                                    retriever=retriever, completion=complete)
        retriever.search.assert_not_called()
        self.assertEqual(json.loads(plain["messages"][1]["content"]), {"question": "Проверки?"})
        result = rag.answer_question("Проверки?", "rag", key="test", model="fake",
                                     retriever=retriever, completion=complete)
        sources = json.loads(result["messages"][1]["content"])["sources"]
        self.assertEqual(sources[0]["text"], CHUNK["text"])
        self.assertEqual(sources[0]["chunk_id"], CHUNK["chunk_id"])
        self.assertEqual(plain["messages"][0], result["messages"][0])
        self.assertEqual(len(result["messages"]), 2)  # no prior answer/history

    def test_empty_retrieval_and_invalid_citations(self):
        retriever = Mock()
        retriever.search.return_value = []
        result = rag.answer_question("Что известно?", "rag", key="test", model="fake",
                                     retriever=retriever, completion=lambda *args: response("Не знаю [S1]."))
        self.assertEqual(json.loads(result["messages"][1]["content"])["sources"], [])
        self.assertEqual(result["citation_check"]["unknown_references"], ["S1"])

    def test_api_parameters_usage_and_missing_content(self):
        body = {"choices": [{"message": {"content": "Ответ"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 42}, "model": "returned-model"}
        with patch("rag.urlopen", return_value=io.BytesIO(json.dumps(body).encode())) as request:
            result = rag.complete(rag.messages("Вопрос"), "not-a-real-key", "fake")
        sent = json.loads(request.call_args.args[0].data)
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertNotIn("max_tokens", sent)
        self.assertNotIn("Отвечай кратко", sent["messages"][0]["content"])
        self.assertEqual(result["usage"]["total_tokens"], 42)
        for payload in ({}, {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
                        {"choices": [{"message": {"content": "Обрыв"}, "finish_reason": "length"}]}):
            with self.subTest(payload=payload), patch("rag.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
                with self.assertRaises(RuntimeError):
                    rag.complete(rag.messages("Вопрос"), "test", "fake")

    def test_http_error_does_not_leak_body_or_retry(self):
        error = HTTPError(rag.API_URL, 401, "denied", {}, io.BytesIO(b"secret-body"))
        with patch("rag.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as raised:
                rag.complete(rag.messages("Вопрос"), "secret-key", "fake")
        self.assertNotIn("secret", str(raised.exception))
        self.assertEqual(request.call_count, 1)

    def test_failure_preserves_successful_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with patch("main.answer_question", side_effect=[response(), RuntimeError("HTTP 503")]):
                with self.assertRaisesRegex(RuntimeError, "503"):
                    main.run_cases([{"id": "q01", "question": "Вопрос"}], ["plain", "rag"],
                                   path=path, model="fake", key="secret-key", retriever=None,
                                   console=Console(file=io.StringIO()))
            saved = json.loads(path.read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["items"][0]["results"]["plain"]["answer"], response()["answer"])
            self.assertIn("error", saved["items"][0]["results"]["rag"])
            self.assertNotIn("secret-key", path.read_text())

    def test_corpus_drift_and_missing_strategy_are_rejected(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.executescript("CREATE TABLE meta(key, value); CREATE TABLE chunks(strategy);")
            digest = hashlib.sha256(b"a\0body\0").hexdigest()
            connection.execute("INSERT INTO meta VALUES('corpus_sha256', ?)", (digest,))
            connection.execute("INSERT INTO chunks VALUES('section_windows')")
            indexer = SimpleNamespace(check_model=Mock(), read_documents=lambda: [("a", "body")])
            self.assertEqual(validate_index(connection, indexer)["corpus_sha256"], digest)
            indexer.read_documents = lambda: [("a", "edited")]
            with self.assertRaisesRegex(ValueError, "Корпус изменился"):
                validate_index(connection, indexer)
            indexer.read_documents = lambda: [("a", "body")]
            connection.execute("DELETE FROM chunks")
            with self.assertRaisesRegex(ValueError, "нет section_windows"):
                validate_index(connection, indexer)

    def test_control_set_has_ten_explicit_expectations(self):
        cases = json.loads(main.QUESTIONS.read_text())
        self.assertEqual(len({case["id"] for case in cases}), 10)
        self.assertEqual(sum(case["kind"] == "unanswerable" for case in cases), 2)
        self.assertEqual(sum(case["kind"] == "multi_source" for case in cases), 2)
        for case in cases:
            self.assertTrue(case["question"] and case["expected"])
            for source in case["sources"]:
                self.assertTrue((main.DAY.parent / source).is_file())

    def test_show_is_offline_and_does_not_load_retriever(self):
        report = {"created_at": "test", "status": "complete", "items": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            main.save_report(path, report)
            with patch("main.Retriever") as retriever, patch("main.answer_question") as complete:
                with patch("main.Console", return_value=Console(file=io.StringIO())):
                    self.assertEqual(main.main(["show", str(path)]), 0)
            retriever.assert_not_called()
            complete.assert_not_called()

    def test_review_preserves_evidence_without_model_calls(self):
        report = {"items": [{"id": "q01", "sources": [], "results": {"plain": response()}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            main.save_report(path, report)
            with patch("main.answer_question") as complete, patch("main.Retriever") as retriever:
                with patch("main.Console", return_value=Console(file=io.StringIO())):
                    code = main.main(["review", str(path), "--question", "q01", "--mode", "plain",
                                      "--quality", "1", "--unsupported", "no", "--citations", "na",
                                      "--retrieval", "na", "--abstention", "na", "--note", "Неполный ответ"])
            self.assertEqual(code, 0)
            result = json.loads(path.read_text())["items"][0]["results"]["plain"]
            self.assertEqual(result["answer"], response()["answer"])
            self.assertEqual(result["review"]["quality"], 1)
            retriever.assert_not_called()
            complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
