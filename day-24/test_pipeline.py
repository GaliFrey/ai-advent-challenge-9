import json
import tempfile
import unittest
from pathlib import Path

from evidence import answer_text, validate
from pipeline import Runner, Settings, load_cases

CHUNK = {"chunk_id": "chunk-1", "source": "day-11/README.md", "section": "Память", "title": "День 11",
         "text": "Новая задача очищает краткосрочную и рабочую память, сохраняя долговременную.", "score": .8}
CASES = [{"id": "q1", "question": "Какая память очищается?"}]


def grounded(chunks):
    c = chunks[0]
    return {"status": "answered", "answer": [{"text": "Очищаются краткосрочная и рабочая память.",
            "citations": [{"source_id": c["source_id"], "quote": c["text"]}]}],
            "sources": [{"source_id": c["source_id"], **{k: c[k] for k in ("source", "section", "chunk_id")}}], "clarification": ""}


def completion(messages, key, model):
    user = json.loads(messages[1]["content"])
    if "response" in user:
        answer = {"verdict": "pass", "checks": {"support": True, "coverage": True, "abstention": True},
                  "explanation": "Подтверждено цитатой; все доступные сведения отражены."}
    elif "sources" not in user:
        answer = {"query": "День 11: очистка памяти"}
    else:
        answer = grounded(user["sources"])
    return {"answer": json.dumps(answer, ensure_ascii=False), "usage": {"total_tokens": 10}, "llm_seconds": .01}


class FakeRetriever:
    metadata = {"test": True}

    def __init__(self, index):
        pass

    def load(self):
        pass

    def search(self, question, k):
        return [CHUNK.copy()]


class FakeReranker:
    def rank(self, question, chunks):
        return [{**c, "rerank_score": .8} for c in chunks]


class PipelineTests(unittest.TestCase):
    def run_case(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            from session_log import SessionLog
            path = Path(directory) / "result.json"
            runner = Runner(key="test-secret", retriever_factory=FakeRetriever,
                            reranker_factory=FakeReranker, **kwargs)
            report = runner.run(CASES, path, SessionLog(Path(directory) / "logs"))
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertNotIn("test-secret", path.read_text())
            return report

    def test_sources_quotes_and_original_question(self):
        calls = []
        def complete(messages, key, model):
            calls.append(messages)
            return completion(messages, key, model)
        report = self.run_case(completion=complete)
        item = report["items"][0]
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(calls), 3)
        self.assertEqual(json.loads(calls[1][1]["content"])["question"], CASES[0]["question"])
        self.assertEqual(item["checks"]["quotes"], 1)
        self.assertEqual(item["judge"]["assessment"]["verdict"], "pass")
        payload = json.loads(calls[2][1]["content"])
        self.assertEqual(payload["response"], item["response"])
        self.assertEqual(len(calls[2]), 2)
        self.assertNotIn("expected", payload)
        self.assertNotIn("query", payload)

    def test_threshold_refusal_skips_generation_and_asks_clarification(self):
        calls = []
        def complete(messages, key, model):
            calls.append(messages)
            return completion(messages, key, model)
        report = self.run_case(completion=complete, settings=Settings(rerank_threshold=.9))
        item = report["items"][0]
        self.assertEqual(len(calls), 2)
        self.assertEqual(item["origin"], "threshold")
        self.assertEqual(item["response"]["status"], "unknown")
        self.assertTrue(item["response"]["clarification"])
        self.assertIn("Не знаю", answer_text(item["response"]))
        self.assertEqual(item["response"]["sources"], [])

    def test_model_refuses_when_thematic_chunk_has_no_answer(self):
        def complete(messages, key, model):
            if "sources" not in json.loads(messages[1]["content"]):
                return completion(messages, key, model)
            return {"answer": json.dumps({"status": "unknown", "answer": [], "sources": [],
                                          "clarification": "Уточните нужную версию документа."}), "usage": {}, "llm_seconds": 0}
        item = self.run_case(completion=complete)["items"][0]
        self.assertTrue(item["chunks"])
        self.assertEqual(item["response"]["status"], "unknown")

    def test_invalid_model_response_is_not_published_as_verified(self):
        def complete(messages, key, model):
            if "sources" not in json.loads(messages[1]["content"]):
                return completion(messages, key, model)
            return {"answer": "Неподтверждённый ответ", "usage": {}, "llm_seconds": 0}
        report = self.run_case(completion=complete)
        self.assertEqual(report["status"], "invalid")
        self.assertIsNone(report["items"][0]["response"])
        self.assertEqual(report["items"][0]["judge"]["status"], "skipped")

    def test_partial_failure_and_cancel(self):
        def fail(messages, key, model):
            if "sources" in json.loads(messages[1]["content"]):
                raise RuntimeError("sensitive text")
            return completion(messages, key, model)
        report = self.run_case(completion=fail)
        self.assertEqual(report["error"], {"stage": "answer", "type": "RuntimeError"})
        self.assertNotIn("sensitive text", json.dumps(report))
        with tempfile.TemporaryDirectory() as directory:
            from session_log import SessionLog
            report = Runner(key="test", completion=completion, retriever_factory=FakeRetriever).run(
                CASES, Path(directory) / "result.json", SessionLog(Path(directory) / "logs"), cancelled=lambda: True)
            self.assertEqual(report["status"], "cancelled")

    def test_ten_questions_and_automatic_assessment(self):
        self.assertEqual(len(load_cases()), 10)
        report = self.run_case(completion=completion)
        self.assertEqual(report["items"][0]["judge"]["status"], "complete")

    def test_settings(self):
        for settings in ({"top_k_before": 0}, {"top_k_after": 21}, {"rerank_threshold": float('nan')}, {"rerank_threshold": 1.1}):
            with self.assertRaises(ValueError):
                Settings(**settings)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{**CHUNK, "source_id": "S1"}]

    def test_exact_quote_check_does_not_claim_semantic_validity(self):
        response = grounded(self.chunks)
        response["answer"][0]["text"] = "Несвязанный вывод"
        _, checks = validate(json.dumps(response), self.chunks)
        self.assertTrue(checks["passed"])
        self.assertNotIn("semantic", checks)

    def test_missing_quotes_unknown_references_and_forged_metadata(self):
        for change in (lambda r: r["answer"][0].update(citations=[]),
                       lambda r: r["answer"][0]["citations"][0].update(source_id="S99"),
                       lambda r: r["sources"][0].update(chunk_id="fake"),
                       lambda r: r["answer"][0]["citations"][0].update(quote="Выдуманная цитата"),
                       lambda r: r.update(sources=[])):
            response = grounded(self.chunks)
            change(response)
            self.assertFalse(validate(json.dumps(response), self.chunks)[1]["passed"])

    def test_malformed_output_and_refusal_without_clarification(self):
        for raw in ("not json", "[]", '{"status":"answered"}',
                    '{"status":"unknown","answer":[],"sources":[],"clarification":""}'):
            self.assertFalse(validate(raw, self.chunks)[1]["passed"])
