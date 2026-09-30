import json
import tempfile
import unittest
from pathlib import Path

from pipeline import Runner, Settings, rewrite_question, select_chunks
from session_log import SessionLog


CASES = [{"id": "q1", "question": "Исходный вопрос", "expected": [], "sources": []}]


def chunk(number, score):
    return {"chunk_id": str(number), "source": "day-20/README.md", "title": "Отчёт",
            "section": "Проверка", "text": f"Фрагмент {number}", "score": score}


class FakeRetriever:
    metadata = {"corpus_sha256": "fixture"}
    calls = []

    def __init__(self, path):
        pass

    def load(self):
        pass

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return [chunk(1, 0.8), chunk(2, 0.5), chunk(3, 0.2)][:top_k]


class FakeReranker:
    def rank(self, question, chunks):
        assert question == "Исходный вопрос"
        scores = {"1": 0.05, "2": 0.8, "3": 0.9}
        return sorted([{**row, "rerank_score": scores[row['chunk_id']]} for row in chunks],
                      key=lambda row: -row['rerank_score'])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        FakeRetriever.calls = []

    def test_unchanged_rewrite_is_reported_without_retries(self):
        calls = []

        def completion(messages, key, model):
            calls.append(messages)
            return {'answer': '{"query": "  Исходный   вопрос  "}', 'usage': {}, 'llm_seconds': .1}

        result = rewrite_question('Исходный вопрос', key='test', model='test', completion=completion)
        self.assertFalse(result['changed'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0][1]['content'])['question'], 'Исходный вопрос')

    def test_rewrite_invalid_query_does_not_silently_fall_back_to_original(self):
        for query in ('', None, [], 'x' * 2001):
            with self.subTest(query_type=type(query).__name__):
                def completion(messages, key, model):
                    return {'answer': json.dumps({'query': query}), 'usage': {}, 'llm_seconds': 0}
                with self.assertRaises(ValueError):
                    rewrite_question('Исходный вопрос', key='test', model='test', completion=completion)

    def test_four_modes_share_rewrite_and_candidates_but_not_answers(self):
        calls = []

        def complete(messages, key, model):
            calls.append(messages)
            rewrite = "query" in messages[0]["content"]
            return {"answer": '{"query": "Поисковая формулировка"}' if rewrite else "Ответ [S1]",
                    "usage": {"total_tokens": 12}, "llm_seconds": 0.01, "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            runner = Runner(key="secret", settings=Settings(3, 2, 0.6, 0.1), completion=complete,
                            retriever_factory=FakeRetriever, reranker_factory=FakeReranker)
            updates = []
            report = runner.run(CASES, path, SessionLog(Path(directory) / "logs"),
                                lambda report: updates.append(json.loads(json.dumps(report))))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(calls), 5)
            self.assertEqual(FakeRetriever.calls, [("Исходный вопрос", 2), ("Поисковая формулировка", 3)])
            results = report['items'][0]['results']
            self.assertTrue(report['items'][0]['rewrite']['changed'])
            self.assertEqual([c['chunk_id'] for c in results['rewrite_filter']['chunks']], ['1'])
            self.assertEqual([c['chunk_id'] for c in results['rewrite_rerank']['chunks']], ['3', '2'])
            self.assertEqual(results['rewrite_rerank']['candidates'][0]['rank_before'], 3)
            for messages in (calls[0], *calls[2:]):
                self.assertEqual(json.loads(messages[1]['content'])['question'], 'Исходный вопрос')
                self.assertEqual(len(messages), 2)
            self.assertTrue(any(update['items'][0]['results'].get('rewrite_rerank', {}).get('chunks')
                                and 'answer' not in update['items'][0]['results']['rewrite_rerank'] for update in updates))
            self.assertNotIn('secret', path.read_text())
            self.assertEqual(json.loads(path.read_text())['status'], 'complete')

    def test_failure_keeps_baseline_and_records_failed_stage_without_message(self):
        count = 0

        def fail_rewrite(messages, key, model):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('secret provider response')
            return {'answer': 'Baseline', 'usage': {}, 'llm_seconds': 0.01}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            log = SessionLog(Path(directory) / 'logs')
            report = Runner(key='secret', completion=fail_rewrite, retriever_factory=FakeRetriever,
                            reranker_factory=FakeReranker).run(CASES, path, log)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error']['stage'], 'rewrite')
            self.assertEqual(report['items'][0]['results']['baseline']['answer'], 'Baseline')
            self.assertNotIn('secret', log.path.read_text())
            self.assertNotIn('secret', path.read_text())
            self.assertIn('stage_failed', log.path.read_text())

    def test_empty_filtered_context_still_answers_with_original_question(self):
        annotated, selected = select_chunks([chunk(1, 0.2)], 5, 0.8)
        self.assertEqual(selected, [])
        self.assertEqual(annotated[0]['decision'], 'Ниже порога')
        self.assertEqual(select_chunks([chunk(1, 0.8)], 1, 0.8)[1][0]['source_id'], 'S1')

    def test_cancellation_stops_before_another_paid_call_and_saves_results(self):
        calls = []

        def complete(messages, key, model):
            calls.append(messages)
            return {'answer': 'Первый ответ', 'usage': {}, 'llm_seconds': 0}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            report = Runner(key='test', completion=complete, retriever_factory=FakeRetriever,
                            reranker_factory=FakeReranker).run(
                                CASES, path, SessionLog(Path(directory) / 'logs'),
                                cancelled=lambda: bool(calls))
            self.assertEqual(len(calls), 1)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error']['type'], 'InterruptedError')
            self.assertEqual(json.loads(path.read_text())['items'][0]['results']['baseline']['answer'], 'Первый ответ')

    def test_empty_context_is_passed_to_llm_not_replaced_by_unfiltered_chunks(self):
        requests = []

        def complete(messages, key, model):
            requests.append(messages)
            return {'answer': '{"query": "Поисковая формулировка"}' if 'query' in messages[0]['content'] else 'Неизвестно',
                    'usage': {}, 'llm_seconds': 0}

        with tempfile.TemporaryDirectory() as directory:
            report = Runner(key='test', settings=Settings(3, 2, 1, 1), completion=complete,
                            retriever_factory=FakeRetriever, reranker_factory=FakeReranker).run(
                                CASES, Path(directory) / 'report.json', SessionLog(Path(directory) / 'logs'))
            self.assertEqual(report['status'], 'complete')
            for request in requests[-2:]:
                payload = json.loads(request[1]['content'])
                self.assertEqual(payload['sources'], [])
                self.assertEqual(payload['question'], 'Исходный вопрос')

    def test_invalid_settings(self):
        for settings in ((2, 3, .3, .1), (20, 5, float('nan'), .1), (20, 5, .3, 2)):
            with self.assertRaises(ValueError):
                Settings(*settings)
