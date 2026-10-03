import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmark import run_scenario
from evidence import validate
from memory import empty_state, validate_plan
from pipeline import Runner, Settings, load_session, new_session
from scenarios import SCENARIOS


class FakeRetriever:
    metadata = {'corpus': 'test'}
    def __init__(self, path):
        self.queries = []
    def load(self):
        pass
    def search(self, query, k):
        self.queries.append(query)
        return [{'source': 'day-11/README.md', 'section': 'Память', 'chunk_id': 'c1',
                 'text': 'Рабочая память сохраняется между сессиями.', 'score': .8}]


class FakeReranker:
    def __init__(self):
        self.queries = []
    def rank(self, query, chunks):
        self.queries.append(query)
        return [{**c, 'rerank_score': .8} for c in chunks]


def scripted_completion(messages, key, model):
    payload = json.loads(messages[1]['content'])
    if 'current_turn' in payload:
        question = payload['question']
        case = next(c for s in SCENARIOS for c in s['turns'] if c['question'] == question)
        data = {'query': case['resolved'], 'resolved_question': case['resolved'],
                'state': case['expected_state'], 'searches': [{'query': case['resolved'], 'source': None}]}
    elif 'expected_state' in payload:
        data = {'checks': {'support': True, 'memory': True, 'continuity': True},
                'verdict': 'pass', 'explanation': 'Синтетическая оценка для проверки кода'}
    else:
        chunk = payload['sources'][0]
        data = {'status': 'answered', 'answer': [{'text': chunk['text'],
                'citations': [{'source_id': chunk['source_id'], 'quote': chunk['text']}]}],
                'sources': [{k: chunk[k] for k in ('source_id', 'source', 'section', 'chunk_id')}],
                'clarification': ''}
    return {'answer': json.dumps(data, ensure_ascii=False), 'usage': {'total_tokens': 10}, 'llm_seconds': 0}


def factory(**kwargs):
    return Runner(**kwargs, completion=scripted_completion,
                  retriever_factory=FakeRetriever, reranker_factory=FakeReranker)


class ChatTests(unittest.TestCase):
    def test_both_long_scenarios_state_replacement_retrieval_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            for scenario in SCENARIOS:
                runner = factory(key='fake')
                path = Path(tmp) / (scenario['id'] + '.json')
                session = run_scenario(runner, scenario, path, validation_only=True)
                self.assertEqual(len(session['turns']), 12)
                self.assertTrue(all(t['status'] == 'complete' for t in session['turns']))
                self.assertEqual(session['state'], scenario['turns'][-1]['expected_state'])
                self.assertEqual(len(runner.retriever.queries), 12)
                self.assertEqual(runner.reranker.queries[-1], scenario['turns'][-1]['resolved'])
                self.assertEqual(load_session(path), session)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                request = json.loads(session['turns'][-1]['answer_messages'][1]['content'])
                self.assertEqual(len(request['history']), 3)
                self.assertEqual(request['task_state']['goal'][0]['turn'], 1)
                self.assertEqual(request['task_state']['constraints'][0]['turn'], 9)
                self.assertNotIn('expected_state', request)
                self.assertTrue(all(t['assessment']['result']['verdict'] == 'pass' for t in session['turns']))

    def test_false_memory_provenance_rejected_before_retrieval(self):
        def bad(messages, key, model):
            out = scripted_completion(messages, key, model)
            data = json.loads(out['answer'])
            data['state']['goal'][0]['quote'] = 'слова ассистента, которых пользователь не писал'
            out['answer'] = json.dumps(data)
            return out
        with tempfile.TemporaryDirectory() as tmp:
            session = new_session()
            runner = Runner(key='fake', completion=bad, retriever_factory=lambda _: self.fail('Search started'))
            t = runner.send(session, SCENARIOS[0]['turns'][0]['question'], Path(tmp) / 's.json')
            self.assertEqual(t['status'], 'failed')
            self.assertEqual(session['state'], empty_state())
            self.assertEqual(t['error']['stage'], 'validate')

    def test_unknown_empty_sources_and_no_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = new_session(settings=Settings(rerank_threshold=1))
            runner = factory(key='fake')
            t = runner.send(session, SCENARIOS[0]['turns'][0]['question'], Path(tmp) / 's.json')
            self.assertEqual(t['response']['status'], 'unknown')
            self.assertEqual(t['response']['sources'], [])
            self.assertNotIn('answer_output', t)
            self.assertTrue(t['response']['clarification'])

    def test_invalid_answer_not_accepted_and_task_state_survives_failure(self):
        def invalid(messages, key, model):
            if 'sources' in json.loads(messages[1]['content']):
                return {'answer': 'bad-json', 'usage': {}}
            return scripted_completion(messages, key, model)
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(key='fake', completion=invalid, retriever_factory=FakeRetriever, reranker_factory=FakeReranker)
            session = new_session()
            path = Path(tmp) / 's.json'
            t = runner.send(session, SCENARIOS[0]['turns'][0]['question'], path)
            self.assertEqual(t['status'], 'invalid')
            self.assertIsNone(t['response'])
            self.assertTrue(load_session(path)['state']['goal'])

    def test_cancellation_before_network_is_persisted_and_sessions_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = new_session()
            path = Path(tmp) / 's.json'
            t = factory(key='fake').send(session, 'test', path, cancelled=lambda: True)
            self.assertEqual(t['status'], 'cancelled')
            self.assertEqual(load_session(path)['turns'][0]['status'], 'cancelled')
            self.assertEqual(new_session()['state'], empty_state())

    def test_invented_old_provenance_cannot_modify_existing_memory(self):
        turns = [{'number': 1, 'question': 'Цель: сравнение'}, {'number': 2, 'question': 'Уточнение'}]
        state = empty_state()
        state['goal'] = [{'text': 'новая интерпретация', 'turn': 1, 'quote': 'сравнение'}]
        raw = json.dumps({'query': 'q', 'resolved_question': 'q', 'state': state,
                          'searches': [{'query': 'q', 'source': None}]})
        with self.assertRaises(ValueError):
            validate_plan(raw, turns, empty_state())

    def test_reference_to_old_response_does_not_validate_current_evidence(self):
        data = {'status': 'answered', 'answer': [{'text': 'a', 'citations': [{'source_id': 'S1', 'quote': 'old'}]}],
                'sources': [{'source_id': 'S1', 'source': 'old', 'section': 'old', 'chunk_id': 'old'}], 'clarification': ''}
        _, checks = validate(json.dumps(data), [])
        self.assertFalse(checks['passed'])
