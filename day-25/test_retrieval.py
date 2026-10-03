import copy
import json
import unittest

from memory import empty_state, validate_plan, prune_resolved_questions, plan_messages
from retrieval import search_candidates, rerank_candidates, select_context


SEARCHES = [{'query': 'стратегии управления контекстом', 'source': 'day-10/README.md'},
            {'query': 'слои памяти новая сессия новая задача', 'source': 'day-11/README.md'}]


class ScopedRetriever:
    metadata = {'test': True}
    def __init__(self, path=None):
        self.calls = []
    def load(self):
        pass
    def search(self, query, k, source=None):
        self.calls.append((query, k, source))
        return [{'chunk_id': f'{source}:{i}', 'source': source, 'section': 'Память',
                 'text': 'Документированный факт', 'score': 1 - i * .1} for i in range(k)]


class ScoredReranker:
    def __init__(self):
        self.queries = []
    def rank(self, query, chunks):
        self.queries.append(query)
        base = .99 if query == SEARCHES[0]['query'] else .2
        return [{**c, 'rerank_score': base - int(c['chunk_id'].rsplit(':', 1)[1]) * .01} for c in chunks]


class RetrievalTests(unittest.TestCase):
    def test_pipeline_uses_structured_searches_and_closes_answered_question(self):
        import tempfile
        from pathlib import Path
        from pipeline import Runner, new_session, load_session

        question = 'Сравни память дней 10 и 11 без кода.'
        state = empty_state()
        state['open_questions'] = [{'text': 'Сравнение памяти', 'turn': 1, 'quote': question}]
        data = {'query': 'память дней 10 и 11', 'resolved_question': question,
                'state': state, 'searches': SEARCHES}
        calls = []

        def completion(messages, key, model):
            payload = json.loads(messages[1]['content'])
            calls.append(payload)
            if 'current_turn' in payload:
                result = data
            else:
                chunks = payload['sources']
                result = {'status': 'answered', 'clarification': '',
                          'answer': [{'text': c['text'], 'citations': [{'source_id': c['source_id'], 'quote': c['text']}]} for c in chunks],
                          'sources': [{k: c[k] for k in ('source_id', 'source', 'section', 'chunk_id')} for c in chunks]}
            return {'answer': json.dumps(result, ensure_ascii=False), 'usage': {}, 'llm_seconds': 0}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dialog.json'
            session = new_session()
            runner = Runner(key='fake', completion=completion, retriever_factory=ScopedRetriever, reranker_factory=ScoredReranker)
            turn = runner.send(session, question, path)
            self.assertEqual(turn['status'], 'complete')
            self.assertEqual(runner.reranker.queries, [s['query'] for s in SEARCHES])
            self.assertEqual({s['source'] for s in calls[-1]['sources']}, {s['source'] for s in SEARCHES})
            self.assertEqual(session['state']['open_questions'], [])
            self.assertTrue(turn['plan']['state']['open_questions'])
            self.assertEqual(load_session(path), session)

    def test_comparison_keeps_both_sources_without_diluting_rerank_question(self):
        retriever, reranker = ScopedRetriever(), ScoredReranker()
        candidates = search_candidates(retriever, SEARCHES, 8)
        ranked = rerank_candidates(reranker, SEARCHES, candidates)
        annotated, selected = select_context(ranked, 3, .1, 2)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(retriever.calls, [(s['query'], 4, s['source']) for s in SEARCHES])
        self.assertEqual(reranker.queries, [s['query'] for s in SEARCHES])
        self.assertEqual({c['source'] for c in selected}, {'day-10/README.md', 'day-11/README.md'})
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({c['source_id'] for c in selected}), 3)
        self.assertTrue(all(c['rerank_score'] >= .1 for c in selected))
        self.assertEqual(sum(c['decision'] == 'В контексте' for c in annotated), 3)

    def test_coverage_never_bypasses_threshold(self):
        ranked = rerank_candidates(ScoredReranker(), SEARCHES, search_candidates(ScopedRetriever(), SEARCHES, 8))
        _, selected = select_context(ranked, 3, .9, 2)
        self.assertTrue(all(c['source'] == 'day-10/README.md' for c in selected))
        _, selected = select_context(ranked, 3, 1, 2)
        self.assertEqual(selected, [])

    def test_queries_are_deduplicated_and_budget_is_shared(self):
        class Shared(ScopedRetriever):
            def search(self, query, k, source=None):
                return [{'chunk_id': 'shared', 'source': 'day-11/README.md', 'section': 'Память',
                         'text': 'Факт', 'score': .8}]
        searches = [{'query': 'память', 'source': None}, {'query': 'сессия', 'source': None}]
        candidates = search_candidates(Shared(), searches, 2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['query_ids'], [0, 1])
        self.assertLessEqual(len(search_candidates(ScopedRetriever(), SEARCHES, 1)), 1)

    def test_source_scope_cannot_reference_secrets_or_outside_corpus(self):
        state = empty_state()
        data = {'query': 'память', 'resolved_question': 'Где хранится рабочая память?',
                'state': state, 'searches': [{'query': 'рабочая память файл хранения', 'source': 'day-11/README.md'}]}
        turns = [{'number': 1, 'question': 'Где хранится рабочая память дня 11?'}]
        self.assertEqual(validate_plan(json.dumps(data), turns, state)['searches'], data['searches'])
        for source in ('../.env', 'day-11/.env', 'day-25/README.md'):
            bad = copy.deepcopy(data)
            bad['searches'][0]['source'] = source
            with self.assertRaises(ValueError):
                validate_plan(json.dumps(bad), turns, state)

    def test_memory_removes_only_accepted_complete_questions_and_exposes_old_outcomes(self):
        state = empty_state()
        state['open_questions'] = [{'text': f'Вопрос {i}', 'turn': i, 'quote': f'Вопрос {i}'} for i in (1, 2, 3)]
        turns = [
            {'number': 1, 'question': 'Вопрос 1', 'checks': {'passed': True}, 'response': {'status': 'answered', 'clarification': ''}},
            {'number': 2, 'question': 'Вопрос 2', 'checks': {'passed': True}, 'response': {'status': 'answered', 'clarification': 'Не хватает данных'}},
            {'number': 3, 'question': 'Вопрос 3', 'checks': {'passed': False}, 'response': None},
        ]
        result = prune_resolved_questions(state, turns)
        self.assertEqual([e['turn'] for e in result['open_questions']], [2, 3])
        self.assertEqual(len(state['open_questions']), 3)
        request = json.loads(plan_messages('Итог?', 12, state, [], turns)[1]['content'])
        self.assertEqual([o['turn'] for o in request['question_outcomes']], [1, 2, 3])
        self.assertTrue(request['question_outcomes'][0]['accepted'])
