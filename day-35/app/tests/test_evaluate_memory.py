import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from personal_assistent import evaluate_memory as evaluation
from personal_assistent.storage import Store


def tool(name, arguments, identity):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': identity, 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class ScriptedModel:
    max_tokens = 2000

    def __init__(self, *responses):
        self.responses = list(responses)
        self.retrieval_fixtures = list(responses)

    async def complete(self, messages):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def retrieve(self, messages):
        from retrieval_stub import retrieval_reply
        return retrieval_reply(messages, self)


class DatasetTests(unittest.TestCase):
    def test_frozen_dataset_has_eight_groups_and_valid_evidence(self):
        data, digest = evaluation.load_dataset()
        self.assertEqual(len(data['notes']), 45)
        self.assertEqual(len(digest), 64)
        self.assertEqual(set(evaluation.Counter(c['group'] for c in data['cases']).values()), {4})
        self.assertEqual([c['id'] for c in data['cases']], [f'Q{i:02}' for i in range(1, 33)])
        for identity, target in [('N39', '90 дней'), ('N40', 'KZN-4827')]:
            text = next(n['content'] for n in data['notes'] if n['id'] == identity)
            self.assertNotIn(target, text[:160])
        self.assertFalse(any('90 дней' in n['content'] for n in data['notes'] if n['id'] != 'N39'))

    def test_each_store_is_fresh_and_preserves_dates_and_source_text(self):
        data, _ = evaluation.load_dataset()
        with evaluation.memory_store(data) as store:
            row = dict(store.db.execute('SELECT * FROM notes WHERE id=38').fetchone())
            self.assertEqual(row['event_date'], '2026-10-12')
            self.assertEqual(row['created_at'], evaluation.timestamp('2026-10-14T09:00:00+03:00'))
            self.assertEqual(row['original'], row['content'])
            store.db.execute('DELETE FROM notes WHERE id=38')
        with evaluation.memory_store(data) as store:
            self.assertEqual(store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 45)
            self.assertEqual(store.history(), [])
            self.assertEqual(store.db.execute('PRAGMA database_list').fetchone()['file'], '')

    def test_default_command_is_offline_even_with_bot_database_configured(self):
        with tempfile.TemporaryDirectory() as folder:
            personal = Path(folder) / 'personal.sqlite3'
            personal.write_bytes(b'private database sentinel')
            output = Path(folder) / 'report.json'
            with (patch.dict('os.environ', {'ASSISTANT_DB': str(personal)}),
                  patch.object(evaluation, 'load_dotenv', side_effect=AssertionError('Must not load secrets')),
                  patch.object(evaluation.httpx, 'AsyncClient', side_effect=AssertionError('Network forbidden')),
                  patch.object(evaluation, 'Store', wraps=Store) as constructor,
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(evaluation.main(['--output', str(output)]), 0)
            self.assertEqual(constructor.call_count, 32)
            self.assertTrue(all(c.args == (':memory:',) for c in constructor.call_args_list))
            self.assertEqual(personal.read_bytes(), b'private database sentinel')
            report = json.loads(output.read_text())
            self.assertEqual(report['api_calls_attempted'], 0)
            self.assertEqual(len(report['items']), 32)
            self.assertEqual(report['mode'], 'first_fts_search')
            self.assertTrue(all(i['answer'] is None and i['review']['answer'] is None for i in report['items']))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'existing.json'
            output.write_text('keep me')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main(['--output', str(output)])
            self.assertEqual(output.read_text(), 'keep me')

    def test_invalid_case_does_not_create_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'report.json'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main(['--case', 'Q99', '--output', str(output)])
            self.assertFalse(output.exists())


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data, self.digest = evaluation.load_dataset()

    async def test_full_questions_find_workplace_and_coffee_machine_history(self):
        for identity in ('Q14', 'Q17'):
            with self.subTest(question=identity):
                case = next(c for c in self.data['cases'] if c['id'] == identity)
                result = await evaluation.evaluate_case(self.data, case)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['first_search']['source_coverage']['missing'], [])

    async def test_actual_agent_sees_preview_then_original_without_expected_answer(self):
        case = self.data['cases'][27]
        model = ScriptedModel(
            tool('search_notes', {'query': 'Казань'}, 'search'),
            tool('get_note', {'id': 40}, 'original'),
            {'role': 'assistant', 'content': json.dumps({'kind': 'answer', 'text': 'KZN-4827'})})
        result = await evaluation.evaluate_case(self.data, case, model=model, allowance={'remaining': 6})
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], 'KZN-4827')
        self.assertEqual(result['first_search']['sources'], ['N40'])
        self.assertEqual(result['model_sources'], ['N40'])
        self.assertEqual(result['tool_counts'], {'search_notes': 1, 'get_note': 1})
        preview_messages = result['model_calls'][1]['messages']
        original_messages = result['model_calls'][2]['messages']
        self.assertNotIn('KZN-4827', json.dumps(preview_messages, ensure_ascii=False))
        self.assertIn('KZN-4827', json.dumps(original_messages, ensure_ascii=False))
        for call in result['model_calls']:
            text = json.dumps(call['messages'], ensure_ascii=False)
            for field in ('expected_answer', 'critical_errors', 'required_facts'):
                self.assertNotIn(field, text)
        self.assertIsNone(result['review']['grounding'])

    async def test_shared_limit_stops_before_another_model_request(self):
        allowance = {'remaining': 1}
        model = ScriptedModel(tool('search_notes', {'query': 'Казань'}, 'search'))
        result = await evaluation.evaluate_case(self.data, self.data['cases'][27], model=model, allowance=allowance)
        self.assertEqual(result['error_type'], 'CallLimit')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['model_calls']), 1)
        self.assertEqual(allowance['remaining'], 0)
        # A raw tool result in a checkpoint is not proof it reached the model.
        self.assertEqual(result['model_sources'], [])
        self.assertEqual(result['first_search']['sources'], ['N40'])

    async def test_timeout_keeps_tool_trace_and_failed_status(self):
        class SlowModel:
            max_tokens = 2000

            async def complete(self, messages):
                await asyncio.sleep(10)

        result = await evaluation.evaluate_case(self.data, self.data['cases'][0], model=SlowModel(),
                                                allowance={'remaining': 2}, timeout=0.01)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertEqual(result['model_calls'][0]['error_type'], 'CancelledError')

    async def test_partial_report_survives_failure_and_has_no_exception_details(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            args = type('Args', (), dict(case=['Q01', 'Q02', 'Q03'], live=True, temperature=0,
                                        max_tokens=2000, context_budget=60000, context_reserve=2000,
                                        max_calls=10, case_timeout=120))()
            model = ScriptedModel(
                {'role': 'assistant', 'content': json.dumps({'kind': 'answer', 'text': '12 сентября 2026'})},
                RuntimeError('secret error body'))
            with (patch.object(evaluation, 'DeepSeek', return_value=model),
                  patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'fake-test-key'}),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertFalse(await evaluation.run(args, self.data, self.digest, path))
            report = json.loads(path.read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual([i['status'] for i in report['items']], ['complete', 'failed'])
            self.assertEqual(report['api_calls_attempted'], 2)
            self.assertIsNone(report['items'][0]['tokens']['prompt_tokens'])
            self.assertNotIn('secret error body', path.read_text())
            self.assertNotIn('fake-test-key', path.read_text())

    async def test_summary_requests_share_call_allowance(self):
        class SummaryModel(ScriptedModel):
            async def summarize(self, messages):
                return {'role': 'assistant', 'content': 'summary'}

        wrapper = evaluation.TracedModel(SummaryModel(), {'remaining': 1})
        await wrapper.summarize([{'role': 'user', 'content': 'synthetic'}])
        with self.assertRaises(evaluation.CallLimit):
            await wrapper.complete([])
        self.assertEqual(wrapper.trace[0]['purpose'], 'summary')
