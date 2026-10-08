import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from cases import load_cases
from clients import LocalClient
from core import BASELINE, Settings, digest, frozen_input, load_report, messages_for, new_report, profiles, save_report, validate_answer
from engine import Engine
from judge import DIMENSIONS, Judge, decision, quality, validate_review
from resources_monitor import ResourceMonitor, ollama_rss


CASE = {'id': 'fixture', 'question': 'Что вернёт функция?',
        'sources': [{'source_id': 'S1', 'text': 'Функция возвращает undefined.'}],
        'memory': {'notes': '', 'user_questions': [], 'prior_answers': []},
        'criteria': {'facts': ['undefined']}}
REVIEW = {**dict.fromkeys(DIMENSIONS, 2), 'critical_errors': [], 'needs_manual_review': False, 'reason': 'По источнику S1.'}


class FakeLocal:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.fail = False
        self.block = False

    async def metadata(self, model):
        return {'tag': {'name': model, 'digest': 'test-model'}, 'version': {'version': 'test'}}

    async def warmup(self, settings):
        self.calls.append(('warmup', settings.num_ctx))
        return {'wall_seconds': .1, 'cache_policy': 'test'}

    async def generate(self, messages, settings, update):
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            self.calls.append(('generate', copy.deepcopy(messages), settings))
            if self.block:
                await asyncio.sleep(60)
            if self.fail:
                raise RuntimeError('sensitive fake provider body')
            context = json.loads(messages[1]['content'])
            quote = context['sources'][0]['text'][:20]
            raw = json.dumps({'status': 'answered', 'answer': 'Ответ [S1]',
                              'citations': [{'source_id': 'S1', 'quote': quote}], 'gaps': ''}, ensure_ascii=False)
            update(raw, '')
            return {'done_reason': 'stop', 'wall_seconds': 2 if settings.thinking else 1,
                    'tokens_per_second': 50, 'input_tokens': 100, 'output_tokens': 30,
                    'first_content_seconds': .2}
        finally:
            self.active -= 1


class FakeJudge:
    available = True
    async def compare(self, case, before, after):
        return {'status': 'complete', 'orders': [
            {'reviews': {'before': copy.deepcopy(REVIEW), 'after': copy.deepcopy(REVIEW)}},
            {'reviews': {'before': copy.deepcopy(REVIEW), 'after': copy.deepcopy(REVIEW)}}]}


class FakeMonitor:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    def result(self):
        return {'max_ram_bytes': 1024, 'max_gpu_bytes': None, 'samples': []}


class CoreTests(unittest.TestCase):
    def test_unknown_reference_cannot_pass_with_a_valid_citation(self):
        response = {'status': 'answered', 'answer': 'Возвращает undefined [S1]. Другой факт [S99].',
                    'citations': [{'source_id': 'S1', 'quote': 'Функция возвращает undefined.'}], 'gaps': ''}
        _, checks = validate_answer(json.dumps(response), CASE['sources'])
        self.assertFalse(checks['passed'])
        self.assertIn('Неизвестная ссылка [S99] в ответе.', checks['errors'])
        response['answer'] = 'Возвращает undefined [S1].'
        self.assertTrue(validate_answer(json.dumps(response), CASE['sources'])[1]['passed'])

    def test_fixed_sources_and_memory_between_profiles(self):
        cases = load_cases()
        self.assertEqual([c['split'] for c in cases].count('tune'), 6)
        self.assertEqual([c['split'] for c in cases].count('holdout'), 4)
        for case in cases:
            self.assertTrue(all(s['text'] and not s['incomplete'] for s in case['sources']))
            original = copy.deepcopy(case)
            before = messages_for(case, BASELINE)
            after = messages_for(case, profiles()['compact-prompt'])
            self.assertEqual(before[1], after[1])
            self.assertNotEqual(before[0], after[0])
            self.assertEqual(case, original)
            self.assertEqual(digest(frozen_input(case)), digest(json.loads(before[1]['content'])))

    def test_small_context_does_not_truncate(self):
        case = copy.deepcopy(CASE)
        case['sources'][0]['text'] = 'Большой документ ' * 2000
        original = copy.deepcopy(case)
        with self.assertRaisesRegex(ValueError, 'не обрезаны'):
            messages_for(case, profiles()['context-8192'])
        self.assertEqual(case, original)

    def test_atomic_report_roundtrip_and_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            report = new_report('manual')
            path = save_report(report, directory)
            self.assertEqual(load_report(path), report)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(path.with_suffix('.tmp').exists())
            report['id'] = '../escape'
            with self.assertRaises(ValueError):
                save_report(report, directory)

    def test_judge_strict_types(self):
        for bad in ('{}', '[]', '{"A":{},"B":{}}', 'not json'):
            with self.assertRaises((ValueError, TypeError)):
                validate_review(bad)
        review = copy.deepcopy(REVIEW)
        review['correctness'] = True
        with self.assertRaises(ValueError):
            validate_review(json.dumps({'A': review, 'B': REVIEW}))

    def test_quality_regression_cannot_be_hidden_by_speed(self):
        review = copy.deepcopy(REVIEW)
        review['grounding'] = 1
        comparison = {'before': {'status': 'complete', 'metrics': {'wall_seconds': 10}},
                      'after': {'status': 'complete', 'metrics': {'wall_seconds': 1}},
                      'manual': {'before': REVIEW, 'after': review}}
        self.assertFalse(decision([comparison])['accepted'])
        comparison['manual']['after'] = REVIEW
        self.assertTrue(decision([comparison])['accepted'])
        comparison['after']['status'] = 'invalid'
        self.assertFalse(decision([comparison])['accepted'])

    def test_order_disagreement_requires_manual_review(self):
        b = copy.deepcopy(REVIEW)
        b['completeness'] = 1
        comparison = {'judge': {'status': 'complete', 'orders': [
            {'reviews': {'before': REVIEW, 'after': REVIEW}}, {'reviews': {'before': REVIEW, 'after': b}}]}}
        self.assertIsNone(quality(comparison, 'after'))
        comparison['manual'] = {'after': REVIEW}
        self.assertEqual(quality(comparison, 'after'), REVIEW)

    def test_rss_counts_only_ollama(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, name, rss in [(1, 'ollama', 10), (2, 'other', 100), (3, 'llama-server', 20)]:
                folder = root / str(pid)
                folder.mkdir()
                (folder / 'comm').write_text(name)
                (folder / 'status').write_text(f'VmRSS:\t{rss} kB\n')
            value, pids = ollama_rss(root)
            self.assertEqual(value, 30*1024)
            self.assertEqual(set(pids), {1, 3})


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.local = FakeLocal()
        self.engine = Engine(local=self.local, judge=FakeJudge(), monitor_factory=FakeMonitor,
                             directory=Path(self.directory.name))

    async def asyncTearDown(self):
        self.directory.cleanup()

    async def test_sequential_pair_and_order(self):
        report = await self.engine.manual(CASE, BASELINE, profiles()['compact-prompt'])
        comparison = report['comparisons'][0]
        self.assertEqual(self.local.max_active, 1)
        self.assertEqual(comparison['before']['input_sha256'], comparison['after']['input_sha256'])
        self.assertEqual(comparison['before']['messages'][1], comparison['after']['messages'][1])
        self.assertTrue(comparison['decision']['accepted'])
        self.assertIsNone(comparison['after']['metrics']['max_gpu_bytes'])
        self.assertEqual(len(list(Path(self.directory.name).glob('*.json'))), 1)

    async def test_errors_are_saved_without_sensitive_body(self):
        self.local.fail = True
        report = await self.engine.manual(CASE, BASELINE, profiles()['compact-prompt'])
        result = report['comparisons'][0]['after']
        self.assertEqual(result['status'], 'error')
        self.assertNotIn('sensitive', result['error'])
        self.assertEqual(report['comparisons'][0]['judge']['status'], 'skipped')

    async def test_cancellation_persists_partial_report(self):
        self.local.block = True
        task = asyncio.create_task(self.engine.manual(CASE, BASELINE, profiles()['no-thinking']))
        for _ in range(50):
            if self.local.active:
                break
            await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.engine.report['status'], 'cancelled')
        self.assertEqual(self.engine.report['comparisons'][0]['before']['status'], 'cancelled')
        restored = load_report(next(Path(self.directory.name).glob('*.json')))
        self.assertEqual(restored['status'], 'cancelled')

    async def test_context_limit_skips_model(self):
        case = copy.deepcopy(CASE)
        case['sources'][0]['text'] *= 4000
        result = await self.engine.run_one(case, profiles()['context-8192'])
        self.assertEqual(result['status'], 'context_limit')
        self.assertEqual(self.local.calls, [])

    async def test_cloud_judge_order_mapping_and_invalid_output(self):
        class Client:
            def __init__(self):
                self.inputs = []
            async def generate(self, messages, settings, update):
                self.inputs.append(json.loads(messages[1]['content']))
                update(json.dumps({'A': REVIEW, 'B': REVIEW}), '')
                return {'done_reason': 'stop'}
        client = Client()
        record = await Judge(client=client).compare(CASE, {'raw': 'first'}, {'raw': 'second'})
        self.assertEqual(record['status'], 'complete')
        self.assertEqual(client.inputs[0]['answers'], {'A': 'first', 'B': 'second'})
        self.assertEqual(client.inputs[1]['answers'], {'A': 'second', 'B': 'first'})
        self.assertNotIn('baseline', json.dumps(client.inputs))
        async def malformed(messages, settings, update):
            update('{}', '')
            return {'done_reason': 'stop'}
        client.generate = malformed
        record = await Judge(client=client).compare(CASE, {'raw': 'first'}, {'raw': 'second'})
        self.assertEqual(record['status'], 'error')

    async def test_resource_monitor_missing_gpu(self):
        async def sampler():
            return {'ram_bytes': 123, 'gpu_bytes': None}
        async with ResourceMonitor(sampler) as monitor:
            await asyncio.sleep(.01)
        result = monitor.result()
        self.assertIsNone(result['max_gpu_bytes'])
        self.assertEqual(result['max_ram_bytes'], 123)
        self.assertGreaterEqual(len(result['samples']), 2)

    async def test_client_payload_and_warmup(self):
        bodies = []
        def handler(request):
            if request.url.path == '/api/ps':
                return httpx.Response(200, json={'models': [{'name': 'qwen3:14b'}]})
            body = json.loads(request.content)
            bodies.append(body)
            if not body['messages']:
                return httpx.Response(200, json={'done': True, 'load_duration': 1000})
            return httpx.Response(200, text=json.dumps({'message': {'content': '{}'}, 'done': True,
                'done_reason': 'stop', 'eval_count': 3, 'eval_duration': 100000000}) + '\n')
        client = LocalClient(httpx.MockTransport(handler))
        settings = Settings(thinking=False, temperature=.2, num_ctx=8192, num_predict=1536)
        await client.warmup(settings)
        await client.generate([{'role': 'user', 'content': 'test'}], settings, lambda a,b: None)
        self.assertEqual(bodies[0]['keep_alive'], 0)
        self.assertEqual(bodies[1]['options']['num_ctx'], 8192)
        self.assertEqual(bodies[2]['options'], {'temperature': .2, 'num_ctx': 8192, 'num_predict': 1536, 'seed': 42})
        self.assertFalse(bodies[2]['think'])
        self.assertEqual(bodies[2]['format']['required'], ['status', 'answer', 'gaps', 'citations'])

    async def test_batch_holds_out_final_cases_and_resume_does_not_rerun(self):
        def cases(split):
            return [dict(copy.deepcopy(CASE), id=split+'-1', split=split)]
        with patch('engine.load_cases', cases):
            report = await self.engine.experiment()
            self.assertEqual(len(report['steps']), 6)
            self.assertEqual(len(report['final']['comparisons']), 3)
            for step in report['steps']:
                self.assertTrue(all(report['comparisons'][i]['case_id'] == 'tune-1' for i in step['comparisons']))
            finals = [report['comparisons'][i] for i in report['final']['comparisons']]
            self.assertEqual([c['order'] for c in finals], [['before', 'after'], ['after', 'before'], ['before', 'after']])
            call_count = len(self.local.calls)
            await self.engine.experiment(report)
            self.assertEqual(len(self.local.calls), call_count)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_uses_archived_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(local=FakeLocal(), judge=FakeJudge(), monitor_factory=FakeMonitor,
                            directory=Path(directory))
            def cases(split):
                return [dict(copy.deepcopy(CASE), id=split+'-1')]
            with patch('engine.load_cases', cases):
                report = await engine.experiment()
            with patch('engine.load_cases', side_effect=AssertionError('must not reload dataset')):
                await engine.experiment(report)
            self.assertEqual(report['status'], 'complete')
            report['cases'][0]['question'] = 'changed'
            await engine.experiment(report)
            self.assertEqual(report['status'], 'error')

    async def test_identical_final_profiles_are_not_called_optimization(self):
        class ConstantTimeLocal(FakeLocal):
            async def generate(self, messages, settings, update):
                result = await super().generate(messages, settings, update)
                result['wall_seconds'] = 2
                return result
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(local=ConstantTimeLocal(), judge=FakeJudge(), monitor_factory=FakeMonitor,
                            directory=Path(directory))
            def cases(split):
                return [dict(copy.deepcopy(CASE), id=split+'-1')]
            with patch('engine.load_cases', cases):
                report = await engine.experiment()
            self.assertEqual(report['winner'], BASELINE)
            self.assertEqual(report['final']['decision']['status'], 'no_change')
            self.assertFalse(report['final']['decision']['accepted'])

    async def test_manual_schema_retrieval_happens_once(self):
        class Docs:
            calls = []
            async def retrieve(self, queries, settings, notify):
                self.calls.append((queries, settings.source))
                return {'sources': copy.deepcopy(CASE['sources']), 'server': {'version': 'test'}}
        docs = Docs()
        engine = Engine(docs=docs)
        case = await engine.retrieve_manual('collaborator', 'SQL/XML', 'schema')
        self.assertEqual(docs.calls, [(['collaborator'], 'schema')])
        self.assertEqual(case['memory']['notes'], 'SQL/XML')
        self.assertEqual(case['sources'], CASE['sources'])

    async def test_manual_review_resume_reuses_generations(self):
        class DisputedJudge(FakeJudge):
            async def compare(self, case, before, after):
                result = await super().compare(case, before, after)
                result['orders'][1]['reviews']['after']['grounding'] = 1
                return result
        with tempfile.TemporaryDirectory() as directory:
            local = FakeLocal()
            engine = Engine(local=local, judge=DisputedJudge(), monitor_factory=FakeMonitor, directory=Path(directory))
            def cases(split):
                return [dict(copy.deepcopy(CASE), id=split+'-1')]
            with patch('engine.load_cases', cases):
                report = await engine.experiment()
                self.assertEqual(report['status'], 'manual_review')
                self.assertEqual(len(report['steps']), 1)
                self.assertIsNone(report['final'])
                local_calls = len(local.calls)
                comparison = report['comparisons'][0]
                comparison['manual'] = {'before': copy.deepcopy(REVIEW), 'after': copy.deepcopy(REVIEW)}
                engine.judge = FakeJudge()
                # The reviewed first comparison must not be regenerated.
                await engine.experiment(report)
                first = report['comparisons'][0]
                self.assertIs(first, comparison)
                self.assertEqual(first['manual']['after']['grounding'], 2)
                self.assertEqual(report['steps'][0]['status'], 'accepted')
                self.assertEqual(len(report['final']['comparisons']), 3)
                self.assertGreater(len(local.calls), local_calls)

    async def test_monitor_failure_is_missing_data_not_generation_failure(self):
        async def unavailable():
            raise OSError('unavailable')
        async with ResourceMonitor(unavailable) as monitor:
            await asyncio.sleep(.01)
        self.assertIsNone(monitor.result()['max_gpu_bytes'])
        self.assertIsNone(monitor.result()['max_ram_bytes'])
        self.assertTrue(monitor.result()['samples'][0]['sample_error'])


if __name__ == '__main__':
    unittest.main()
