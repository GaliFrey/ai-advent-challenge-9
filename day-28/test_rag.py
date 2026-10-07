import asyncio
import copy
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import httpx
from textual.widgets import Button, Input, Select, Static, Switch, TextArea

from clients import CloudClient, DocsClient, LocalClient
from core import Settings, fit_sources, load_session, memory_snapshot, new_session, parse_plan, prepare_messages, provider_memory, save_session, validate_answer
from pipeline import Runner
from tui import RagApp, memory_text


def source():
    return {'source_id': 'S1', 'ref': 'local:doc', 'title': 'ArrayOptFirstElem', 'corpus': 'Datex',
            'chunk_id': 'x', 'heading': 'Описание', 'text': 'Пустой массив возвращает undefined.', 'incomplete': False, 'pages': []}


def answer():
    return json.dumps({'status': 'answered', 'answer': 'Для пустого массива возвращается undefined. [S1]',
        'citations': [{'source_id': 'S1', 'quote': 'Пустой массив возвращает undefined.'}], 'gaps': ''}, ensure_ascii=False)


class FakeDocs:
    def __init__(self, sources=None):
        self.calls = []
        self.sources = [source()] if sources is None else sources

    async def retrieve(self, queries, settings, notify):
        self.calls.append(queries)
        notify('Поиск')
        return {'sources': copy.deepcopy(self.sources), 'searches': [], 'server': {}, 'candidates': []}


class FakeModel:
    def __init__(self, failure=False, block=False, raw=None):
        self.calls, self.failure, self.block, self.raw = [], failure, block, raw
        self.entered = asyncio.Event()

    async def generate(self, messages, settings, update, *, planning=False):
        self.calls.append(copy.deepcopy(messages))
        if planning:
            raw = json.dumps({'question': 'ArrayOptFirstElem на пустом массиве', 'queries': ['ArrayOptFirstElem']})
        else:
            self.entered.set()
            if self.failure:
                raise ValueError('Тестовая ошибка модели')
            update('частичный' if self.block else (self.raw or answer()), '')
            if self.block:
                await asyncio.Event().wait()
            return {'done_reason': 'stop', 'wall_seconds': .1, 'input_tokens': 100,
                    'output_tokens': 50, 'first_content_seconds': .01}
        update(raw, '')
        return {'done_reason': 'stop'}


class CoreTests(unittest.TestCase):
    def test_shared_sources_fit_multibyte_text_and_preserve_provider_history(self):
        session = new_session()
        session['turns'] = [{'question': 'Первый вопрос', 'results': {
            name: {'model': name, 'status': 'complete', 'response': {'answer': name * 100, 'gaps': ''}}
            for name in ('local', 'cloud')}}]
        memory = memory_snapshot(session)
        sources = [{**source(), 'source_id': f'S{i}', 'text': 'Документация. ' * 1000} for i in range(4)]
        original = copy.deepcopy(sources)
        settings = Settings()
        fitted, remembered = fit_sources('Уточнение', memory, sources, settings)
        self.assertEqual(sources, original)
        self.assertEqual(remembered, memory)
        self.assertEqual([s['source_id'] for s in fitted], [s['source_id'] for s in sources])
        for name in ('local', 'cloud'):
            messages, context = prepare_messages('Уточнение', provider_memory(remembered, name), fitted, settings)
            self.assertLessEqual(context['input_bytes'], context['byte_budget'])
            self.assertEqual(context['memory']['excluded_questions'], 0)
        for s in fitted:
            self.assertTrue(s['incomplete'])
            self.assertTrue(s['budget_truncated'])
            self.assertNotIn('\ufffd', s['text'])
        with self.assertRaisesRegex(ValueError, 'минимальные источники'):
            fit_sources('Уточнение', {**memory, 'notes': 'x' * 50000}, sources, settings)

    def test_citations_do_not_accept_unknown_or_fabricated_quotes(self):
        data, checks = validate_answer(answer(), [source()])
        self.assertTrue(checks['passed'])
        for sid, quote in [('S2', 'Пустой массив возвращает undefined.'), ('S1', 'Не существует.'), ('S1', '')]:
            invalid = copy.deepcopy(data)
            invalid['citations'] = [{'source_id': sid, 'quote': quote}]
            self.assertFalse(validate_answer(json.dumps(invalid), [source()])[1]['passed'])
        for raw in ['[]', '{', '{"status":"unknown","answer":"","gaps":"","citations":[]}']:
            self.assertFalse(validate_answer(raw, [source()])[1]['passed'])

    def test_memory_retains_questions_and_notes_without_partial_raw_or_thinking(self):
        session = new_session()
        session['notes'] = 'Используем Portal'
        session['turns'] = [{'question': str(i), 'results': {'local': {'raw': 'HALLUCINATION'}}} for i in range(8)]
        memory = memory_snapshot(session)
        self.assertEqual(memory['user_questions'], ['2', '3', '4', '5', '6', '7'])
        self.assertEqual(memory['excluded_questions'], 2)
        self.assertNotIn('HALLUCINATION', json.dumps(memory))
        self.assertEqual(memory['notes'], 'Используем Portal')

    def test_memory_includes_refusals_invalid_answers_and_latest_repeat_per_provider(self):
        session = new_session()
        session['turns'] = [{'question': 'Удали дубли', 'results': {
            'local': {'model': 'qwen', 'status': 'complete', 'response':
                      {'answer': '', 'gaps': 'Не нашёл функцию'}},
            'cloud': {'status': 'disabled'}}},
            {'question': 'Удали дубли', 'repeat_of': 1, 'results': {
                'local': {'status': 'error'},
                'cloud': {'model': 'deepseek', 'status': 'invalid', 'raw': json.dumps(
                    {'answer': 'Используй Distinct [S1]', 'gaps': ''})}}}]
        memory = memory_snapshot(session)
        self.assertEqual(memory['user_questions'], ['Удали дубли'])
        local = provider_memory(memory, 'local')
        cloud = provider_memory(memory, 'cloud')
        self.assertEqual(local['prior_answers'][0]['gaps'], 'Не нашёл функцию')
        self.assertEqual(cloud['prior_answers'][0]['status'], 'invalid')
        self.assertNotIn('Distinct', json.dumps(local))
        self.assertNotIn('Не нашёл функцию', json.dumps(cloud, ensure_ascii=False))
        self.assertIn('Не нашёл функцию', memory_text(local))
        self.assertIn('Distinct', memory_text(cloud))
        self.assertIn('Проверка не пройдена', memory_text(cloud))
        session['turns'].append({'question': 'Удали дубли', 'repeat_of': 2, 'results': {
            'local': {'model': 'qwen', 'status': 'complete', 'response':
                      {'answer': 'Новый ответ', 'gaps': ''}}, 'cloud': {'status': 'disabled'}}})
        memory = memory_snapshot(session)
        self.assertEqual(memory['user_questions'], ['Удали дубли'])
        self.assertEqual(provider_memory(memory, 'local')['prior_answers'][0]['answer'], 'Новый ответ')
        self.assertEqual(provider_memory(memory, 'cloud')['prior_answers'], cloud['prior_answers'])

    def test_context_budget_rejects_excess_and_retains_sources(self):
        memory = {'notes': '', 'user_questions': ['x'*2000]*6, 'excluded_questions': 0,
                  'prior_answers': [{'answer': str(i) * 2000} for i in range(6)]}
        messages, budget = prepare_messages('Вопрос', memory, [source()], Settings(num_ctx=4096, num_predict=1024))
        self.assertLessEqual(budget['input_bytes'], budget['byte_budget'])
        self.assertGreater(budget['memory']['excluded_questions'], 0)
        self.assertIn(source()['text'], messages[-1]['content'])
        with self.assertRaises(ValueError):
            prepare_messages('x'*20000, memory, [source()], Settings(num_ctx=4096, num_predict=1024))
        self.assertEqual(len(memory['user_questions']), 6)
        self.assertEqual(len(budget['memory']['user_questions']), len(budget['memory']['prior_answers']))
        self.assertEqual(budget['memory']['prior_answers'], memory['prior_answers'][budget['memory']['excluded_questions']:])

    def test_storage_preserves_partial_answers_and_recovers_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            session = new_session()
            session['turns'] = [{'question': 'q', 'status': 'running', 'sources': [],
                'results': {'local': {'status': 'running', 'raw': 'partial'}}}]
            path = save_session(session, directory)
            restored = load_session(path)
            self.assertEqual(restored['turns'][0]['status'], 'interrupted')
            self.assertEqual(restored['turns'][0]['results']['local']['status'], 'interrupted')
            self.assertEqual(restored['turns'][0]['results']['local']['raw'], 'partial')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(path.with_suffix('.tmp').exists())
            session['id'] = '../escape'
            with self.assertRaises(ValueError):
                save_session(session, directory)

    def test_settings_and_planner_contract(self):
        for setting in (Settings(top_k=0), Settings(temperature=float('nan')), Settings(num_ctx=4096, num_predict=4096)):
            with self.assertRaises(ValueError):
                setting.validate()
        for raw in ['{}', '[]', '{"question":"q","queries":[""]}']:
            with self.assertRaises(ValueError):
                parse_plan(raw)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_questions_keep_first_two_answers_with_large_russian_sources(self):
        sources = [{**source(), 'source_id': f'S{i}',
                    'text': source()['text'] + '\n' + 'Подробная документация. ' * 700} for i in range(1, 5)]
        response = json.loads(answer())
        response['answer'] += '\n' + 'Пример и пояснение. ' * 100
        local = FakeModel(raw=json.dumps(response, ensure_ascii=False))
        cloud_response = {**response, 'answer': response['answer'] + '\nОблачный вариант.'}
        cloud = FakeModel(raw=json.dumps(cloud_response, ensure_ascii=False))
        runner, session = Runner(FakeDocs(sources), local, cloud), new_session()
        questions = ['Как в WebTutor удалить дубли из массива?',
                     'А что делает ArraySelectDistinct? Покажи пример для массива сотрудников.',
                     'Сравни свой первый ответ с тем, что ты сейчас нашёл. Что нужно исправить?']
        for question in questions:
            turn = await runner.run(session, question, Settings(cloud=True, planner=False))
            self.assertEqual(turn['status'], 'complete')
        for name, expected in (('local', response), ('cloud', cloud_response)):
            context = turn['provider_contexts'][name]
            self.assertEqual(context['memory']['user_questions'], questions[:2])
            self.assertEqual(context['memory']['excluded_questions'], 0)
            self.assertEqual([a['answer'] for a in context['memory']['prior_answers']], [expected['answer']] * 2)
            self.assertLessEqual(context['input_bytes'], context['byte_budget'])

    async def test_budget_truncation_archives_full_text_validates_and_repeats_sent_sources(self):
        sources = [{**source(), 'text': source()['text'] + '\n' + 'Длинный материал. ' * 3000}]
        docs, local, cloud = FakeDocs(sources), FakeModel(), FakeModel()
        runner, session = Runner(docs, local, cloud), new_session()
        first = await runner.run(session, 'q', Settings(cloud=True, planner=False))
        self.assertEqual(first['status'], 'complete')
        self.assertEqual(first['sources'][0]['read_text'], sources[0]['text'])
        self.assertTrue(first['sources'][0]['budget_truncated'])
        payload = json.loads(first['messages'][-1]['content'])
        self.assertNotIn('read_text', payload['sources'][0])
        self.assertEqual(payload['sources'][0]['text'], first['sources'][0]['text'])
        second = await runner.run(session, 'q', Settings(cloud=True, planner=False), repeat=0)
        self.assertEqual(second['provider_messages'], first['provider_messages'])
        self.assertEqual(second['sources'], first['sources'])
        self.assertEqual(len(docs.calls), 1)

    async def test_each_provider_receives_own_history_planner_and_exact_repeat(self):
        docs, local = FakeDocs(), FakeModel()
        cloud = FakeModel(raw=json.dumps({'status': 'unknown', 'answer': '',
            'citations': [], 'gaps': 'Облачный отказ'}, ensure_ascii=False))
        runner, session = Runner(docs, local, cloud), new_session()
        settings = Settings(cloud=True)
        await runner.run(session, 'Первый вопрос', settings)
        second = await runner.run(session, 'Что ты имел в виду?', settings)
        local_payload = json.loads(local.calls[-1][-1]['content'])
        cloud_payload = json.loads(cloud.calls[-1][-1]['content'])
        self.assertEqual(local_payload['sources'], cloud_payload['sources'])
        self.assertEqual(local_payload['memory']['user_questions'], ['Первый вопрос'])
        self.assertIn('undefined', local_payload['memory']['prior_answers'][0]['answer'])
        self.assertEqual(cloud_payload['memory']['prior_answers'][0]['gaps'], 'Облачный отказ')
        self.assertNotIn('Облачный отказ', json.dumps(local_payload, ensure_ascii=False))
        planner = json.loads(second['planner']['messages'][-1]['content'])
        self.assertEqual(planner['memory']['prior_answers'], local_payload['memory']['prior_answers'])
        self.assertNotEqual(second['provider_contexts']['local']['sha256'], second['provider_contexts']['cloud']['sha256'])
        with tempfile.TemporaryDirectory() as directory:
            session = load_session(save_session(session, directory))
        session['notes'] = 'Новая заметка'
        repeat = await runner.run(session, 'Что ты имел в виду?', settings, repeat=1)
        self.assertEqual(repeat['provider_messages'], second['provider_messages'])
        self.assertEqual(local.calls[-1], second['provider_messages']['local'])
        self.assertEqual(cloud.calls[-1], second['provider_messages']['cloud'])
        self.assertEqual(len(docs.calls), 2)

    async def test_legacy_repeat_preserves_original_common_messages(self):
        docs, local, cloud = FakeDocs(), FakeModel(), FakeModel()
        runner, session = Runner(docs, local, cloud), new_session()
        first = await runner.run(session, 'q', Settings(planner=False))
        first.pop('provider_messages')
        first.pop('provider_contexts')
        repeat = await runner.run(session, 'q', Settings(cloud=True, planner=False), repeat=0)
        self.assertEqual(local.calls[-1], first['messages'])
        self.assertEqual(cloud.calls[-1], first['messages'])
        self.assertEqual(repeat['status'], 'complete')

    async def test_common_context_independent_errors_and_repeat_without_retrieval(self):
        docs, local, cloud = FakeDocs(), FakeModel(), FakeModel(failure=True)
        runner, session = Runner(docs, local, cloud), new_session()
        settings = Settings(cloud=True)
        first = await runner.run(session, 'Исходный вопрос', settings)
        self.assertEqual(first['results']['local']['status'], 'complete')
        self.assertEqual(first['results']['cloud']['status'], 'error')
        self.assertEqual(local.calls[-1], cloud.calls[-1])
        self.assertEqual(first['status'], 'attention')
        session['notes'] = 'Изменённая память'
        second = await runner.run(session, 'Исходный вопрос', settings, repeat=0)
        self.assertEqual(len(docs.calls), 1)
        self.assertEqual(second['messages'], first['messages'])
        self.assertEqual(second['context']['sha256'], first['context']['sha256'])
        self.assertEqual(second['repeat_of'], 1)

    async def test_no_cloud_calls_when_disabled_and_no_generation_without_sources(self):
        local, cloud, docs = FakeModel(), FakeModel(), FakeDocs()
        await Runner(docs, local, cloud).run(new_session(), 'q', Settings(planner=False))
        self.assertEqual(len(cloud.calls), 0)
        self.assertEqual(len(local.calls), 1)
        session = new_session()
        await Runner(FakeDocs([]), local, cloud).run(session, 'q', Settings(cloud=True, planner=False))
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(cloud.calls), 0)
        self.assertEqual(session['turns'][0]['status'], 'no_sources')

    async def test_cancel_preserves_partial_text_and_stops_both(self):
        local, cloud, session = FakeModel(block=True), FakeModel(block=True), new_session()
        task = asyncio.create_task(Runner(FakeDocs(), local, cloud).run(session, 'q', Settings(cloud=True, planner=False)))
        await asyncio.wait_for(local.entered.wait(), 2)
        await asyncio.wait_for(cloud.entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        turn = session['turns'][0]
        self.assertEqual(turn['status'], 'cancelled')
        for result in turn['results'].values():
            self.assertEqual(result['status'], 'cancelled')
            self.assertEqual(result['raw'], 'частичный')

    async def test_invalid_json_remains_visible_but_not_accepted(self):
        turn = await Runner(FakeDocs(), FakeModel(raw='Wrong'), FakeModel()).run(new_session(), 'q', Settings(planner=False))
        self.assertEqual(turn['results']['local']['status'], 'invalid')
        self.assertNotIn('response', turn['results']['local'])


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_cloud_contract_usage_and_safe_http_errors(self):
        requests = []
        events = [{'choices': [{'delta': {'content': 'answer'}, 'finish_reason': None}]},
                  {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 4}}]
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, text='\n'.join('data: '+json.dumps(x) for x in events)+'\ndata: [DONE]\n')
        updates = []
        metrics = await CloudClient('test', httpx.MockTransport(handler)).generate(
            [{'role':'user','content':'q'}], Settings(), lambda *x: updates.append(x))
        self.assertEqual(metrics['output_tokens'], 4)
        self.assertEqual(metrics['done_reason'], 'stop')
        self.assertEqual(updates, [('answer','')])
        self.assertEqual(requests[0]['thinking'], {'type':'disabled'})
        self.assertEqual(requests[0]['max_tokens'], 3072)
        self.assertIsNone(metrics['tokens_per_second'])
        transport = httpx.MockTransport(lambda req: httpx.Response(401, text='secret-body'))
        with self.assertRaisesRegex(ValueError, 'HTTP 401') as error:
            await CloudClient('private-key', transport).generate([], Settings(), lambda *x: None)
        self.assertNotIn('secret-body', str(error.exception))
        self.assertNotIn('private-key', str(error.exception))

    async def test_broken_cloud_stream_and_local_options(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text='data: {"choices": []}\n'))
        with self.assertRaisesRegex(ValueError, 'оборвался'):
            await CloudClient('test', transport).generate([], Settings(), lambda *x: None)
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, text=json.dumps({'done':True,'done_reason':'stop','message':{},'eval_count':10,'eval_duration':1000000000}))
        client = LocalClient(httpx.MockTransport(handler))
        await client.generate([], Settings(thinking=True), lambda *x: None, planning=True)
        self.assertFalse(requests[0]['think'])
        self.assertEqual(requests[0]['options']['num_predict'], 512)

    async def test_mcp_reads_cards_and_preserves_incompleteness_and_counterpart(self):
        class Session:
            pass
        from contextlib import asynccontextmanager
        class StubDocs(DocsClient):
            def __init__(self):
                self.calls = []
            @asynccontextmanager
            async def connect(self):
                yield Session()
            async def call(self, session, name, args):
                self.calls.append((name, copy.deepcopy(args)))
                if name == 'webtutor_docs_status':
                    return {'cache': {'state':'ready'}}
                if name == 'webtutor_search':
                    return {'results':[{'ref':'portal:1','title':'A','chunkId':'c',
                        'alsoIn':[{'ref':'datex:1','title':'A','chunkId':'d','corpus':'Datex'}]}]}
                return {'summary':{'lead':'actual documentation'}, 'body':'body', 'hasMore':True, 'nextCursor':len(self.calls)}
        docs = StubDocs()
        retrieved = await docs.retrieve(['A'], Settings(top_k=2), lambda *x:None)
        self.assertEqual([s['ref'] for s in retrieved['sources']], ['portal:1','datex:1'])
        self.assertTrue(all(s['incomplete'] for s in retrieved['sources']))
        self.assertIn('actual documentation', retrieved['sources'][0]['text'])
        self.assertEqual(sum(name=='webtutor_read' for name,args in docs.calls), 6)
        self.assertEqual(docs.calls[2][1]['chunk_id'], 'c')


class TuiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from ollama import OllamaManager
        def offline(request):
            raise httpx.ConnectError('offline', request=request)
        self.manager_patch = patch('tui.OllamaManager', side_effect=lambda: OllamaManager(
            transport=httpx.MockTransport(offline)))
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)

    async def wait_done(self, app, pilot):
        for _ in range(50):
            await pilot.pause(.02)
            if app.generation_task and app.generation_task.done():
                await app.generation_task
                return
        self.fail('TUI did not finish')

    async def test_send_repeat_notes_reopen_and_responsive_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs, local, cloud = FakeDocs(), FakeModel(), FakeModel()
            app = RagApp(directory=tmp, runner=Runner(docs, local, cloud))
            async with app.run_test(size=(180,60)) as pilot:
                app.query_one('#cloud', Switch).value = True
                app.query_one('#notes', TextArea).load_text('Нужна версия Portal')
                app.query_one('#question', TextArea).load_text('Что вернёт функция?')
                app.action_send()
                await self.wait_done(app, pilot)
                self.assertEqual(app.session['turns'][0]['status'], 'complete')
                self.assertEqual(app.query_one('#local-scroll').region.width, app.query_one('#cloud-scroll').region.width)
                app.action_send(repeat=0)
                await self.wait_done(app, pilot)
                self.assertEqual(len(docs.calls), 1)
                self.assertEqual(app.session['notes'], 'Нужна версия Portal')
                await pilot.resize_terminal(120,40)
                self.assertTrue(app.query_one('#send').region.bottom <= 40)
                self.assertTrue(app.query_one('#repeat').region.right <= 120)
            paths = list(Path(tmp).glob('*.json'))
            self.assertEqual(len(paths), 1)
            restored = load_session(paths[0])
            self.assertEqual(len(restored['turns']), 2)
            reopen = RagApp(directory=tmp, session_path=paths[0], runner=app.runner)
            async with reopen.run_test(size=(120,40)) as pilot:
                await pilot.pause()
                self.assertEqual(reopen.index, 1)
                self.assertEqual(reopen.query_one('#notes', TextArea).text, 'Нужна версия Portal')
                self.assertFalse(reopen.query_one('#repeat', Button).disabled)
                self.assertEqual(reopen.query_one('#sources', Select).value, 0)

    async def test_invalid_settings_and_missing_cloud_key_preserve_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = RagApp(directory=tmp, runner=Runner(FakeDocs(), FakeModel(), CloudClient('')))
            async with app.run_test(size=(120,40)) as pilot:
                app.query_one('#question', TextArea).load_text('Не потерять вопрос')
                app.query_one('#top_k', Input).value = '0'
                app.action_send()
                self.assertEqual(app.query_one('#question', TextArea).text, 'Не потерять вопрос')
                self.assertEqual(app.session['turns'], [])
                app.query_one('#top_k', Input).value = '4'
                app.query_one('#cloud', Switch).value = True
                app.action_send()
                self.assertEqual(app.query_one('#question', TextArea).text, 'Не потерять вопрос')
                self.assertEqual(app.session['turns'], [])

    async def test_stop_saves_partial_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = FakeModel(block=True)
            app = RagApp(directory=tmp, runner=Runner(FakeDocs(), local, FakeModel()))
            async with app.run_test(size=(120,40)) as pilot:
                app.query_one('#planner', Switch).value = False
                app.query_one('#question', TextArea).load_text('q')
                app.action_send()
                await asyncio.wait_for(local.entered.wait(), 2)
                app.action_stop()
                await self.wait_done(app, pilot)
                self.assertEqual(app.session['turns'][0]['results']['local']['raw'], 'частичный')
                restored = load_session(next(Path(tmp).glob('*.json')))
                self.assertEqual(restored['turns'][0]['status'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
