import asyncio
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import httpx
from textual.widgets import Input, Select, Static, Switch, TextArea, TabbedContent

from chat import OllamaClient, Settings, load_session, new_session, prepare_messages, save_session
from tui import ChatApp


class CoreTests(unittest.TestCase):
    def test_history_budget_preserves_current_question_and_complete_pairs(self):
        session = new_session()
        for i in range(8):
            session['turns'].append({'user': str(i) * 150, 'content': 'a' * 150, 'status': 'complete'})
        session['turns'].append({'user': 'FAILED', 'content': 'partial', 'status': 'cancelled'})
        messages, context = prepare_messages(session, 'current', Settings(num_ctx=2048, num_predict=512))
        self.assertEqual(messages[-1], {'role': 'user', 'content': 'current'})
        self.assertGreater(context['excluded'], 0)
        self.assertEqual(context['included'] + context['excluded'], 8)
        self.assertLessEqual(context['estimated_bytes'], context['budget'])
        self.assertNotIn('FAILED', str(messages))
        self.assertEqual([m['role'] for m in messages[1:-1]], ['user', 'assistant'] * context['included'])
        with self.assertRaises(ValueError):
            prepare_messages(session, 'a' * 5000, Settings(num_ctx=2048, num_predict=512))

    def test_storage_recovers_interrupted_turn_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = new_session()
            session['settings']['thinking'] = True
            session['turns'].append({'user': 'Привет', 'content': 'часть', 'thinking': '', 'status': 'running', 'metrics': {}})
            path = save_session(session, tmp)
            restored = load_session(path)
            self.assertTrue(restored['settings']['thinking'])
            self.assertEqual(restored['turns'][0]['status'], 'interrupted')
            self.assertEqual(restored['turns'][0]['content'], 'часть')
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_invalid_settings(self):
        for settings in (Settings(temperature=float('nan')), Settings(num_ctx=100),
                         Settings(num_predict=5000), Settings(model=''), Settings(thinking='false')):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_thinking_metrics_and_request(self):
        captured = []
        events = [ {'message': {'thinking': 'reason'}, 'done': False},
                   {'message': {'content': 'answer'}, 'done': False},
                   {'message': {}, 'done': True, 'eval_count': 20, 'eval_duration': 500000000,
                    'prompt_eval_count': 10, 'load_duration': 100000000, 'done_reason': 'stop'}]
        def handler(request):
            captured.append(json.loads(request.content))
            self.assertEqual(str(request.url), 'http://127.0.0.1:11434/api/chat')
            return httpx.Response(200, text='\n'.join(json.dumps(e) for e in events))
        updates = []
        result = await OllamaClient(transport=httpx.MockTransport(handler)).generate(
            [{'role': 'user', 'content': 'test'}], Settings(thinking=True), lambda *args: updates.append(args))
        self.assertEqual(updates, [('', 'reason'), ('answer', '')])
        self.assertEqual(result['tokens_per_second'], 40)
        self.assertEqual(result['input_tokens'], 10)
        self.assertLessEqual(result['first_fragment_seconds'], result['first_content_seconds'])
        self.assertTrue(captured[0]['think'])
        self.assertTrue(captured[0]['stream'])
        self.assertEqual(captured[0]['options']['num_ctx'], 8192)

    async def test_missing_model_broken_stream_and_connection(self):
        for status, body, expected in [(404, '{}', 'Модель не найдена'),
                (200, '{"message":{"content":"partial"},"done":false}', 'оборвался'),
                (200, '{"error":"unsupported thinking"}', 'unsupported thinking'),
                (200, 'not json', 'Ошибка потока')]:
            with self.subTest(status=status, body=body):
                client = OllamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(status, text=body)))
                with self.assertRaisesRegex(ValueError, expected):
                    await client.generate([], Settings(), lambda *args: None)
        def broken(request):
            raise httpx.ConnectError('offline')
        with self.assertRaisesRegex(ValueError, 'Нет соединения'):
            await OllamaClient(transport=httpx.MockTransport(broken)).generate([], Settings(), lambda *args: None)


class FakeClient:
    def __init__(self, wait=False):
        self.messages = []
        self.wait = wait
        self.started = asyncio.Event()

    async def generate(self, messages, settings, update):
        self.messages.append(messages)
        update('Ответ', 'Думаю' if settings.thinking else '')
        self.started.set()
        if self.wait:
            await asyncio.Event().wait()
        return {'input_tokens': 15, 'output_tokens': 5, 'tokens_per_second': 50.0,
                'wall_seconds': .1, 'first_fragment_seconds': .01, 'first_content_seconds': .02,
                'load_seconds': 0.0, 'done_reason': 'stop'}


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_screen_history_settings_save_and_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            app = ChatApp(directory=tmp, client=client)
            async with app.run_test(size=(180, 60)) as pilot:
                self.assertFalse(app.query(TabbedContent))
                self.assertEqual(app.query_one("#sidebar").max_scroll_y, 0)
                app.query_one('#thinking', Switch).value = True
                app.query_one('#temperature', Input).value = '0.3'
                for question in ('Меня зовут Анна', 'Как меня зовут?'):
                    app.query_one('#input', TextArea).load_text(question)
                    await pilot.click('#send')
                    await app.generation_task
                    await pilot.pause()
                self.assertEqual(len(client.messages[-1]), 4)
                self.assertEqual(client.messages[-1][1]['content'], 'Меня зовут Анна')
                self.assertIn('50.00 ток./с', str(app.query_one('#metrics', Static).content))
                self.assertIn('Вход: 30 ток.', str(app.query_one('#totals', Static).content))
                path = next(Path(tmp).glob('*.json'))
                self.assertEqual(load_session(path)['settings']['temperature'], .3)
                await pilot.click('#new')
                self.assertEqual(len(app.session['turns']), 0)
                app.query_one('#sessions', Select).value = str(path)
                await pilot.click('#open')
                self.assertEqual(len(app.session['turns']), 2)
                self.assertTrue(app.query_one('#thinking', Switch).value)
                self.assertTrue(app.query_one('#send').region.bottom <= 60)

    async def test_cancel_preserves_partial_and_excludes_it_from_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(wait=True)
            app = ChatApp(directory=tmp, client=client)
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one('#input', TextArea).load_text('test')
                app.action_send()
                await client.started.wait()
                app.action_stop()
                await app.generation_task
                await pilot.pause()
                self.assertEqual(app.session['turns'][0]['status'], 'cancelled')
                self.assertEqual(app.session['turns'][0]['content'], 'Ответ')
                self.assertFalse(app.query_one('#send').disabled)
                messages, context = prepare_messages(app.session, 'next', Settings())
                self.assertEqual(context['included'], 0)
                self.assertEqual(len(messages), 2)
                self.assertEqual(load_session(next(Path(tmp).glob('*.json')))['turns'][0]['status'], 'cancelled')

    async def test_settings_error_does_not_clear_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = ChatApp(directory=tmp, client=FakeClient())
            async with app.run_test(size=(180, 60)) as pilot:
                app.query_one('#num_predict', Input).value = '999999'
                app.query_one('#input', TextArea).load_text('Сохрани этот вопрос')
                app.action_send()
                await pilot.pause()
                self.assertIsNone(app.generation_task)
                self.assertEqual(app.query_one('#input', TextArea).text, 'Сохрани этот вопрос')
                self.assertIn('Проверьте настройки', str(app.query_one('#status', Static).content))


if __name__ == '__main__':
    unittest.main()
