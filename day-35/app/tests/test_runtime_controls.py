import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_assistent.agent import Agent, SYSTEM
from personal_assistent.config import Config
from personal_assistent.requests import Requests
from personal_assistent.runtime import Runtime
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner
from test_assistant import NOW, OWNER, FakeTelegram, Model, answer, call, update


class WaitingModel:
    def __init__(self, first=None):
        self.first = first
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        if self.first is not None:
            first, self.first = self.first, None
            return first
        self.waiting.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return answer(text=json.dumps({'kind': 'answer', 'text': 'Ответ на вопрос'}))

    async def retrieve(self, messages):
        from retrieval_stub import retrieval_reply
        return retrieval_reply(messages)


class ControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(':memory:')
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.store.db.close()

    async def start(self, first=None, timeout=5, model=None):
        self.store.enqueue(update('Выполни просьбу и ответь на вопрос'), NOW)
        self.model = model or WaitingModel(first)
        self.runtime = Runtime(self.store, FakeTelegram(), Agent(self.store, self.model, lambda: NOW),
                               OWNER, lambda: NOW, processing_timeout=timeout)
        task = asyncio.create_task(self.runtime.process_once())
        self.tasks.append(task)
        await asyncio.wait_for(self.model.waiting.wait(), 1)
        return task

    def enqueue(self, text, number, owner=OWNER):
        incoming = update(text, number)
        incoming['message']['from']['id'] = owner
        self.store.enqueue(incoming, NOW)

    def status(self, number):
        return self.store.db.execute('SELECT status FROM jobs WHERE id=?', (number,)).fetchone()[0]

    async def completed(self, number):
        async with asyncio.timeout(1):
            while self.status(number) == 'pending':
                await asyncio.sleep(.005)

    def reply(self, number):
        return ''.join(r[0] for r in self.store.db.execute(
            'SELECT text FROM outbox WHERE key LIKE ? ORDER BY key', (f'job:{number}:%',)))

    def saved_call(self, tool):
        self.store.enqueue(update('Сохрани факт'), NOW)
        self.store.bind_job(1)
        self.store.checkpoint(1, {
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': 'Сохрани факт'}, answer(tool)],
            'result_contract': 2, 'context_version': 1, 'session': 0,
            'rounds': 1, 'pending': None})

    async def test_persisted_cancel_prevents_resuming_saved_tool_call(self):
        self.saved_call(call('create_note', {'content': 'Не выполнять после отмены'}, 'save'))
        self.enqueue('/cancel', 2)
        runtime = Runtime(self.store, None, Agent(self.store, Model(), lambda: NOW), OWNER, lambda: NOW)
        await runtime.process_once()
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
        self.assertEqual([self.status(n) for n in (1, 2)], ['done', 'done'])
        self.assertIn('остановлена', self.reply(1))

    async def test_persisted_new_prevents_resuming_saved_tool_call(self):
        self.saved_call(call('create_note', {'content': 'Старая просьба'}, 'save'))
        self.enqueue('/new', 2)
        runtime = Runtime(self.store, None, Agent(self.store, Model(), lambda: NOW), OWNER, lambda: NOW)
        await runtime.process_once()
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertEqual(self.store.history(), [])

    async def test_crash_during_new_rolls_back_session_and_interrupted_job(self):
        class Crash(BaseException):
            pass

        self.saved_call(call('create_note', {'content': 'Старая просьба'}, 'save'))
        self.enqueue('/new', 2)
        self.store.set_meta('summary', {'text': 'Старая сводка', 'through': 0})
        self.store.db.commit()
        runtime = Runtime(self.store, None, Agent(self.store, Model(), lambda: NOW), OWNER, lambda: NOW)
        with patch.object(self.store, 'complete_command', side_effect=Crash):
            with self.assertRaises(Crash):
                await runtime.process_once()
        self.assertEqual(self.store.get_meta('session', 0), 0)
        self.assertEqual(self.store.get_meta('summary')['text'], 'Старая сводка')
        self.assertEqual([self.status(n) for n in (1, 2)], ['pending', 'pending'])
        self.assertEqual(self.store.history(), [])
        self.assertIsNone(self.store.next_outbox(NOW))
        await runtime.process_once()
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertEqual(self.store.get_meta('summary'), {})
        self.assertEqual([self.status(n) for n in (1, 2)], ['done', 'done'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)

    async def test_crash_during_cancel_rolls_back_both_job_completion_and_draft_cancellation(self):
        class Crash(BaseException):
            pass

        self.saved_call(call('execute_request', {'request_id': 1}, 'execute'))
        prepared = ToolRunner(self.store).run(call('prepare_request', {
            'action': 'create_note', 'parameters': {'content': 'Не выполнять'}}, 'prepare'),
            self.store.next_job(), NOW)
        self.assertEqual(prepared['request_id'], 1)
        self.enqueue('Отмена', 2)
        runtime = Runtime(self.store, None, Agent(self.store, Model(), lambda: NOW), OWNER, lambda: NOW)
        with patch.object(self.store, 'complete_command', side_effect=Crash):
            with self.assertRaises(Crash):
                await runtime.process_once()
        self.assertEqual([self.status(n) for n in (1, 2)], ['pending', 'pending'])
        self.assertEqual(Requests(self.store).active()['status'], 'ready')
        self.assertEqual(self.store.history(), [])
        self.assertIsNone(self.store.next_outbox(NOW))
        await runtime.process_once()
        self.assertIsNone(Requests(self.store).active())
        self.assertEqual([self.status(n) for n in (1, 2)], ['done', 'done'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
        self.assertEqual(len(self.store.history()), 2)

    async def test_observational_commands_work_while_model_waits_and_messages_keep_order(self):
        task = await self.start()
        self.enqueue('Следующий обычный вопрос', 2)
        for number, command in enumerate(('/stats', '/profile', '/request', '/help'), 3):
            self.enqueue(command, number)
        await self.completed(6)
        self.assertFalse(task.done())
        self.assertEqual(self.status(2), 'pending')
        self.assertEqual(self.model.calls, 1)
        self.assertIn('Статистики API пока нет', self.reply(3))
        self.assertIn('Часовой пояс', self.reply(4))
        self.assertIn('Незавершённых поручений нет', self.reply(5))
        self.assertIn('/cancel', self.reply(6))
        self.assertEqual(self.store.history(), [])
        self.model.release.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.status(2), 'pending')
        self.runtime.agent = Agent(self.store, Model(answer(text='Второй ответ')), lambda: NOW)
        await self.runtime.process_once()
        self.assertEqual(self.store.history()[2]['content'], 'Следующий обычный вопрос')

    async def test_cancel_interrupts_generation_and_cancels_ready_draft_without_execution(self):
        task = await self.start(answer(call('prepare_request', {
            'action': 'create_note', 'parameters': {'content': 'Не сохранять'}}, 'prepare')))
        self.enqueue('/cancel', 2)
        await asyncio.wait_for(task, 1)
        self.assertTrue(self.model.cancelled.is_set())
        self.assertIsNone(Requests(self.store).active())
        self.assertIsNone(self.store.get_meta('pending'))
        self.assertEqual(self.store.db.execute('SELECT status FROM requests').fetchone()[0], 'cancelled')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
        self.assertEqual([self.status(n) for n in (1, 2)], ['done', 'done'])
        self.assertIn('остановлена', self.reply(2))
        self.assertFalse(await self.runtime.process_once())

    async def test_cancel_preserves_and_confirms_committed_action(self):
        task = await self.start(answer(call('create_note', {'content': 'Сохранённый факт'}, 'save')))
        self.enqueue('Отмена', 2)
        await asyncio.wait_for(task, 1)
        self.assertIn('Памятка сохранена: "Сохранённый факт"', self.reply(1))
        self.assertIn('остановлена', self.reply(1))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
        self.assertEqual(self.status(2), 'done')

    async def test_unauthorized_cancel_cannot_interrupt(self):
        task = await self.start()
        self.enqueue('/cancel', 2, owner=999)
        self.enqueue('/stats', 3)
        await self.completed(3)
        self.assertFalse(task.done())
        self.assertFalse(self.model.cancelled.is_set())
        self.model.release.set()
        await asyncio.wait_for(task, 1)
        await self.runtime.process_once()
        self.assertEqual(self.status(2), 'ignored')

    async def test_new_stops_old_job_before_changing_session(self):
        task = await self.start()
        self.enqueue('/new', 2)
        await asyncio.wait_for(task, 1)
        self.assertTrue(self.model.cancelled.is_set())
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertEqual(self.store.history(), [])
        self.assertEqual([self.status(n) for n in (1, 2)], ['done', 'done'])

    async def test_invalid_new_arguments_do_not_stop_generation(self):
        task = await self.start()
        self.enqueue('/new confirm', 2)
        await self.completed(2)
        self.assertFalse(task.done())
        self.assertIn('Используйте /new без аргументов', self.reply(2))
        self.model.release.set()
        await asyncio.wait_for(task, 1)

    async def test_message_after_new_binds_to_new_session(self):
        task = await self.start()
        self.enqueue('/new', 2)
        self.enqueue('Следующий вопрос', 3)
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.status(3), 'pending')
        self.runtime.agent = Agent(self.store, Model(answer(text='Второй ответ')), lambda: NOW)
        await self.runtime.process_once()
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertEqual(self.store.history()[0]['content'], 'Следующий вопрос')
        self.assertEqual(self.store.db.execute('SELECT session FROM jobs WHERE id=3').fetchone()[0], 1)

    async def test_new_cannot_overtake_an_earlier_ordinary_message(self):
        task = await self.start()
        self.enqueue('Предыдущий вопрос', 2)
        self.enqueue('/new', 3)
        self.enqueue('/stats', 5)
        await self.completed(5)
        self.assertEqual([self.status(n) for n in (2, 3)], ['pending', 'pending'])
        self.model.release.set()
        await asyncio.wait_for(task, 1)
        self.runtime.agent = Agent(self.store, Model(answer(text='Ответ на предыдущий вопрос')), lambda: NOW)
        await self.runtime.process_once()
        await self.runtime.process_once()
        await self.runtime.process_once()
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertIn('Новая беседа начата', self.reply(3))

    async def test_deadline_preserves_committed_action_and_allows_next_message(self):
        task = await self.start(answer(call('create_note', {'content': 'Сохранить один раз'}, 'save')), timeout=.08)
        await asyncio.wait_for(task, 1)
        self.assertTrue(self.model.cancelled.is_set())
        self.assertIn('лимит времени', self.reply(1))
        self.assertIn('Памятка сохранена', self.reply(1))
        self.assertEqual(self.status(1), 'done')
        self.enqueue('Новый вопрос', 2)
        self.runtime.agent = Agent(self.store, Model(answer(text='Новый ответ')), lambda: NOW)
        await self.runtime.process_once()
        self.assertIn('Новый ответ', self.reply(2))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_shutdown_preserves_checkpoint_and_resumes_once(self):
        task = await self.start(answer(call('create_note', {'content': 'Один факт'}, 'save')))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.model.cancelled.is_set())
        self.assertEqual(self.status(1), 'pending')
        self.assertIsNotNone(self.store.next_job()['checkpoint'])
        self.runtime.agent = Agent(self.store, Model(answer(text='Завершено')), lambda: NOW)
        await self.runtime.process_once()
        self.assertEqual(self.status(1), 'done')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_cancel_interrupts_summary_without_updating_summary_boundary(self):
        class Summarizing(WaitingModel):
            async def complete(self, messages):
                raise AssertionError('Only summary should start')

            async def summarize(self, messages):
                return await WaitingModel.complete(self, messages)

        with self.store.db:
            for _ in range(11):
                self.store.db.execute("INSERT INTO history(role,content) VALUES('user','Вопрос')")
                self.store.db.execute("INSERT INTO history(role,content) VALUES('assistant','Ответ')")
        task = await self.start(model=Summarizing())
        self.enqueue('/cancel', 2)
        await asyncio.wait_for(task, 1)
        self.assertTrue(self.model.cancelled.is_set())
        self.assertIsNone(self.store.get_meta('summary'))
        self.assertEqual(self.status(1), 'done')


class DeadlineConfigTests(unittest.TestCase):
    def test_processing_deadline_defaults_override_and_validation(self):
        base = {'TELEGRAM_BOT_TOKEN': 'fake', 'TELEGRAM_OWNER_ID': '42', 'DEEPSEEK_API_KEY': 'fake'}
        with patch('personal_assistent.config.load_dotenv'), patch.dict(os.environ, base, clear=True):
            self.assertEqual(Config.load().processing_timeout, 120)
        for value in ('0', '-1', 'nan', 'inf', 'bad'):
            with self.subTest(value=value), patch('personal_assistent.config.load_dotenv'), patch.dict(
                    os.environ, {**base, 'ASSISTANT_PROCESSING_TIMEOUT': value}, clear=True):
                with self.assertRaisesRegex(ValueError, 'ASSISTANT_PROCESSING_TIMEOUT'):
                    Config.load()
        with patch('personal_assistent.config.load_dotenv'), patch.dict(
                os.environ, {**base, 'ASSISTANT_PROCESSING_TIMEOUT': '15.5'}, clear=True):
            self.assertEqual(Config.load().processing_timeout, 15.5)

    def test_cancelled_job_does_not_resume_after_reopening_database(self):
        async def exercise(path):
            store = Store(path)
            try:
                model = WaitingModel()
                store.enqueue(update('Вопрос'), NOW)
                runtime = Runtime(store, None, Agent(store, model, lambda: NOW), OWNER, lambda: NOW)
                task = asyncio.create_task(runtime.process_once())
                try:
                    await asyncio.wait_for(model.waiting.wait(), 1)
                    store.enqueue(update('/cancel', 2), NOW)
                    await asyncio.wait_for(task, 1)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            finally:
                store.db.close()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.sqlite3'
            asyncio.run(exercise(path))
            store = Store(path)
            try:
                self.assertIsNone(store.next_job())
                self.assertEqual(store.db.execute('SELECT count(*) FROM history').fetchone()[0], 2)
            finally:
                store.db.close()
