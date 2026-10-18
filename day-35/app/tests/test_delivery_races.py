import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from personal_assistent.runtime import Runtime
from personal_assistent.storage import Store
from personal_assistent.telegram import Telegram, TelegramError
from personal_assistent.tools import ToolRunner, local_time
from test_assistant import NOW, OWNER, FakeTelegram, call, update


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(':memory:')
        self.store.enqueue(update('Напомни проверить фильтр'), NOW)
        self.job = self.store.next_job()
        self.runner = ToolRunner(self.store)
        self.counter = 0
        self.reminder = self.runner.run(call('create_reminder', {
            'content': 'Фильтр', 'due_at': local_time(NOW + 60)}, 'create'), self.job, NOW)
        self.telegram = FakeTelegram()
        self.runtime = Runtime(self.store, self.telegram, None, OWNER, lambda: NOW + 61)
        self.runtime.schedule_once()

    async def asyncTearDown(self):
        self.store.db.close()

    def tool(self, name, **args):
        self.counter += 1
        return self.runner.run(call(name, args, str(self.counter)), self.job, NOW + 61)

    async def test_cancel_before_send_removes_delivery(self):
        self.assertIn('cancelled_reminder_id', self.tool('cancel_reminder', id=self.reminder['id']))
        self.assertFalse(await self.runtime.deliver_once())
        self.assertEqual(self.telegram.sent, [])

    async def test_cancel_and_reschedule_during_send_are_rejected_and_actual_delivery_recorded(self):
        started, release = asyncio.Event(), asyncio.Event()
        original = self.telegram.send

        async def send(*args):
            started.set()
            await release.wait()
            return await original(*args)

        self.telegram.send = send
        task = asyncio.create_task(self.runtime.deliver_once())
        try:
            await asyncio.wait_for(started.wait(), 1)
            for name, args in (
                    ('cancel_reminder', {}),
                    ('update_reminder', {'content': 'Новый текст', 'due_at': local_time(NOW + 600)})):
                with self.subTest(action=name):
                    result = self.tool(name, id=self.reminder['id'], **args)
                    self.assertEqual(result['error']['code'], 'delivery_in_progress')
                    self.assertIn('не выполнены', result['error']['message'])
            self.assertEqual(self.store.db.execute('SELECT status FROM reminders').fetchone()[0], 'queued')
            release.set()
            await asyncio.wait_for(task, 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(self.telegram.sent), 1)
        saved = self.store.db.execute('SELECT * FROM reminders').fetchone()
        self.assertEqual((saved['status'], saved['content']), ('sent', 'Фильтр'))
        self.assertIsNotNone(self.store.db.execute('SELECT telegram_message_id FROM outbox').fetchone()[0])

    async def test_transport_failure_keeps_unknown_outcome_even_after_later_rejection(self):
        self.telegram.error = httpx.ReadTimeout('Unknown outcome')
        await self.runtime.deliver_once()
        self.assertEqual(self.tool('cancel_reminder', id=self.reminder['id'])['error']['code'], 'delivery_in_progress')
        self.telegram.error = TelegramError(429, 60)
        self.runtime.clock = lambda: NOW + 65
        await self.runtime.deliver_once()
        self.assertEqual(self.tool('cancel_reminder', id=self.reminder['id'])['error']['code'], 'delivery_in_progress')
        self.assertEqual(self.store.db.execute('SELECT delivery_started FROM outbox').fetchone()[0], 1)
        self.telegram.error = None
        self.runtime.clock = lambda: NOW + 126
        await self.runtime.deliver_once()
        self.assertEqual(self.store.db.execute('SELECT status FROM reminders').fetchone()[0], 'sent')

    async def test_explicit_rejection_allows_cancel_before_retry(self):
        self.telegram.error = TelegramError(429, 60)
        await self.runtime.deliver_once()
        self.assertEqual(self.store.db.execute('SELECT delivery_started FROM outbox').fetchone()[0], 0)
        self.assertIn('cancelled_reminder_id', self.tool('cancel_reminder', id=self.reminder['id']))
        self.assertIsNone(self.store.next_outbox(NOW + 1000))

    async def test_invalid_telegram_response_is_unknown_not_confirmed_rejection(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content='invalid response'))) as client:
            self.runtime.telegram = Telegram(client, 'fake')
            await self.runtime.deliver_once()
        self.assertEqual(self.tool('cancel_reminder', id=self.reminder['id'])['error']['code'], 'delivery_in_progress')

    async def test_malformed_success_keeps_unknown_outcome_and_worker_can_retry(self):
        invalid = (None, {}, [], {'message_id': True}, {'message_id': 0}, {'message_id': '5'})
        for attempt, result in enumerate(invalid, 1):
            with self.subTest(result=result):
                self.runtime.clock = lambda: NOW + attempt * 1000
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda _: httpx.Response(200, json={'ok': True, 'result': result}))) as client:
                    self.runtime.telegram = Telegram(client, 'fake')
                    with self.assertLogs('personal_assistent.runtime', level='WARNING'):
                        self.assertTrue(await self.runtime.deliver_once())
                row = self.store.db.execute('SELECT * FROM outbox').fetchone()
                self.assertEqual((row['attempts'], row['delivery_started']), (attempt, 1))
                self.assertIsNone(row['sent_at'])
                self.assertEqual(self.tool('cancel_reminder', id=self.reminder['id'])['error']['code'], 'delivery_in_progress')
        self.runtime.telegram = self.telegram
        self.runtime.clock = lambda: NOW + 10000
        self.assertTrue(await self.runtime.deliver_once())
        self.assertEqual(self.store.db.execute('SELECT status FROM reminders').fetchone()[0], 'sent')


class DeliveryPersistenceTests(unittest.TestCase):
    def test_started_delivery_survives_restart_and_schema_upgrade_preserves_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.sqlite3'
            store = Store(path)
            store.enqueue(update(), NOW)
            job = store.next_job()
            reminder = ToolRunner(store).run(call('create_reminder', {
                'content': 'Фильтр', 'due_at': local_time(NOW + 60)}, 'create'), job, NOW)
            store.prepare_reminders(NOW + 61, lambda _: 'Напоминание')
            with store.db:
                store.db.execute('ALTER TABLE outbox DROP COLUMN delivery_started')
            store.db.close()
            store = Store(path)
            try:
                row = store.next_outbox(NOW + 61)
                self.assertEqual(row['text'], 'Напоминание')
                self.assertEqual(row['delivery_started'], 0)
                store.start_delivery(row['key'])
            finally:
                store.db.close()
            store = Store(path)
            try:
                result = ToolRunner(store).run(call('cancel_reminder', {'id': reminder['id']}, 'cancel'), job, NOW + 62)
                self.assertEqual(result['error']['code'], 'delivery_in_progress')
                self.assertIsNotNone(store.next_outbox(NOW + 62))
            finally:
                store.db.close()
