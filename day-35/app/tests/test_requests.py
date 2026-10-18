import asyncio
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from personal_assistent.agent import Agent
from personal_assistent.runtime import Runtime
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner, local_time
from test_assistant import Model, NOW, OWNER, update, answer, call


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.store.enqueue(update('Напомни завтра купить молоко'), NOW)
        self.job = self.store.next_job()
        self.runner = ToolRunner(self.store)
        self.counter = 0

    def tearDown(self):
        self.store.db.close()

    def tool(self, name, arguments=None, now=NOW, job=None):
        self.counter += 1
        return self.runner.run(call(name, arguments or {}, str(self.counter)), job or self.job, now)

    def prepare(self, action='create_reminder', values=None, continued=False, **kwargs):
        return self.tool('prepare_request', {'action': action, 'parameters': values or {},
                         'continue_request': continued}, **kwargs)

    def ready_reminder(self):
        return self.prepare(values={'content': 'Купить молоко', 'due_date': '2026-10-17', 'due_time': '09:00'})

    def test_missing_reminder_text_asks_what_to_remind(self):
        draft = self.prepare(values={'due_date': '2026-10-17', 'due_time': '09:00'})
        self.assertEqual(draft['question'], 'О чём напомнить?')
        self.assertEqual(draft['pending']['question'], 'О чём напомнить?')
        self.assertIn('О чём напомнить?', self.runner.requests.describe())
        blocked = self.tool('execute_request', {'request_id': draft['request_id']})
        self.assertEqual(blocked['error']['message'], 'О чём напомнить?')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    def test_missing_note_text_keeps_note_question(self):
        draft = self.prepare('create_note')
        self.assertEqual(draft['question'], 'Какой текст нужно сохранить?')
        self.assertEqual(draft['pending']['question'], 'Какой текст нужно сохранить?')

    def test_date_survives_time_only_followup_and_execution(self):
        draft = self.prepare(values={'content': 'Купить молоко', 'due_date': '2026-10-17'})
        self.assertEqual(draft['missing_fields'], [])
        self.assertEqual(draft['status'], 'ready')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
        ready = self.prepare(None, {'due_time': '09:00'}, True)
        self.assertEqual(ready['request_id'], draft['request_id'])
        self.assertEqual(ready['parameters']['content'], 'Купить молоко')
        self.assertEqual(ready['parameters']['due_at'], '2026-10-17T09:00:00+03:00')
        result = self.tool('execute_request', {'request_id': draft['request_id']})
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['result']['source_message_id'], 1)

    def test_time_correction_changes_only_supplied_field(self):
        ready = self.ready_reminder()
        corrected = self.prepare(None, {'due_time': '10:00'}, True)
        self.assertEqual(corrected['request_id'], ready['request_id'])
        self.assertEqual(corrected['parameters']['due_at'], '2026-10-17T10:00:00+03:00')
        self.assertEqual(corrected['parameters']['content'], 'Купить молоко')

    def test_incomplete_request_cannot_execute(self):
        draft = self.prepare(values={'content': 'Молоко'})
        result = self.tool('execute_request', {'request_id': draft['request_id']})
        self.assertEqual(result['error']['code'], 'request_incomplete')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
        self.assertEqual(self.runner.requests.active()['status'], 'waiting')

    def test_repeated_execution_with_different_call_ids_is_idempotent(self):
        ready = self.ready_reminder()
        first = self.tool('execute_request', {'request_id': ready['request_id']})
        second = self.tool('execute_request', {'request_id': ready['request_id']})
        self.assertEqual(first, second)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 1)

    def test_expired_ready_creation_preserves_original_due_for_delivery(self):
        ready = self.prepare(values={'content': 'Чай', 'due_at': local_time(NOW + 60)})
        result = self.tool('execute_request', {'request_id': ready['request_id']}, now=NOW + 120)
        self.assertEqual(result['status'],'done')
        row=self.store.db.execute('SELECT * FROM reminders').fetchone()
        self.assertEqual(row['due_at'],NOW+60)
        self.store.prepare_reminders(NOW+120,lambda r:r['content'])
        self.assertIsNotNone(self.store.next_outbox(NOW+120))

    def test_cancellation_does_not_cancel_an_existing_reminder(self):
        existing = self.tool('create_reminder', {'content': 'Ранее созданное', 'due_at': local_time(NOW + 500)})
        ready = self.ready_reminder()
        self.assertEqual(self.tool('cancel_request')['status'], 'cancelled')
        self.assertIsNone(self.store.get_meta('pending'))
        result = self.tool('execute_request', {'request_id': ready['request_id']})
        self.assertEqual(result['error']['code'], 'request_closed')
        row = self.store.db.execute('SELECT * FROM reminders WHERE id=?', (existing['id'],)).fetchone()
        self.assertEqual(row['status'], 'pending')

    def test_new_request_supersedes_old_without_reusing_fields(self):
        previous = self.ready_reminder()
        current = self.prepare('create_note', {'content': 'Купил чай'})
        self.assertEqual(current['status'], 'ready')
        self.assertNotIn('due_at', current['parameters'])
        self.assertEqual(self.store.db.execute('SELECT status FROM requests WHERE id=?', (previous['request_id'],)).fetchone()[0], 'cancelled')
        self.assertEqual(self.tool('execute_request', {'request_id': previous['request_id']})['error']['code'], 'request_closed')

    def test_invalid_patch_does_not_destroy_previous_draft(self):
        draft = self.ready_reminder()
        result = self.prepare(None, {'event_date': '2026-10-17'}, True)
        self.assertEqual(result['error']['code'], 'invalid_request_fields')
        self.assertEqual(self.runner.requests.view(self.runner.requests.active())['parameters'], draft['parameters'])

    def test_aware_time_and_conflicting_schedule_rejected(self):
        result = self.prepare(values={'due_time': '09:00+05:00'})
        self.assertEqual(result['error']['code'], 'invalid_arguments')
        result = self.prepare(values={'due_at': local_time(NOW + 60), 'due_date': '2026-10-17'})
        self.assertEqual(result['error']['code'], 'conflicting_schedule')
        self.assertIsNone(self.runner.requests.active())

    def test_changing_only_existing_reminder_time_preserves_its_date(self):
        reminder = self.tool('create_reminder', {'content': 'Молоко', 'due_at': '2026-10-17T09:00:00+03:00'})
        ready = self.prepare('update_reminder', {'id': reminder['id'], 'due_time': '10:00'})
        self.assertEqual(ready['status'], 'ready')
        self.assertEqual(ready['parameters']['due_at'], '2026-10-17T10:00:00+03:00')
        self.assertEqual(ready['parameters']['content'], 'Молоко')

    def test_numbered_choice_cannot_reset_draft_and_target_another_record(self):
        first = self.tool('create_note', {'content': 'Первый'})
        second = self.tool('create_note', {'content': 'Второй'})
        draft = self.prepare('delete_note')
        job = {**self.job, 'payload': json.dumps(update('2')),
               'pending': {'selection': {'record_type': 'note', 'id': second['id']}}}
        wrong = self.prepare('delete_note', {'id': first['id']}, False, job=job)
        self.assertEqual(wrong['error']['code'], 'selection_mismatch')
        correct = self.prepare('delete_note', {}, False, job=job)
        self.assertEqual(correct['request_id'], draft['request_id'])
        self.assertEqual(correct['parameters']['id'], second['id'])
        self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=?', (first['id'],)).fetchone()[0], 0)

    def test_update_requires_changes_not_just_a_target(self):
        note = self.tool('create_note', {'content': 'Исходный текст', 'event_date': '2026-10-15'})
        draft = self.prepare('update_note', {'id': note['id']})
        self.assertEqual(draft['missing_fields'], ['changes'])
        updated = self.prepare(None, {'event_date': '2026-10-14'}, True)
        self.assertEqual(updated['parameters']['content'], 'Исходный текст')
        result = self.tool('execute_request', {'request_id': draft['request_id']})
        self.assertEqual(result['result']['event_date'], '2026-10-14')

    def test_stale_target_is_not_overwritten(self):
        note = self.tool('create_note', {'content': 'Исходный текст'})
        draft = self.prepare('update_note', {'id': note['id'], 'content': 'Новая версия'})
        with self.store.db:
            self.store.db.execute('UPDATE notes SET content=? WHERE id=?', ('Параллельное изменение', note['id']))
        result = self.tool('execute_request', {'request_id': draft['request_id']})
        self.assertEqual(result['error']['code'], 'request_stale')
        self.assertIsNone(self.runner.requests.active())
        self.assertEqual(self.store.db.execute('SELECT content FROM notes').fetchone()[0], 'Параллельное изменение')
        self.assertEqual(self.store.db.execute('SELECT status FROM requests WHERE id=?', (draft['request_id'],)).fetchone()[0], 'failed')

    def test_execution_rolls_back_both_action_and_state(self):
        draft = self.ready_reminder()
        original = self.runner.requests.apply
        def crash(*args):
            original(*args)
            raise RuntimeError('crash after mutation')
        with patch.object(self.runner.requests, 'apply', side_effect=crash):
            with self.assertRaises(RuntimeError):
                self.tool('execute_request', {'request_id': draft['request_id']})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
        self.assertEqual(self.runner.requests.active()['status'], 'ready')
        self.assertEqual(self.tool('execute_request', {'request_id': draft['request_id']})['status'], 'done')

    def test_tool_cache_failure_rolls_back_action_and_request_transition(self):
        draft = self.ready_reminder()
        command = call('execute_request', {'request_id': draft['request_id']}, 'execution')
        self.store.db.execute("CREATE TRIGGER fail_cache BEFORE INSERT ON tool_results WHEN NEW.key='1:execution' BEGIN SELECT RAISE(ABORT, 'simulated'); END")
        self.store.db.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.runner.run(command, self.job, NOW)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
        self.assertEqual(self.runner.requests.active()['status'], 'ready')
        self.store.db.execute('DROP TRIGGER fail_cache')
        self.store.db.commit()
        self.assertEqual(self.runner.run(command, self.job, NOW)['status'], 'done')
        self.assertEqual(self.runner.run(command, self.job, NOW)['status'], 'done')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 1)

    def test_request_action_cannot_be_changed_during_followup(self):
        draft = self.ready_reminder()
        result = self.prepare('delete_note', {'id': 1}, True)
        self.assertEqual(result['error']['code'], 'request_action_mismatch')
        self.assertEqual(self.runner.requests.active()['id'], draft['request_id'])
        self.assertEqual(self.runner.requests.active()['action'], 'create_reminder')

    def test_new_session_cancels_draft_without_executing_it(self):
        draft = self.ready_reminder()
        self.store.enqueue(update('/new', 2), NOW)
        self.store.new_command(2, '/new', OWNER, NOW)
        self.assertIsNone(self.runner.requests.active())
        self.assertEqual(self.tool('execute_request', {'request_id': draft['request_id']})['error']['code'], 'request_not_found')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    def test_selection_is_applied_and_wrong_target_rejected(self):
        note1 = self.tool('create_note', {'content': 'Первый'})
        note2 = self.tool('create_note', {'content': 'Второй'})
        draft = self.prepare('delete_note')
        job = {**self.job, 'pending': {'selection': {'record_type': 'note', 'id': note2['id']}}}
        wrong = self.prepare(None, {'id': note1['id']}, True, job=job)
        self.assertEqual(wrong['error']['code'], 'selection_mismatch')
        ready = self.prepare(None, {}, True, job=job)
        self.assertEqual(ready['parameters']['id'], note2['id'])
        self.tool('execute_request', {'request_id': draft['request_id']})
        self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=?', (note1['id'],)).fetchone()[0], 0)
        self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=?', (note2['id'],)).fetchone()[0], 1)


class RequestFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_draft_survives_restart_then_time_only_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'draft.db'
            store = Store(path)
            try:
                store.enqueue(update('Напомни завтра купить молоко'), NOW)
                model = Model(answer(call('prepare_request', {'action': 'create_reminder',
                    'parameters': {'content': 'Купить молоко'}})))
                await Agent(store, model, lambda: NOW).process(store.next_job())
                identity = ToolRunner(store).requests.active()['id']
                self.assertEqual(store.history()[-1]['content'], 'На какую дату поставить напоминание? Во сколько напомнить?')
            finally:
                store.db.close()
            store = Store(path)
            try:
                store.enqueue(update('В девять утра', 2), NOW + 3600)
                model = Model(answer(call('prepare_request', {'parameters': {'due_date':'2026-10-17','due_time': '09:00'}, 'continue_request': True})),
                    answer(call('execute_request', {'request_id': identity}, 'execute')), answer(text='Напомню завтра в 09:00.'))
                await Agent(store, model, lambda: NOW + 3600).process(store.next_job())
                row = store.db.execute('SELECT * FROM reminders').fetchone()
                self.assertEqual(local_time(row['due_at']), '2026-10-17T09:00:00+03:00')
                self.assertEqual(row['source_message_id'], 1)
                self.assertEqual(store.db.execute('SELECT count(*) FROM requests').fetchone()[0], 1)
            finally:
                store.db.close()

    async def test_view_and_cancel_are_local_and_preserve_existing_data(self):
        store = Store(':memory:')
        try:
            store.enqueue(update(), NOW)
            runner = ToolRunner(store)
            runner.run(call('prepare_request', {'action': 'create_reminder',
                'parameters': {'content': 'Молоко', 'due_date': '2026-10-17'}}), store.next_job(), NOW)
            store.complete_command(1, 'Во сколько?', NOW)
            for number, text in ((2, '/request'), (3, 'Отмена')):
                store.enqueue(update(text, number), NOW)
                await Runtime(store, None, None, OWNER, lambda: NOW).process_once()
            self.assertIsNone(runner.requests.active())
            self.assertEqual(store.db.execute('SELECT status FROM requests').fetchone()[0], 'cancelled')
            self.assertEqual(store.history(), [])
        finally:
            store.db.close()

    async def test_cancel_clears_legacy_clarification_without_touching_records(self):
        store = Store(':memory:')
        try:
            with store.db:
                store.set_meta('pending', {'kind': 'reminder', 'question': 'Во сколько?', 'origin': update()['message']})
            store.enqueue(update('Отмена'), NOW)
            await Runtime(store, None, None, OWNER, lambda: NOW).process_once()
            self.assertIsNone(store.get_meta('pending'))
            self.assertEqual(store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
            self.assertEqual(store.db.execute('SELECT count(*) FROM requests').fetchone()[0], 0)
        finally:
            store.db.close()

    async def test_request_and_cancel_do_not_change_new_session(self):
        store = Store(':memory:')
        try:
            for number, text in enumerate(('/new', '/request', '/cancel', '/new confirm'), 1):
                store.enqueue(update(text, number), NOW)
                await Runtime(store, None, None, OWNER, lambda: NOW).process_once()
                self.assertEqual(store.get_meta('session'), 1)
            self.assertEqual(store.get_meta('session'), 1)
            self.assertIsNone(store.get_meta('new_confirmation'))
        finally:
            store.db.close()

    async def test_read_only_question_preserves_draft_and_choices(self):
        store = Store(':memory:')
        try:
            store.enqueue(update(), NOW)
            runner = ToolRunner(store)
            draft = runner.run(call('prepare_request', {'action': 'create_reminder',
                'parameters': {'content': 'Молоко'}}), store.next_job(), NOW)
            store.complete_command(1, draft['question'], NOW)
            pending = store.get_meta('pending')
            store.enqueue(update('Что ты сейчас уточняешь?', 2), NOW)
            model = Model(answer(call('get_request', {})), answer(text='Уточняю время напоминания про молоко.'))
            await Agent(store, model, lambda: NOW).process(store.next_job())
            self.assertEqual(store.get_meta('pending'), pending)
            self.assertEqual(runner.requests.active()['status'], 'waiting')
        finally:
            store.db.close()

    async def test_cancel_in_same_response_does_not_restore_cancelled_question(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Напомни завтра про молоко. Нет, отмена.'), NOW)
            model = Model(answer(call('prepare_request', {'action': 'create_reminder',
                'parameters': {'content': 'Молоко', 'due_date': '2026-10-17'}}, 'prepare'),
                call('cancel_request', {}, 'cancel')), answer(text='Поручение отменено.'))
            await Agent(store, model, lambda: NOW).process(store.next_job())
            self.assertIsNone(store.get_meta('pending'))
            self.assertIsNone(ToolRunner(store).requests.active())
            self.assertEqual(store.history()[-1]['content'], 'Поручение отменено.')
        finally:
            store.db.close()

    async def test_restart_after_ready_checkpoint_uses_same_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ready.db'
            store = Store(path)
            try:
                store.enqueue(update('Сохрани факт'), NOW)
                model = Model(answer(call('prepare_request', {'action': 'create_note',
                    'parameters': {'content': 'Факт'}})), asyncio.CancelledError())
                with self.assertRaises(asyncio.CancelledError):
                    await Agent(store, model, lambda: NOW).process(store.next_job())
                self.assertEqual(ToolRunner(store).requests.active()['status'], 'ready')
            finally:
                store.db.close()
            store = Store(path)
            try:
                model = Model(answer(call('execute_request', {'request_id': 1}, 'execute')), answer(text='Сохранено.'))
                await Agent(store, model, lambda: NOW).process(store.next_job())
                self.assertEqual(store.db.execute('SELECT count(*) FROM requests').fetchone()[0], 1)
                self.assertEqual(store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
            finally:
                store.db.close()

    async def test_schema_upgrade_keeps_legacy_data_and_clarification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'upgrade.db'
            store = Store(path)
            try:
                store.enqueue(update('Запомни факт'), NOW)
                ToolRunner(store).run(call('create_note', {'content': 'Сохранённый факт'}), store.next_job(), NOW)
                with store.db:
                    store.set_meta('pending', {'kind': 'reminder', 'question': 'Когда?', 'origin': update()['message']})
                    store.db.execute('DROP TABLE requests')
            finally:
                store.db.close()
            store = Store(path)
            try:
                self.assertEqual(store.db.execute('SELECT content FROM notes').fetchone()[0], 'Сохранённый факт')
                self.assertEqual(store.get_meta('pending')['question'], 'Когда?')
                self.assertIsNone(ToolRunner(store).requests.active())
                self.assertIn('provided', {row[1] for row in store.db.execute('PRAGMA table_info(requests)')})
            finally:
                store.db.close()

    async def test_resume_after_execution_checkpoint_does_not_repeat_action(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Сохрани факт'), NOW)
            model = Model(answer(call('prepare_request', {'action': 'create_note', 'parameters': {'content': 'Факт'}})),
                answer(call('execute_request', {'request_id': 1}, 'execute')), asyncio.CancelledError())
            with self.assertRaises(asyncio.CancelledError):
                await Agent(store, model, lambda: NOW).process(store.next_job())
            self.assertEqual(store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
            await Agent(store, Model(answer(text='Сохранено.')), lambda: NOW).process(store.next_job())
            self.assertEqual(store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
            self.assertEqual(store.db.execute('SELECT status FROM requests').fetchone()[0], 'done')
        finally:
            store.db.close()
