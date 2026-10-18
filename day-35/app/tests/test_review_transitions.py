"""Regression coverage for selection invariants, cancelled questions and dated history."""
import asyncio
import unittest
from unittest.mock import patch

from personal_assistent.agent import Agent
from personal_assistent.context import Budget, Context, ContextFull
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner, local_time
from test_assistant import NOW, Model, answer, call, update
from test_context import state


class ExecutionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.store.enqueue(update('Синтетическая проверка'), NOW)
        self.job = self.store.next_job()
        self.runner = ToolRunner(self.store)
        self.counter = 0

    def tearDown(self):
        self.store.db.close()

    def tool(self, name, args, pending=None):
        self.counter += 1
        return self.runner.run(call(name, args, str(self.counter)),
                               {**self.job, 'pending': pending}, NOW)

    def test_execution_rechecks_selected_id_and_type_for_all_target_actions(self):
        for action in ('delete_note', 'update_note', 'cancel_reminder', 'update_reminder'):
            for wrong_type in (False, True):
                with self.subTest(action=action, wrong_type=wrong_type):
                    kind = 'note' if action.endswith('note') else 'reminder'
                    parameters = {'content': 'Исходный текст'}
                    if kind == 'reminder':
                        parameters['due_at'] = local_time(NOW + 3600)
                    target = self.tool('create_' + kind, parameters)
                    changes = {'id': target['id']}
                    if action.startswith('update_'):
                        changes['content'] = 'Новый текст'
                    draft = self.tool('prepare_request', {'action': action, 'parameters': changes})
                    selected_kind = ('reminder' if kind == 'note' else 'note') if wrong_type else kind
                    # Seed the actual alternative directly: creating it through the
                    # agent would intentionally supersede the prepared request.
                    if selected_kind == 'note':
                        other = self.store.db.execute("INSERT INTO notes(content,original,created_at,updated_at,source_message_id) VALUES(?,?,?,?,?)",
                                                      ('Выбранная запись', '', NOW, NOW, 1)).lastrowid
                    else:
                        other = self.store.db.execute("INSERT INTO reminders(content,due_at,created_at,source_message_id) VALUES(?,?,?,?)",
                                                      ('Выбранная запись', NOW + 3600, NOW, 1)).lastrowid
                    self.store.db.commit()
                    selected = {'record_type': selected_kind, 'id': other, 'number': 2}
                    pending = {'selection': selected, 'request_id': draft['request_id']}
                    result = self.tool('execute_request', {'request_id': draft['request_id']}, pending)
                    self.assertEqual(result['error']['code'], 'selection_mismatch')
                    saved = self.store.db.execute('SELECT * FROM ' + ('notes' if kind == 'note' else 'reminders')
                                                  + ' WHERE id=?', (target['id'],)).fetchone()
                    self.assertEqual(saved['content'], 'Исходный текст')
                    self.assertEqual(saved['deleted'] if kind == 'note' else saved['status'],
                                     0 if kind == 'note' else 'pending')
                    self.assertEqual(self.runner.requests.active()['status'], 'ready')

    def test_rejected_execution_can_be_reprepared_for_current_selection(self):
        first = self.tool('create_note', {'content': 'Первая'})
        second = self.tool('create_note', {'content': 'Вторая'})
        draft = self.tool('prepare_request', {'action': 'delete_note', 'parameters': {'id': first['id']}})
        pending = {'selection': {'record_type': 'note', 'id': second['id']}, 'request_id': draft['request_id']}
        blocked = self.tool('execute_request', {'request_id': draft['request_id']}, pending)
        self.assertEqual(blocked['error']['code'], 'selection_mismatch')
        corrected = self.tool('prepare_request', {'action': 'delete_note', 'parameters': {'id': second['id']}}, pending)
        self.assertNotEqual(corrected['request_id'], draft['request_id'])
        self.assertEqual(self.tool('execute_request', {'request_id': corrected['request_id']}, pending)['status'], 'done')
        self.assertEqual([row[0] for row in self.store.db.execute('SELECT deleted FROM notes ORDER BY id')], [0, 1])

    def test_matching_selection_executes_and_completed_request_replays(self):
        target = self.tool('create_note', {'content': 'Первая'})
        draft = self.tool('prepare_request', {'action': 'delete_note', 'parameters': {'id': target['id']}})
        pending = {'selection': {'record_type': 'note', 'id': target['id']}}
        self.assertEqual(self.tool('execute_request', {'request_id': draft['request_id']}, pending)['status'], 'done')
        pending['selection']['id'] += 1
        self.assertEqual(self.tool('execute_request', {'request_id': draft['request_id']}, pending)['status'], 'done')


class ConflictingTransitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_selection_survives_rejection_and_next_user_message(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Удали памятку'), NOW)
            runner = ToolRunner(store)
            job = store.next_job()
            first = runner.run(call('create_note', {'content': 'Первая'}, 'first'), job, NOW)
            second = runner.run(call('create_note', {'content': 'Вторая'}, 'second'), job, NOW)
            draft = runner.run(call('prepare_request', {'action': 'delete_note',
                               'parameters': {'id': first['id']}}, 'prepare'), job, NOW)
            runner.run(call('ask_clarification', {'question': 'Какую?', 'kind': 'note', 'choices': [
                {'record_type': 'note', 'id': first['id']}, {'record_type': 'note', 'id': second['id']}]}, 'ask'), job, NOW)
            store.complete_command(job['id'], 'Выберите памятку', NOW)
            for identity, text in ((2, '2'), (3, 'Попробуй выполнить снова')):
                store.enqueue(update(text, identity), NOW)
                model = Model(answer(call('execute_request', {'request_id': draft['request_id']}, 'execute')),
                              answer(text='Не выполнено.'))
                await Agent(store, model, lambda: NOW).process(store.next_job())
                self.assertEqual(store.get_meta('pending')['selection']['id'], second['id'])
                self.assertEqual([row[0] for row in store.db.execute('SELECT deleted FROM notes ORDER BY id')], [0, 0])
        finally:
            store.db.close()

    async def test_cancelled_legacy_question_is_not_restored(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Уточни время, затем отмени поручение'), NOW)
            model = Model(answer(call('ask_clarification', {'question': 'Во сколько?', 'kind': 'reminder'}, 'ask'),
                                 call('cancel_request', {}, 'cancel')), answer(text='Поручение отменено.'))
            await Agent(store, model, lambda: NOW).process(store.next_job())
            self.assertIsNone(store.get_meta('pending'))
            self.assertNotIn('Во сколько?', store.history()[-1]['content'])
        finally:
            store.db.close()

    async def test_cancel_after_another_action_does_not_restore_old_question(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Уточни время, сохрани факт и отмени уточнение'), NOW)
            model = Model(answer(call('ask_clarification', {'question': 'Когда?', 'kind': 'other'}, 'ask'),
                                 call('create_note', {'content': 'Отдельный факт'}, 'save'),
                                 call('cancel_request', {}, 'cancel')), answer(text='Завершено.'))
            await Agent(store, model, lambda: NOW).process(store.next_job())
            self.assertIsNone(store.get_meta('pending'))
            self.assertNotIn('Когда?', store.history()[-1]['content'])
            self.assertEqual(store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
        finally:
            store.db.close()

    async def test_new_question_after_cancellation_remains_active(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Отмени первый вопрос и задай второй'), NOW)
            model = Model(answer(call('ask_clarification', {'question': 'Первый?', 'kind': 'other'}, 'first'),
                                 call('cancel_request', {}, 'cancel'),
                                 call('ask_clarification', {'question': 'Второй?', 'kind': 'other'}, 'second')))
            await Agent(store, model, lambda: NOW).process(store.next_job())
            self.assertEqual(store.get_meta('pending')['question'], 'Второй?')
            self.assertNotIn('Первый?', store.history()[-1]['content'])
            self.assertIn('Второй?', store.history()[-1]['content'])
        finally:
            store.db.close()

    async def test_cancelled_question_stays_cancelled_after_checkpoint_resume(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Уточни, затем отмени'), NOW)
            model = Model(answer(call('ask_clarification', {'question': 'Когда?', 'kind': 'other'}, 'ask'),
                                 call('cancel_request', {}, 'cancel')), answer(text='Отменено.'))
            save = store.checkpoint
            def interrupt(job_id, checkpoint):
                save(job_id, checkpoint)
                if checkpoint['messages'][-1].get('tool_call_id') == 'cancel':
                    raise asyncio.CancelledError()
            with patch.object(store, 'checkpoint', side_effect=interrupt):
                with self.assertRaises(asyncio.CancelledError):
                    await Agent(store, model, lambda: NOW).process(store.next_job())
            await Agent(store, model, lambda: NOW).process(store.next_job())
            self.assertIsNone(store.get_meta('pending'))
            self.assertNotIn('Когда?', store.history()[-1]['content'])
        finally:
            store.db.close()


class DatedHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_work_context_includes_original_times_without_changing_history(self):
        store = Store(':memory:')
        try:
            store.set_meta('profile', {'timezone': 'Asia/Yekaterinburg'})
            store.enqueue(update('Сегодня произошло событие'), NOW)
            store.bind_job(1)
            store.finish(1, 'Сегодня произошло событие', 'Понятно', NOW + 30)
            store.set_meta('profile', {'timezone': 'America/New_York'})
            raw = store.history()
            model = Model(answer(text='Не изменяю даты.'))
            store.enqueue(update('Когда это было?', 2), NOW + 86400)
            await Agent(store, model, lambda: NOW + 86400).process(store.next_job())
            history = model.inputs[0][1:3]
            self.assertIn(local_time(NOW, 'Asia/Yekaterinburg'), history[0]['content'])
            self.assertIn(local_time(NOW + 30, 'Asia/Yekaterinburg'), history[1]['content'])
            self.assertTrue(history[0]['content'].endswith(raw[0]['content']))
            self.assertEqual(store.history()[:2], raw)
            self.assertEqual([m['role'] for m in history], ['user', 'assistant'])
        finally:
            store.db.close()

    async def test_history_timestamps_are_counted_in_context_budget(self):
        store = Store(':memory:')
        try:
            store.enqueue(update('Вчера было событие'), NOW)
            store.finish(1, 'Вчера было событие', 'Принято', NOW)
            context = Context(store, None, Budget())
            current = state(store)
            raw = context.with_profile(current['messages'][:1] + store.history() + current['messages'][1:])
            context.budget = Budget(Budget().estimate(raw) + 2000, 2000)
            with self.assertRaises(ContextFull):
                await context.prepare(current)
        finally:
            store.db.close()

    async def test_legacy_unknown_time_is_not_replaced_by_current_date(self):
        store = Store(':memory:')
        try:
            with store.db:
                store.db.execute("INSERT INTO history(role,content) VALUES('user','Вчера случилось событие')")
                store.db.execute("INSERT INTO history(role,content,source_time) VALUES('assistant','Принято',?)", (NOW,))
            context = Context(store, None, Budget())
            messages = context.compose(state(store))
            self.assertIn('Время сообщения: неизвестно', messages[1]['content'])
            self.assertIn(local_time(NOW), messages[2]['content'])
            self.assertEqual(store.history()[0]['content'], 'Вчера случилось событие')
        finally:
            store.db.close()
