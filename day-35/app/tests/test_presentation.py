import json
import sqlite3
import asyncio
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

from personal_assistent.agent import Agent
from personal_assistent.presentation import public_text
from personal_assistent.runtime import Runtime, HELP
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner
from test_assistant import update, call, answer, Model, FakeTelegram, NOW, OWNER


def seed_notes(store):
    with store.db:
        for identity, content in ((2, 'Не удалять эту запись'),
                                  (101, 'Оля обещала не пить газировку'),
                                  (9001, 'Оля обещала не материться')):
            store.db.execute('INSERT INTO notes(id,content,original,created_at,updated_at,source_message_id) VALUES(?,?,?,?,?,?)',
                             (identity, content, content, NOW, NOW, 1))
            store.db.execute('INSERT INTO notes_fts(rowid,content) VALUES(?,?)', (identity, content))


def choices_call(identity='choose'):
    return call('ask_clarification', {'question': 'Какую памятку удалить?', 'kind': 'other',
                'choices': [{'record_type': 'note', 'id': 101}, {'record_type': 'note', 'id': 9001}]}, identity)


class PresentationTests(unittest.TestCase):
    def test_event_identifiers_hidden_but_quoted_user_identifiers_preserved(self):
        text='Событие №42: «Документ event_id=99». event_occurrence_id=7'
        self.assertEqual(public_text(text),'Событие: «Документ event_id=99».')

    def test_internal_markers_removed_without_losing_content(self):
        pairs = [
            ('Памятка №9001: Оля обещала не материться.', 'Памятка: Оля обещала не материться.'),
            ('Напоминание #101 создано.', 'Напоминание создано.'),
            ('Удалена памятка с ID: 9001.', 'Удалена памятка.'),
            ('Обновление ID памятки: 101.', 'Обновление памятки.'),
            ('Сохранено (ID: 101).', 'Сохранено.'),
        ]
        for original, expected in pairs:
            with self.subTest(original=original):
                self.assertEqual(public_text(original), expected)

    def test_quoted_user_content_is_not_mistaken_for_database_metadata(self):
        text = 'Сохранено: «Серийный номер устройства (ID: 101), см. памятку №42 в инструкции».'
        self.assertEqual(public_text(text), text)

    def test_existing_outbox_migrates_without_losing_queued_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.db'
            db = sqlite3.connect(path)
            db.executescript("""
                CREATE TABLE outbox(key TEXT PRIMARY KEY,text TEXT NOT NULL,reminder_id INTEGER,
                    available_at REAL NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,
                    sent_at REAL,telegram_message_id INTEGER);
                INSERT INTO outbox(key,text,reminder_id,available_at)
                    VALUES('legacy:0000:last','Old queued text',8001,100);
            """)
            db.close()
            store = Store(path)
            try:
                row = store.next_outbox(100)
                self.assertEqual(row['text'], 'Old queued text')
                self.assertEqual(row['presentation_done'], 0)
                self.assertEqual(row['reminder_id'], 8001)
                with store.db:
                    store.queue_message('new', 'New text', now=101)
                new = store.db.execute("SELECT presentation_done FROM outbox WHERE key LIKE 'new:%'").fetchone()
                self.assertEqual(new[0], 1)
            finally:
                store.db.close()

    def test_ordinary_numbers_and_temporary_numbering_preserved(self):
        text = '1. Купить 2 литра молока к 22:17 16.10.2026.\n2. Проверить заказ №9001 и внешний ID 77123.'
        self.assertEqual(public_text(text), text)
        self.assertNotIn('№1', HELP)


class PrivateIdsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(':memory:')
        self.telegram = FakeTelegram()

    async def asyncTearDown(self):
        self.store.db.close()

    async def test_numbered_selection_survives_restart_and_cannot_target_database_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bot.db'
            self.store.db.close()
            self.store = Store(path)
            seed_notes(self.store)
            self.store.enqueue(update('Удали памятку про обещание Оли'), NOW)
            await Agent(self.store, Model(answer(choices_call())), lambda: NOW).process(self.store.next_job())
            question = self.store.history()[-1]['content']
            self.assertIn('1. «Оля обещала не пить газировку»', question)
            self.assertIn('2. «Оля обещала не материться»', question)
            self.assertNotIn('9001', question)
            self.assertNotIn('101', question)
            self.assertEqual(self.store.get_meta('pending')['choices'][1]['id'], 9001)
            self.store.db.close()
            self.store = Store(path)
            self.store.enqueue(update('2', 2), NOW + 60)
            model = Model(answer(call('delete_note', {'id': 2}, 'wrong')),
                          answer(call('delete_note', {'id': 9001}, 'right')),
                          answer(text='Памятка №9001 удалена.'))
            await Agent(self.store, model, lambda: NOW + 60).process(self.store.next_job())
            error = json.loads(model.inputs[1][-1]['content'])['error']
            self.assertEqual(error['code'], 'selection_mismatch')
            self.assertEqual(error['details']['id'], 9001)
            self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=2').fetchone()[0], 0)
            self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=101').fetchone()[0], 0)
            self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=9001').fetchone()[0], 1)
            self.assertIn('Памятка удалена: "Оля обещала не материться".', self.store.history()[-1]['content'])
            self.assertIn('Выбранная запись не соответствует текущему уточнению.', self.store.history()[-1]['content'])
            self.assertIsNone(self.store.get_meta('pending'))

    async def test_invalid_selection_preserves_choices_without_model_call(self):
        seed_notes(self.store)
        self.store.enqueue(update('Удали памятку про Олю'), NOW)
        await Agent(self.store, Model(answer(choices_call())), lambda: NOW).process(self.store.next_job())
        pending = self.store.get_meta('pending')
        self.store.enqueue(update('3', 2), NOW)
        model = Model()
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        self.assertEqual(model.inputs, [])
        self.assertEqual(self.store.get_meta('pending'), pending)
        self.assertEqual(self.store.db.execute('SELECT sum(deleted) FROM notes').fetchone()[0], 0)

    async def test_missing_selected_record_does_not_fall_back_to_other_choice(self):
        seed_notes(self.store)
        self.store.enqueue(update('Удали памятку про Олю'), NOW)
        await Agent(self.store, Model(answer(choices_call())), lambda: NOW).process(self.store.next_job())
        with self.store.db:
            self.store.db.execute('UPDATE notes SET deleted=1 WHERE id=9001')
        self.store.enqueue(update('вариант 2.', 2), NOW)
        model = Model(answer(call('delete_note', {'id': 9001})),
                      answer(text='Выбранная памятка уже удалена. Другие записи не менял.'))
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        self.assertEqual(json.loads(model.inputs[1][-1]['content'])['error']['code'], 'note_not_found')
        self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=101').fetchone()[0], 0)
        self.assertNotIn('9001', self.store.history()[-1]['content'])

    async def test_selection_guards_record_type_and_prevents_new_creation(self):
        seed_notes(self.store)
        self.store.enqueue(update('2'), NOW)
        job = {**self.store.next_job(), 'pending': {'selection': {'record_type': 'reminder', 'id': 101}}}
        runner = ToolRunner(self.store)
        wrong_type = runner.run(call('delete_note', {'id': 101}), job, NOW)
        self.assertEqual(wrong_type['error']['code'], 'selection_mismatch')
        creation = runner.run(call('create_note', {'content': 'Новая запись'}, 'create'), job, NOW)
        self.assertEqual(creation['error']['code'], 'selection_mismatch')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 3)
        self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=101').fetchone()[0], 0)

    async def test_duplicate_missing_and_single_choices_are_rejected(self):
        seed_notes(self.store)
        self.store.enqueue(update(), NOW)
        job = self.store.next_job()
        runner = ToolRunner(self.store)
        for index, candidates in enumerate((
                [{'record_type': 'note', 'id': 101}],
                [{'record_type': 'note', 'id': 101}] * 2,
                [{'record_type': 'note', 'id': 101}, {'record_type': 'note', 'id': 999}])):
            result = runner.run(call('ask_clarification', {'question': 'Что выбрать?', 'kind': 'other',
                                'choices': candidates}, str(index)), job, NOW)
            self.assertEqual(result['error']['code'], 'invalid_choices')
            self.assertIsNone(self.store.get_meta('pending'))

    async def test_reminder_body_is_not_rewritten_as_technical_metadata(self):
        content = 'Проверить устройство (ID: 101) и памятку №42 в инструкции'
        with self.store.db:
            self.store.db.execute('INSERT INTO reminders(id,content,due_at,created_at,source_message_id) VALUES(?,?,?,?,?)',
                                 (8001, content, NOW, NOW, 1))
        runtime = Runtime(self.store, self.telegram, None, OWNER, lambda: NOW)
        runtime.schedule_once()
        await runtime.deliver_once()
        self.assertIn(content, self.telegram.sent[0][1])
        self.assertNotIn('№8001', self.telegram.sent[0][1])

    async def test_selected_target_survives_additional_clarification(self):
        seed_notes(self.store)
        self.store.enqueue(update('Исправь памятку про Олю'), NOW)
        await Agent(self.store, Model(answer(choices_call())), lambda: NOW).process(self.store.next_job())
        self.store.enqueue(update('2', 2), NOW)
        question = Model(answer(call('ask_clarification', {'question': 'Как исправить текст?',
                         'kind': 'other', 'continue_request': True})))
        await Agent(self.store, question, lambda: NOW).process(self.store.next_job())
        self.assertEqual(self.store.get_meta('pending')['selection']['id'], 9001)
        self.store.enqueue(update('Оля обещала не ругаться', 3), NOW)
        model = Model(answer(call('update_note', {'id': 2, 'content': 'Неверная правка', 'event_date': None}, 'wrong')),
                      answer(call('update_note', {'id': 9001, 'content': 'Оля обещала не ругаться', 'event_date': None}, 'right')),
                      answer(text='Памятка исправлена.'))
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        self.assertEqual(json.loads(model.inputs[1][-1]['content'])['error']['code'], 'selection_mismatch')
        self.assertEqual(self.store.db.execute('SELECT content FROM notes WHERE id=2').fetchone()[0], 'Не удалять эту запись')
        self.assertEqual(self.store.db.execute('SELECT content FROM notes WHERE id=9001').fetchone()[0], 'Оля обещала не ругаться')

    async def test_choices_survive_another_successful_tool_in_same_response(self):
        seed_notes(self.store)
        self.store.enqueue(update('Удали памятку про Олю и запомни: я купил молоко'), NOW)
        model = Model(answer(choices_call(), call('create_note', {'content': 'Купил молоко'}, 'save')))
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        self.assertIn('1. «Оля', self.store.history()[-1]['content'])
        self.assertIsNotNone(self.store.get_meta('pending'))
        self.assertEqual(self.store.get_meta('pending')['choices'][1]['id'], 9001)
        self.store.enqueue(update('2', 2), NOW)
        model = Model(answer(call('delete_note', {'id': 9001})), answer(text='Памятка удалена.'))
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=9001').fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM notes WHERE content='Купил молоко'").fetchone()[0], 1)

    async def test_cached_choices_are_restored_after_interruption_between_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resume.db'
            self.store.db.close()
            self.store = Store(path)
            seed_notes(self.store)
            self.store.enqueue(update('Удали памятку про Олю и запомни: купил молоко'), NOW)
            model = Model(answer(choices_call(), call('create_note', {'content': 'Купил молоко'}, 'save')))
            checkpoint = self.store.checkpoint
            def interrupt_after_tools(job_id, state):
                checkpoint(job_id, state)
                if sum(m['role'] == 'tool' for m in state['messages']) == 2:
                    raise asyncio.CancelledError
            with patch.object(self.store, 'checkpoint', side_effect=interrupt_after_tools):
                with self.assertRaises(asyncio.CancelledError):
                    await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
            self.assertIsNone(self.store.get_meta('pending'))
            self.store.db.close()
            self.store = Store(path)
            model = Model()
            await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
            self.assertEqual(model.inputs, [])
            self.assertEqual(self.store.get_meta('pending')['choices'][1]['id'], 9001)
            self.assertEqual(self.store.db.execute("SELECT count(*) FROM notes WHERE content='Купил молоко'").fetchone()[0], 1)

    async def test_new_creation_can_supersede_selected_records_clarification(self):
        seed_notes(self.store)
        pending = {'kind': 'other', 'question': 'Как исправить?', 'origin': update('Исправь памятку')['message'],
                   'selection': {'record_type': 'note', 'id': 9001, 'number': 2}}
        with self.store.db:
            self.store.set_meta('pending', pending)
        self.store.enqueue(update('Лучше сохрани новый факт: я купил молоко'), NOW)
        # The new note gets ID 9002; reading it in the same turn must not be
        # rejected by the old selected record's guard.
        model = Model(answer(call('create_note', {'content': 'Купил молоко', 'continue_request': False}, 'save')),
                      answer(call('get_note', {'id': 9002}, 'read')),
                      answer(text='Факт сохранён.'))
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        self.assertNotIn('error', json.loads(model.inputs[1][-1]['content']))
        self.assertNotIn('error', json.loads(model.inputs[2][-1]['content']))
        self.assertIsNone(self.store.get_meta('pending'))
        self.assertEqual(self.store.db.execute('SELECT content FROM notes WHERE id=9001').fetchone()[0], 'Оля обещала не материться')

    async def test_reminder_continuation_chunks_preserve_source_text(self):
        content = 'x' * 1778 + 'Памятка №42 (ID: 101)' + 'y' * 100
        with self.store.db:
            self.store.db.execute('INSERT INTO reminders(id,content,due_at,created_at,source_message_id) VALUES(?,?,?,?,?)',
                                 (8001, content, NOW, NOW, 1))
        runtime = Runtime(self.store, self.telegram, None, OWNER, lambda: NOW)
        runtime.schedule_once()
        while await runtime.deliver_once():
            pass
        self.assertIn(content, ''.join(message[1] for message in self.telegram.sent))
        self.assertEqual(self.telegram.sent[-1][2], 8001)

    async def test_quoted_answer_spanning_telegram_chunks_is_not_formatted_twice(self):
        body = 'x' * 1770 + 'Устройство (ID: 101) и памятка №42' + 'y' * 300
        self.store.enqueue(update('Что сохранили?'), NOW)
        model = Model(answer(text='Памятка №9001: «' + body + '».'))
        runtime = Runtime(self.store, self.telegram, Agent(self.store, model, lambda: NOW), OWNER, lambda: NOW)
        await runtime.process_once()
        while await runtime.deliver_once():
            pass
        delivered = ''.join(row[1] for row in self.telegram.sent)
        self.assertGreater(len(self.telegram.sent), 1)
        self.assertIn(body, delivered)
        self.assertNotIn('№9001', delivered)
        self.assertEqual(delivered, 'Памятка: «' + body + '».')

    async def test_notification_hides_id_but_button_keeps_internal_reference(self):
        with self.store.db:
            self.store.db.execute('INSERT INTO reminders(id,content,due_at,created_at,source_message_id) VALUES(?,?,?,?,?)',
                                 (8001, 'Выпить 2 стакана воды', NOW, NOW, 1))
        runtime = Runtime(self.store, self.telegram, None, OWNER, lambda: NOW)
        runtime.schedule_once()
        await runtime.deliver_once()
        _, text, callback = self.telegram.sent[0]
        self.assertNotIn('8001', text)
        self.assertIn('2 стакана воды', text)
        self.assertEqual(callback, 8001)
        self.assertEqual(self.store.snooze(callback, NOW + 10), 'Отложено на 10 минут.')

    async def test_existing_queued_notification_is_also_formatted_on_delivery(self):
        with self.store.db:
            self.store.db.execute('INSERT INTO outbox(key,text,available_at,reminder_id) VALUES(?,?,?,?)',
                                 ('legacy:0000:last', '⏰ Напоминание №8001\nПочистить зубы', NOW, 8001))
        runtime = Runtime(self.store, self.telegram, None, OWNER, lambda: NOW)
        await runtime.deliver_once()
        self.assertEqual(self.telegram.sent[0][1], '⏰ Напоминание\nПочистить зубы')

    async def test_direct_answer_and_clarification_do_not_reveal_legacy_ids(self):
        self.store.enqueue(update('Что обещала Оля?'), NOW)
        await Agent(self.store, Model(answer(text='Памятка №9001: Оля обещала не материться.')), lambda: NOW).process(self.store.next_job())
        self.assertEqual(self.store.history()[-1]['content'], 'Памятка: Оля обещала не материться.')
        self.assertIn('9001', json.loads(self.store.db.execute('SELECT checkpoint FROM jobs WHERE id=1').fetchone()[0])['messages'][-1]['content'])
        self.store.enqueue(update('Перенеси напоминание про зубы', 2), NOW)
        model = Model(answer(call('ask_clarification', {'question': 'На какое время перенести напоминание №8001?', 'kind': 'other'})))
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        self.assertNotIn('8001', self.store.history()[-1]['content'])
