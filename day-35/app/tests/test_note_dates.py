import json
import unittest

from personal_assistent.agent import Agent
from personal_assistent.context import Budget, Context
from personal_assistent.memory import note_page_preview
from personal_assistent.search import index_values
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner
from test_assistant import NOW, Model, answer, call, update
from test_receipts import Scripted, mixed_answer


SAVED = '2026-10-16T15:00:00+03:00'
BOOKING = '2026-10-10'
TEXT = 'Записался к стоматологу на 3 ноября в 15:00.'


class NoteDateContractTests(unittest.TestCase):
    def note(self, event_date=BOOKING):
        return {'id': 1, 'content': TEXT, 'original': 'Запомни: ' + TEXT,
                'event_date': event_date, 'created_at': SAVED, 'updated_at': '2026-10-17T12:00:00+03:00'}

    def test_preview_labels_booking_date_and_note_save_time_separately(self):
        raw = {'items': [self.note()], 'total': 1, 'next_offset': None, 'incomplete': False}
        result = note_page_preview(raw)
        item = result['items'][0]
        self.assertEqual(item['event_date'], BOOKING)
        self.assertEqual(item['note_saved_at'], SAVED)
        self.assertNotIn('created_at', item)
        self.assertIn('сохран', result['date_meanings']['note_saved_at'])
        self.assertIn('действ', result['date_meanings']['event_date'])
        self.assertEqual(raw['items'][0]['created_at'], SAVED)
        self.assertIn('3 ноября', item['content'])

    def test_unknown_event_date_stays_unknown_even_with_save_and_edit_times(self):
        raw = {'items': [self.note(None)], 'total': 1, 'next_offset': None}
        item = note_page_preview(raw)['items'][0]
        self.assertIsNone(item['event_date'])
        self.assertEqual(item['note_saved_at'], SAVED)

    def test_detail_read_uses_same_meanings_and_preserves_original(self):
        note = self.note()
        chain = [answer(call('get_note', {'id': 1}, 'read')),
                 {'role': 'tool', 'tool_call_id': 'read', 'content': json.dumps(note, ensure_ascii=False)}]
        projected = Context.model_messages(chain)
        detail = json.loads(projected[-1]['content'])
        self.assertEqual(detail['event_date'], BOOKING)
        self.assertEqual(detail['note_saved_at'], SAVED)
        self.assertEqual(detail['note_edited_at'], '2026-10-17T12:00:00+03:00')
        self.assertEqual(detail['original'], note['original'])
        self.assertNotIn('created_at', detail)
        self.assertNotIn('updated_at', detail)
        self.assertEqual(json.loads(chain[-1]['content']), note)
        self.assertEqual(Context.model_messages(projected), projected)

    def test_meanings_are_shared_but_each_record_keeps_its_dates(self):
        chain = [answer(call('search_notes', {'query': 'стоматолог'}, 'search')),
                 {'role': 'tool', 'tool_call_id': 'search',
                  'content': json.dumps({'items': [self.note()], 'total': 1, 'next_offset': None})},
                 answer(call('get_note', {'id': 1}, 'read')),
                 {'role': 'tool', 'tool_call_id': 'read', 'content': json.dumps(self.note())}]
        projected = Context.model_messages(chain)
        page, detail = (json.loads(projected[i]['content']) for i in (1, 3))
        self.assertEqual(set(page['date_meanings']), {'event_date', 'note_saved_at'})
        self.assertEqual(set(detail['date_meanings']), {'note_edited_at'})
        self.assertEqual(detail['event_date'], BOOKING)
        self.assertEqual(detail['note_saved_at'], SAVED)
        self.assertEqual(Context.model_messages(projected), projected)

    def test_failed_detail_read_does_not_gain_fabricated_dates(self):
        error = {'error': {'code': 'note_not_found'}}
        chain = [answer(call('get_note', {'id': 9}, 'read')),
                 {'role': 'tool', 'tool_call_id': 'read', 'content': json.dumps(error)}]
        self.assertEqual(json.loads(Context.model_messages(chain)[-1]['content']), error)


class NoteDateAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_note_read_paths_have_same_dates_and_fit_default_budget(self):
        store = Store(':memory:')
        self.addCleanup(store.db.close)
        store.enqueue(update('Когда я записался к стоматологу и когда сохранил памятку?'), NOW)
        store.db.execute('INSERT INTO notes(content,original,event_date,created_at,updated_at,source_message_id) VALUES(?,?,?,?,?,?)',
                         (TEXT, TEXT, BOOKING, NOW, NOW + 3600, 1))
        store.db.execute('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)', (1, *index_values(TEXT)))
        store.db.commit()
        model = Model(answer(call('search_notes', {'query': 'стоматолог'}, 'search')),
                      answer(call('get_note', {'id': 1}, 'read')),
                      answer(text='Записался 10 октября; памятка сохранена 16 октября. Приём — 3 ноября.'))
        await Agent(store, model, clock=lambda: NOW).process(store.next_job())
        self.assertEqual(len(model.inputs), 3)
        page = json.loads(model.inputs[1][-1]['content'])
        detail = json.loads(model.inputs[2][-1]['content'])
        self.assertEqual(page['items'][0]['event_date'], detail['event_date'])
        self.assertEqual(page['items'][0]['note_saved_at'], detail['note_saved_at'])
        self.assertTrue(all(Budget().fits(messages) for messages in model.inputs))
        raw = json.loads(store.db.execute("SELECT result FROM tool_results WHERE key LIKE '%:read'").fetchone()[0])
        self.assertIn('created_at', raw)
        self.assertNotIn('note_saved_at', raw)
        self.assertEqual(store.db.execute('SELECT event_date FROM notes').fetchone()[0], BOOKING)


class NoteWordingAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.addCleanup(self.store.db.close)
        self.store.db.execute('INSERT INTO notes(content,original,event_date,created_at,updated_at,source_message_id) VALUES(?,?,?,?,?,?)',
                              (TEXT, TEXT, BOOKING, NOW, NOW, 1))
        self.store.db.commit()
        self.store.enqueue(update('Какие есть памятки о стоматологе?'), NOW)

    async def run_agent(self, model):
        await Agent(self.store, model, clock=lambda: NOW).process(self.store.next_job())
        return self.store.history()[-1]['content']

    async def test_ambiguous_date_is_rephrased_without_repeating_read(self):
        good = 'На приём записался 10 октября; приём 3 ноября. Памятку сохранил 16 октября.'
        model = Model(answer(call('get_note', {'id': 1}, 'read')),
                      answer(text='Запись сделана 10 октября.'), answer(text=good))
        self.assertEqual(await self.run_agent(model), good)
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 1)
        self.assertNotIn('Запись сделана', str(self.store.history()))
        self.assertEqual(self.store.db.execute('SELECT content FROM notes').fetchone()[0], TEXT)

    async def test_repeated_ambiguity_stops_after_one_correction(self):
        model = Model(answer(call('get_note', {'id': 1}, 'read')),
                      answer(text='Запись сделана 10 октября.'), answer(text='Запись была сделана 10 октября.'))
        result = await self.run_agent(model)
        self.assertNotIn('сделана', result)
        self.assertIn('Не удалось', result)
        self.assertEqual(len(model.inputs), 3)

    async def test_source_quotes_and_explicit_actions_do_not_trigger_correction(self):
        text = 'В цитате «запись сделана 10 октября». На приём записался 10 октября; памятку сохранил 16 октября и изменил 17 октября.'
        model = Model(answer(call('get_note', {'id': 1}, 'read')), answer(text=text))
        self.assertEqual(await self.run_agent(model), text)
        self.assertEqual(len(model.inputs), 2)

    async def test_restart_after_rejected_draft_preserves_single_correction(self):
        model = Model(answer(call('get_note', {'id': 1}, 'read')),
                      answer(text='Запись сделана 10 октября.'), RuntimeError('network'))
        with self.assertRaises(RuntimeError):
            await self.run_agent(model)
        good = 'На приём записался 10 октября.'
        resumed = Model(answer(text=good))
        job = dict(self.store.db.execute('SELECT * FROM jobs').fetchone())
        await Agent(self.store, resumed, clock=lambda: NOW).process(job)
        self.assertEqual(self.store.history()[-1]['content'], good)
        self.assertEqual(len(resumed.inputs), 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 1)

    async def test_correction_preserves_action_receipt_without_duplicate_writes(self):
        model = Scripted(answer(call('create_note', {'content': 'Проверочный факт'}, 'write'),
                                call('get_note', {'id': 1}, 'read')),
                         mixed_answer('Запись сделана 10 октября.', 'write'),
                         mixed_answer('На приём записался 10 октября.', 'write'),
                         mixed_answer('На приём записался 10 октября.', 'write'))
        result = await self.run_agent(model)
        self.assertIn('На приём записался 10 октября.', result)
        self.assertIn('Проверочный факт', result)
        self.assertNotIn('Запись сделана', result)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 2)
        self.assertEqual(len(self.store.receipts(1)), 1)
