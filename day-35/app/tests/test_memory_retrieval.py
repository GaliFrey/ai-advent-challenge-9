import json
import unittest
from unittest.mock import patch

from personal_assistent.agent import Agent
from personal_assistent.context import Budget, Context
from personal_assistent.memory import absence_scopes, checked_scopes, note_page_preview, unchecked_scopes
from personal_assistent.search import index_values
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner
from test_assistant import NOW, Model, answer, call, update


def seed(store, count=50, content='Памятка про обслуживание принтера'):
    for identity in range(1, count + 1):
        text = f'{content} {identity}'
        store.db.execute('INSERT INTO notes(id,content,original,event_date,created_at,updated_at,source_message_id) VALUES(?,?,?,?,?,?,?)',
                         (identity, text, text, '2026-09-28', NOW, NOW, identity))
        store.db.execute('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)', (identity, *index_values(text)))
    store.db.commit()


class MemoryEvidenceTests(unittest.TestCase):
    def test_common_russian_and_english_absence_claims(self):
        for text in ('Памяток про телефон нет.', 'В записях ничего не найдено.',
                     'Не знаю — в нашей переписке этого не было.', 'Не вижу таких заметок.',
                     "I couldn't find any notes.", 'There are no saved records.'):
            with self.subTest(text=text):
                self.assertTrue(unchecked_scopes(text, []))

    def test_quoted_examples_and_ordinary_chat_do_not_trigger_reads(self):
        for text in ('Фраза «Записей нет» означает отсутствие записей.',
                     'В цитате "There are no notes" говорится об отсутствии.',
                     'Не знаю погоду на завтра.', 'Привет!', 'Ничего страшного.'):
            with self.subTest(text=text):
                self.assertFalse(absence_scopes(text))

    def test_short_negative_answer_to_explicit_memory_question_is_checked(self):
        self.assertEqual(unchecked_scopes('Ничего не найдено.', [], 'Найди памятки о телефоне'), {'notes'})
        self.assertEqual(unchecked_scopes('Не помню.', []), {'any'})

    def test_only_current_successful_read_from_matching_source_counts(self):
        chain = [answer(call('list_reminders', {}, 'r')),
                 {'role': 'tool', 'tool_call_id': 'r', 'content': json.dumps({'items': [], 'total': 0})}]
        self.assertEqual(checked_scopes(chain), {'reminders'})
        self.assertTrue(unchecked_scopes('Памяток нет.', chain))
        self.assertFalse(unchecked_scopes('Напоминаний нет.', chain))
        for content in ('{}', '{broken', '{"error":{"code":"failed"}}', '{"deferred":true}', '{"items":[]}'):
            bad = [answer(call('search_notes', {'query': 'x'}, 'n')),
                   {'role': 'tool', 'tool_call_id': 'n', 'content': content}]
            self.assertEqual(checked_scopes(bad), set())

    def test_preview_bounds_json_escaping_and_keeps_date_distinction(self):
        for text in ('я' * 1000, '🛠' * 1000, '\x01' * 1000):
            raw = {'items': [{'id': 1, 'content': text, 'original': 'Исходная цитата',
                             'event_date': '2026-09-28', 'created_at': '2026-10-16T12:00:00+03:00'}],
                   'total': 1, 'next_offset': None, 'incomplete': False}
            result = note_page_preview(raw)
            self.assertTrue(result['incomplete'])
            self.assertNotIn('original', result['items'][0])
            self.assertLessEqual(len(json.dumps(result['items'][0]['content'], ensure_ascii=False).encode()), 480)
            self.assertEqual(result['items'][0]['event_date'], '2026-09-28')
            self.assertEqual(result['items'][0]['note_saved_at'], '2026-10-16T12:00:00+03:00')
            self.assertEqual(raw['items'][0]['content'], text)

    def test_legacy_large_cached_pages_do_not_skip_unshown_items(self):
        items = [{'id': n, 'content': str(n)} for n in range(25)]
        for total, following in ((50, 25), (25, None)):
            result = note_page_preview({'items': items, 'total': total, 'next_offset': following, 'incomplete': False})
            self.assertEqual(len(result['items']), 5)
            self.assertEqual(result['next_offset'], 5)
            self.assertTrue(result['incomplete'])


class RetrievalAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.addCleanup(self.store.db.close)

    async def run_agent(self, model, text='Когда ремонтировали телефон?'):
        self.store.enqueue(update(text), NOW)
        agent = Agent(self.store, model, clock=lambda: NOW)
        await agent.process(self.store.next_job())
        return agent

    async def test_unchecked_absence_triggers_read_and_corrected_answer(self):
        seed(self.store, 1, 'В телефоне заменили разбитое стекло')
        model = Model(answer(text='Не знаю — в нашей переписке этого не было.'),
                      answer(call('search_notes', {'query': 'телефон стекло'}, 'retry')),
                      answer(text='Стекло заменили 28 сентября 2026 года.'))
        await self.run_agent(model)
        self.assertEqual(len(model.inputs), 3)
        self.assertIn('28 сентября', self.store.history()[-1]['content'])
        self.assertNotIn('этого не было', str(self.store.history()))
        # Two searches, then one original read by the separate collector.
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 3)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_failed_verification_cannot_be_used_to_claim_absence(self):
        model = Model(answer(text='Памяток нет.'), answer(text='Памяток нет.'))
        with patch.object(ToolRunner, '_apply', return_value={'error': {'code': 'unavailable'}}):
            await self.run_agent(model)
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(self.store.history()[-1]['content'], 'Не удалось проверить память по этому вопросу. Нельзя утверждать, что записи отсутствуют.')

    async def test_verified_empty_search_checks_empty_list_in_collection(self):
        model = Model(answer(call('search_notes', {'query': 'телефон'}, 'read')),
                      answer(text='В выполненном поиске памяток про телефон не найдено.'))
        await self.run_agent(model)
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 2)

    async def test_ordinary_chat_does_not_read_memory(self):
        model = Model(answer(text='Привет!'))
        await self.run_agent(model, 'Привет')
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 0)

    async def test_parallel_search_and_list_is_bounded_and_fits_real_budget(self):
        seed(self.store)
        model = Model(answer(call('search_notes', {'query': 'принтер', 'limit': 50}, 'search'),
                             call('list_notes', {'limit': 50}, 'list')),
                      answer(text='Есть памятки про обслуживание принтера.'))
        await self.run_agent(model, 'Что записано про принтер?')
        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(Budget().fits(model.inputs[-1]))
        results = {r['key'].split(':')[-1]: json.loads(r['result']) for r in self.store.db.execute('SELECT * FROM tool_results')}
        self.assertTrue(results['list']['deferred'])
        self.assertEqual(len(results['search']['items']), 5)
        self.assertEqual(results['search']['next_offset'], 5)
        self.assertIn('original', results['search']['items'][0])
        projected = [json.loads(m['content']) for m in model.inputs[-1] if m['role'] == 'tool']
        self.assertNotIn('original', projected[0]['items'][0])
        self.assertEqual(projected[0]['total'], 50)

    async def test_full_original_can_be_read_after_preview(self):
        seed(self.store, 1, 'Текст ' * 150)
        model = Model(answer(call('list_notes', {}, 'list')),
                      answer(call('get_note', {'id': 1}, 'read')),
                      answer(text='Прочитал исходную запись.'))
        await self.run_agent(model, 'Прочитай оригинал памятки')
        page = json.loads(next(m['content'] for m in model.inputs[1] if m['role'] == 'tool'))
        full = json.loads(model.inputs[2][-1]['content'])
        self.assertTrue(page['items'][0]['incomplete'])
        self.assertGreater(len(full['content']), len(page['items'][0]['content']))
        self.assertEqual(full['original'], full['content'])

    async def test_verification_survives_checkpoint_without_repeating_search(self):
        seed(self.store, 1, 'Ремонт телефона')
        model = Model(answer(text='В записях ничего нет.'), RuntimeError('interruption'))
        self.store.enqueue(update('Когда ремонтировали телефон?'), NOW)
        agent = Agent(self.store, model, clock=lambda: NOW)
        with self.assertRaises(RuntimeError):
            await agent.process(self.store.next_job())
        job = dict(self.store.db.execute('SELECT * FROM jobs').fetchone())
        self.assertTrue(json.loads(job['checkpoint'])['memory_guard_attempted'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 1)
        resumed = Agent(self.store, Model(answer(text='Ремонт был 28 сентября.')), clock=lambda: NOW)
        await resumed.process(job)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 2)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM tool_results WHERE key NOT LIKE '%:memory_collect_%'").fetchone()[0], 1)
        self.assertIn('28 сентября', self.store.history()[-1]['content'])

    async def test_reading_wrong_source_does_not_authorize_absence_claim(self):
        model = Model(answer(call('get_profile', {}, 'profile')),
                      answer(text='Памяток про телефон нет.'),
                      answer(text='В выполненном поиске памяток про телефон не найдено.'))
        await self.run_agent(model)
        self.assertEqual(len(model.inputs), 3)
        names = [c['function']['name'] for m in model.inputs[-1] if m['role'] == 'assistant' for c in m.get('tool_calls', [])]
        self.assertIn('search_notes', names)

    async def test_clarification_cannot_deliver_unchecked_absence(self):
        model = Model(answer(call('ask_clarification', {'question': 'Памяток нет. Создать новую?', 'kind': 'other'}, 'bad')),
                      answer(call('search_notes', {'query': 'телефон'}, 'read')),
                      answer(text='В выполненном поиске памяток про телефон не найдено.'))
        await self.run_agent(model)
        bad = json.loads(self.store.db.execute("SELECT result FROM tool_results WHERE key LIKE '%:bad'").fetchone()[0])
        self.assertEqual(bad['error']['code'], 'memory_not_checked')
        self.assertIsNone(self.store.get_meta('pending'))
        self.assertNotIn('Создать новую?', str(self.store.history()))

    async def test_memory_verification_preserves_committed_action_confirmation(self):
        class ExactModel(Model):
            async def complete(self, messages):
                self.inputs.append(json.loads(json.dumps(messages)))
                return self.responses.pop(0)
        model = ExactModel(answer(call('create_note', {'content': 'Купил фильтр'}, 'saved')),
                           answer(text=json.dumps({'kind': 'mixed', 'tool_call_ids': ['saved'], 'text': 'Других памяток нет.'}, ensure_ascii=False)),
                           answer(text=json.dumps({'kind': 'mixed', 'tool_call_ids': ['saved'], 'text': 'В текущем поиске нашёл памятку про фильтр.'}, ensure_ascii=False)),
                           answer(text=json.dumps({'kind': 'mixed', 'tool_call_ids': ['saved'], 'text': 'В текущем поиске нашёл памятку про фильтр.'}, ensure_ascii=False)))
        await self.run_agent(model, 'Запомни: купил фильтр. Что ещё есть в памятках?')
        result = self.store.history()[-1]['content']
        self.assertIn('Памятка сохранена', result)
        self.assertIn('нашёл памятку', result)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_exhausted_read_budget_returns_partial_search_instead_of_overflow(self):
        seed(self.store)
        responses = [answer(call('list_notes', {'offset': n * 5}, f'page{n}')) for n in range(6)]
        model = Model(*responses, answer(text='Поиск неполон: в проверенной части памяти не нашёл подходящую запись.'))
        await self.run_agent(model, 'Просмотри памятки про неизвестный объект')
        self.assertEqual(len(model.inputs), 7)
        self.assertTrue(all(Budget().fits(request) for request in model.inputs))
        results = [json.loads(r[0]) for r in self.store.db.execute('SELECT result FROM tool_results')]
        self.assertTrue(any(r.get('limited') for r in results))
        self.assertIn('Поиск неполон', self.store.history()[-1]['content'])


class NotePaginationTests(unittest.TestCase):
    def test_requested_large_page_is_capped_without_skipping_notes(self):
        store = Store(':memory:')
        self.addCleanup(store.db.close)
        seed(store, 17)
        store.enqueue(update(), NOW)
        runner, job = ToolRunner(store), store.next_job()
        for name, args in (('list_notes', {}), ('search_notes', {'query': 'принтер'})):
            ids, offset = [], 0
            while True:
                result = runner.run(call(name, {**args, 'limit': 50, 'offset': offset}, f'{name}-{offset}'), job, NOW)
                self.assertLessEqual(len(result['items']), 5)
                ids.extend(item['id'] for item in result['items'])
                if result['next_offset'] is None:
                    break
                self.assertGreater(result['next_offset'], offset)
                offset = result['next_offset']
            self.assertEqual(set(ids), set(range(1, 18)))
            self.assertEqual(len(ids), 17)
