"""Scripted offline protocols: these do not test an LLM's query selection."""
import json
import unittest

from personal_assistent.agent import Agent, SYSTEM
from personal_assistent.context import Budget
from personal_assistent.search import index_values
from personal_assistent.storage import Store
from test_assistant import NOW, Model, answer, call, update


class MemoryRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.addCleanup(self.store.db.close)

    def seed(self, *texts):
        with self.store.db:
            for identity, text in enumerate(texts, 1):
                self.store.db.execute(
                    'INSERT INTO notes(id,content,original,event_date,created_at,updated_at,source_message_id) '
                    'VALUES(?,?,?,?,?,?,?)', (identity, text, text, '2026-10-09', NOW, NOW, identity))
                self.store.db.execute('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)',
                                      (identity, *index_values(text)))

    async def run_agent(self, question, *responses):
        self.store.enqueue(update(question), NOW)
        model = Model(*responses)
        await Agent(self.store, model, clock=lambda: NOW).process(self.store.next_job())
        self.assertEqual(model.responses, [])
        self.assertTrue(all(Budget().fits(messages) for messages in model.inputs))
        # Check delivery of the policy, not merely its presence in a module.
        self.assertIn(SYSTEM, model.inputs[0][0]['content'])
        self.assertIn('даже при непустой выдаче', model.inputs[0][0]['content'])
        self.assertIn('Для истории/списка/нескольких фактов', model.inputs[0][0]['content'])
        self.assertIn('один короткий иной запрос', model.inputs[0][0]['content'])
        return model

    def result(self, model, request_index, identity):
        return json.loads(next(m['content'] for m in model.inputs[request_index]
                               if m['role'] == 'tool' and m['tool_call_id'] == identity))

    async def test_irrelevant_nonempty_search_then_retry_and_original(self):
        question = 'Как сэкономил на отоплении?'
        fact = 'Поставил уплотнитель толщиной 8 мм.'
        target = ('Утеплил оконные рамы. Перед работой очистил поверхность, проверил крепления, '
                  'убрал старый клей и просушил древесину. Мастер посоветовал проверить прилегание '
                  'створок по всему периметру и сохранить чек. ' + fact)
        self.seed('В инструкции описано центральное отопление.', target)
        model = await self.run_agent(question,
            answer(call('search_notes', {'query': question}, 'first')),
            answer(call('search_notes', {'query': 'утеплил'}, 'retry')),
            answer(call('get_note', {'id': 2}, 'original')),
            answer(text='Утеплил оконные рамы, поставил уплотнитель толщиной 8 мм.'))
        self.assertEqual([n['id'] for n in self.result(model, 1, 'first')['items']], [1])
        preview = self.result(model, 2, 'retry')['items'][0]
        self.assertEqual(preview['id'], 2)
        self.assertNotIn('original', preview)
        self.assertNotIn(fact, json.dumps(model.inputs[2], ensure_ascii=False))
        self.assertEqual(self.result(model, 3, 'original')['original'], target)
        self.assertIn(fact, json.dumps(model.inputs[3], ensure_ascii=False))
        self.assertIn('8 мм', self.store.history()[-1]['content'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 2)

    async def test_enumeration_keeps_useful_first_source_and_reads_second(self):
        question = 'Какие траты были на домашний свет?'
        self.seed('За свет в кабинете заплатил 900 рублей.',
                  'Для освещения спальни торшер обошёлся в 1700 рублей.',
                  'Электричество в беседке пока не подключал.')
        model = await self.run_agent(question,
            answer(call('search_notes', {'query': question}, 'first')),
            answer(call('get_note', {'id': 1}, 'read-first')),
            answer(call('search_notes', {'query': 'освещение'}, 'retry')),
            answer(call('get_note', {'id': 2}, 'read-second')),
            answer(text='900 рублей за свет в кабинете и 1700 рублей за торшер; всего 2600 рублей.'))
        self.assertEqual([n['id'] for n in self.result(model, 1, 'first')['items']], [1])
        self.assertEqual([n['id'] for n in self.result(model, 3, 'retry')['items']], [2])
        self.assertIn('900 рублей', self.result(model, 4, 'read-first')['original'])
        self.assertIn('1700 рублей', self.result(model, 4, 'read-second')['original'])
        self.assertNotIn('беседке', self.store.history()[-1]['content'])

    async def test_sufficient_single_fact_finishes_without_retry(self):
        question = 'Какой код у посылки для Веры?'
        self.seed('Посылка для Веры: код VR508, не VR580.',
                  'Посылка для Василия: код VR580.')
        model = await self.run_agent(question,
            answer(call('search_notes', {'query': question}, 'first')),
            answer(call('get_note', {'id': 1}, 'original')),
            answer(text='Код посылки для Веры — VR508.'))
        self.assertIn('не VR580', self.result(model, 2, 'original')['original'])
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(self.store.history()[-1]['content'], 'Код посылки для Веры — VR508.')

    async def test_unsuccessful_retry_uses_bounded_list_and_reports_partial_search(self):
        question = 'У кого скрипка?'
        self.seed('В каталоге нарисована скрипка.', 'Запасные струны лежат в шкафу.',
                  'Купил бумагу.', 'Вынес мусор.', 'Полил цветы.', 'Протёр стол.')
        model = await self.run_agent(question,
            answer(call('search_notes', {'query': question}, 'first')),
            answer(call('get_note', {'id': 1}, 'original')),
            answer(call('search_notes', {'query': 'владелец инструмента'}, 'retry')),
            answer(call('list_notes', {'limit': 2}, 'list')),
            answer(text='В проверенной части памяти не нашёл, у кого скрипка; просмотр неполный.'))
        self.assertEqual(self.result(model, 3, 'retry')['items'], [])
        page = self.result(model, 4, 'list')
        self.assertEqual(len(page['items']), 2)
        self.assertEqual(page['total'], 6)
        self.assertEqual(page['next_offset'], 2)
        self.assertTrue(page['incomplete'])
        self.assertIn('просмотр неполный', self.store.history()[-1]['content'])
