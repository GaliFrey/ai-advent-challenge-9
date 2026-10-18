"""Explicit side-protocol tests; scripted choices are not LLM-quality scores."""
import json
import unittest

import httpx

from personal_assistent.agent import Agent, DeepSeek
from personal_assistent.context import Budget
from personal_assistent.search import index_values
from personal_assistent.storage import Store, dump
from test_assistant import Model, answer, call, update, NOW


def decision(**fields):
    return {'role': 'assistant', 'content': dump(fields)}


def scope(kind='single', query='', target='Предмет вопроса', criterion='Факты, отвечающие вопросу'):
    return decision(scope=kind, target=target, criterion=criterion, query=query)


def verification(text, ids, excluded=(), uncertain=()):
    def reply(payload):
        reviews = []
        for note in payload['evidence']:
            identity = note['id']
            status = 'include' if identity in ids else 'uncertain' if identity in uncertain else 'exclude'
            quote = ''.join(p['content'] for p in note['fragments'])[:100] if status == 'include' else ''
            reviews.append(dict(id=identity, decision=status, quote=quote,
                                relation='Тестовая связь с вопросом' if status == 'include' else 'Связь не подтверждена',
                                status='unknown'))
        assert {n['id'] for n in payload['evidence']} == set(ids) | set(excluded) | set(uncertain)
        return decision(answer=text, ids=ids, reviews=reviews)
    return reply


class CollectingModel(Model):
    def __init__(self, responses=(), decisions=()):
        super().__init__(*responses)
        self.decisions = list(decisions)
        self.retrieval_inputs = []

    async def retrieve(self, messages):
        self.retrieval_inputs.append(json.loads(dump(messages)))
        response = self.decisions.pop(0)
        if isinstance(response, Exception):
            raise response
        return response(json.loads(messages[-1]['content'])) if callable(response) else response


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.addCleanup(self.store.db.close)

    def seed(self, *texts):
        with self.store.db:
            for i, text in enumerate(texts, 1):
                self.store.db.execute('INSERT INTO notes VALUES(?,?,?,?,?,?,?,?)',
                                      (i, text, text, '2026-10-08', NOW, NOW, i, 0))
                self.store.db.execute('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)',
                                      (i, *index_values(text)))

    async def run_agent(self, question, decisions):
        self.store.enqueue(update(question), NOW)
        model = CollectingModel([answer(call('search_notes', {'query': question}, 'first')),
                                 answer(text='Непроверенный черновик')], decisions)
        job = self.store.next_job()
        await Agent(self.store, model, clock=lambda: NOW).process(job)
        self.assertEqual(model.decisions, [])
        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(all(Budget().fits(m, tools=False) for m in model.retrieval_inputs))
        phase = json.loads(self.store.db.execute('SELECT checkpoint FROM jobs WHERE id=?', (job['id'],)).fetchone()[0])['memory_collection']
        return model, phase, self.store.history()[-1]['content']

    async def test_alternative_search_and_full_original_beyond_preview(self):
        text = 'Утеплил рамы. ' + 'Описание работ. ' * 150 + 'Уплотнитель толщиной 8 мм.'
        self.seed('Отопление обсуждали на собрании.', text)
        model, phase, reply = await self.run_agent('Что сделал для отопления?', [
            scope('single', query='утеплил'),
            decision(sufficient=False, ids=[]),
            decision(sufficient=True, ids=[2]),
            verification('Утеплил рамы, уплотнитель 8 мм.', [2])])
        self.assertEqual(phase['results']['alternative']['items'][0]['id'], 2)
        original = ''.join(p['original'] for p in phase['evidence'][0]['fragments'])
        self.assertEqual(original, text)
        self.assertIn('8 мм', reply)
        self.assertNotIn('Инструменты возвращают', model.retrieval_inputs[0][0]['content'])
        final = json.loads(model.retrieval_inputs[-1][-1]['content'])
        self.assertNotIn('candidates', final)
        self.assertNotIn('Отопление обсуждали', dump(final['evidence']))

    async def test_collection_forces_scan_and_preserves_first_source(self):
        self.seed('Ремонт кресла — 700 рублей.', 'Столяр укрепил полку — 900 рублей.',
                  'Сервис доставки продуктов открыт.')
        _, phase, reply = await self.run_agent('Какой ремонт делал?', [
            scope('collection', query='столяр'),
            decision(sufficient=True, ids=[1, 2]),
            verification('Кресло — 700 рублей; полка — 900 рублей.', [1, 2])])
        self.assertTrue(phase['scan_complete'])
        self.assertIn('alternative', phase['results'])
        self.assertEqual({e['id'] for e in phase['evidence']}, {1, 2})
        self.assertNotIn('доставки', reply)

    async def test_failed_alternative_forces_list_without_model_tool_call(self):
        self.seed('Зонт чёрный.', 'Получил возмещение за билет.')
        _, phase, reply = await self.run_agent('Возврат денег?', [
            scope('single', query='компенсация'),
            decision(sufficient=False, ids=[]),
            decision(sufficient=True, ids=[2]),
            verification('Возмещение за билет.', [2])])
        self.assertTrue(phase['scan_complete'])
        self.assertEqual(phase['calls'], 4)
        self.assertIn('list_0', phase['results'])
        self.assertIn('билет', reply)

    async def test_single_fact_has_no_extra_search_and_keeps_negative_code(self):
        self.seed('Для Веры код VR508, не VR580.')
        _, phase, reply = await self.run_agent('Какой код для Веры?', [
            scope('single', query=''),
            decision(sufficient=True, ids=[1]),
            verification('VR508, не VR580.', [1])])
        self.assertNotIn('alternative', phase['results'])
        self.assertEqual(phase['pages'], 0)
        self.assertIn('не VR580', reply)

    async def test_mixed_read_answer_keeps_other_verified_sections(self):
        self.seed('Для Веры код VR508.')
        self.store.enqueue(update('Код Веры и мои напоминания?'), NOW)
        model = CollectingModel([
            answer(call('search_notes', {'query': 'Вера'}, 'notes'),
                   call('list_reminders', {}, 'reminders')),
            answer(text='VR508. Напоминаний нет.'),
            answer(text='Код VR508. Напоминаний нет.')], [
                scope('single', query=''),
            decision(sufficient=True, ids=[1]),
                verification('Код VR508.', [1])])
        await Agent(self.store, model, clock=lambda: NOW).process(self.store.next_job())
        self.assertEqual(self.store.history()[-1]['content'], 'Код VR508. Напоминаний нет.')
        self.assertEqual(len(model.retrieval_inputs), 3)
        merged = model.inputs[-1]
        self.assertTrue(any(m['role'] == 'tool' and m.get('tool_call_id') == 'reminders' for m in merged))
        self.assertIn('Код VR508.', merged[-1]['content'])

    async def test_large_candidate_packet_shrinks_without_dropping_ids(self):
        self.seed(*('\\' * 300 for _ in range(80)))
        model, phase, reply = await self.run_agent('Все записи?', [
            scope('collection', query=''),
            decision(sufficient=True, ids=[1]),
            verification('Сохранена последовательность символов.', [1])])
        selection = json.loads(model.retrieval_inputs[1][-1]['content'])
        self.assertEqual({n['id'] for n in selection['candidates']}, set(range(1, 81)))
        self.assertTrue(phase['limited'])
        self.assertNotIn('error_type', phase)
        self.assertIn('Поиск неполный', reply)

    async def test_question_scope_has_no_draft_candidates_or_searches(self):
        self.seed('В мастерскую отдал фен.', 'Планшет отдал мастеру: не включался.',
                  'У приёмника восстановили звук.')
        model, phase, reply = await self.run_agent('Какую электронику отдавал чинить и почему?', [
            scope('collection', criterion='Перечень устройств, переданных для исправления, и причины'),
            decision(sufficient=True, ids=[1, 2, 3]),
            verification('Фен; планшет — не включался; приёмник — звук.', [1, 2, 3])])
        first = json.loads(model.retrieval_inputs[0][-1]['content'])
        self.assertEqual(set(first), {'task', 'question', 'question_time'})
        self.assertEqual(first['question'], 'Какую электронику отдавал чинить и почему?')
        self.assertTrue(phase['scan_complete'])
        self.assertEqual(phase['included_ids'], [1, 2, 3])
        self.assertIn('планшет', reply)
        self.assertIn('Перечень устройств', json.loads(model.retrieval_inputs[1][-1]['content'])['criterion'])

    async def test_verification_filters_read_noise_without_losing_indirect_facts(self):
        self.seed('Мастер выправил погнутую раму велосипеда.',
                  'Лезвия ножниц заточили.',
                  'Не ремонтировал чемодан: ручку временно примотал сам.',
                  'Треснувшую тарелку выбросил.',
                  'Пальто вернули из гардероба без шарфа.',
                  'Купил цветной абажур ради красивого света.')
        model, phase, reply = await self.run_agent('Что было с ремонтом вещей?', [
            scope('collection', criterion='Исправление неисправности, обслуживание и временный самостоятельный ремонт'),
            decision(sufficient=True, ids=[1, 2, 3, 4, 5, 6]),
            verification('Раму выправили, ножницы заточили. Чемодан не ремонтировал: ручку примотал временно.',
                         [1, 2, 3], excluded=[4, 5, 6])])
        final = json.loads(model.retrieval_inputs[-1][-1]['content'])
        self.assertEqual(set(final), {'task', 'question', 'question_time', 'criterion', 'evidence',
                                     'scope', 'scan_complete', 'limited'})
        self.assertEqual({n['id'] for n in final['evidence']}, {1, 2, 3, 4, 5, 6})
        self.assertEqual(phase['included_ids'], [1, 2, 3])
        self.assertNotIn('тарел', reply)
        self.assertNotIn('абажур', reply)
        self.assertIn('не ремонтировал', reply)

    async def test_rejected_sufficient_source_triggers_one_recovery(self):
        self.seed('Кофемашину забрал без ремонта.', 'В кофемашине мастер заменил насос.')
        _, phase, reply = await self.run_agent('Как починили кофемашину?', [
            scope(query='насос'), decision(sufficient=True, ids=[1]),
            verification('Выдуманный ремонт.', [], excluded=[1]),
            decision(sufficient=True, ids=[1, 2]),
            verification('Мастер заменил насос.', [2], excluded=[1])])
        self.assertEqual(phase['calls'], 5)
        self.assertTrue(phase['scan_complete'])
        self.assertEqual(phase['included_ids'], [2])
        self.assertEqual(set(phase['evidence_usage']), {'1_0', '2_0'})
        self.assertIn('заменил насос', reply)
        self.assertNotIn('Выдуманный', reply)

    async def test_recovery_does_not_reset_fragment_limit(self):
        from unittest.mock import patch
        from personal_assistent.retrieval import Collector
        self.seed('Станок забрал без ремонта.', 'У станка заменили двигатель.')
        with patch.object(Collector, 'MAX_FRAGMENTS', 1):
            _, phase, reply = await self.run_agent('Как починили станок?', [
                scope(query='двигатель'), decision(sufficient=True, ids=[1]),
                verification('', [], excluded=[1]), decision(sufficient=True, ids=[1, 2]),
                verification('', [], excluded=[1])])
        self.assertEqual(len(phase['evidence_usage']), 1)
        self.assertNotIn('read_2_0', phase['results'])
        self.assertTrue(phase['limited'])
        self.assertIn('Поиск неполный', reply)

    async def test_real_user_edit_uses_current_full_note_not_old_original(self):
        from personal_assistent.tools import ToolRunner
        self.seed('Для Нины код NX405.')
        self.store.enqueue(update('Исправь код Нины на NX450', 900), NOW)
        job = self.store.next_job()
        ToolRunner(self.store).run(call('update_note', {'id': 1, 'content': 'Для Нины код NX450.'}, 'edit'),
                                   job, NOW)
        # The same timestamp is intentional: authority must not rely on dates.
        self.store.finish(job['id'], 'Исправь код', 'Исправлено', NOW, presented=True)
        _, phase, reply = await self.run_agent('Какой код Нины?', [scope(),
            decision(sufficient=True, ids=[1]), verification('NX450.', [1])])
        self.assertEqual(phase['included_ids'], [1])
        self.assertIn('NX450', reply)
        self.assertNotIn('NX405', reply)

    async def test_legacy_read_fragments_seed_shared_budget_before_new_selection(self):
        from unittest.mock import patch
        from personal_assistent.retrieval import Collector
        from personal_assistent.tools import ToolRunner
        self.seed('Ранее прочитанный факт.', 'Другой факт.')
        self.store.enqueue(update('Другой факт?'), NOW)
        job = self.store.next_job()
        old = dict(id=1, content='Ранее прочитанный факт.', original='Ранее прочитанный факт.', content_offset=0)
        for saved in ('evidence', 'interrupted_read'):
            with self.subTest(saved=saved):
                phase = dict(id='legacy', results={}, read_cache={}, limited=False)
                if saved == 'evidence':
                    phase['evidence'] = [dict(id=1, fragments=[old])]
                else:
                    phase['results']['read_1_0'] = old
                state = dict(session=0, memory_collection=phase)
                collector = Collector(self.store, CollectingModel(), ToolRunner(self.store), Budget(), lambda: NOW)
                with patch.object(Collector, 'MAX_FRAGMENTS', 1):
                    self.assertEqual(collector.evidence([2], state, job), [])
                self.assertEqual(set(phase['evidence_usage']), {'1_0'})
                self.assertNotIn('read_2_0', phase['results'])
                self.assertTrue(phase['limited'])

    async def test_legacy_checkpoint_keeps_spent_calls_and_does_not_reuse_plan(self):
        self.seed('Фильтр купил.')
        self.store.enqueue(update('Фильтр?'), NOW)
        first = CollectingModel([answer(call('search_notes', {'query': 'фильтр'}, 'search')),
                                 answer(text='Черновик')], [RuntimeError('Interrupted')])
        with self.assertRaises(RuntimeError):
            await Agent(self.store, first, clock=lambda: NOW).process(self.store.next_job())
        job = dict(self.store.db.execute('SELECT * FROM jobs').fetchone())
        state = json.loads(job['checkpoint']); phase = state['memory_collection']
        phase.pop('protocol_version', None)
        phase['responses'] = {'plan': dict(scope='single', sufficient=True, query='', ids=[1])}
        phase['calls'] = 5
        self.store.checkpoint(job['id'], state)
        resumed = CollectingModel()
        await Agent(self.store, resumed, clock=lambda: NOW).process(self.store.next_job())
        self.assertEqual(resumed.retrieval_inputs, [])
        final = json.loads(self.store.db.execute('SELECT checkpoint FROM jobs').fetchone()[0])['memory_collection']
        self.assertEqual(final['calls'], 5)
        self.assertEqual(final['error_type'], 'CollectionLimit')

    async def test_single_without_defined_target_is_conservatively_scanned(self):
        self.seed('Маршрутизатор XR-501.')
        _, phase, _ = await self.run_agent('Какая модель устройства?', [
            scope('single', target='  '), decision(sufficient=True, ids=[1]),
            verification('XR-501.', [1])])
        self.assertTrue(phase['scan_complete'])

    async def test_invalid_reviews_quotes_and_answer_ids_are_rejected(self):
        def valid():
            return dict(answer='Подтверждено.', ids=[1], reviews=[dict(id=1, decision='include',
                quote='Исходный факт.', relation='Ответ на вопрос', status='completed')])
        invalid = []
        value = valid(); value['reviews'] = []; invalid.append(value)
        value = valid(); value['reviews'] *= 2; invalid.append(value)
        value = valid(); value['reviews'][0]['id'] = 999; invalid.append(value)
        value = valid(); value['reviews'][0]['quote'] = 'Выдуманная цитата'; invalid.append(value)
        value = valid(); value['reviews'][0]['quote'] = ''; invalid.append(value)
        value = valid(); value['reviews'][0]['relation'] = ' '; invalid.append(value)
        value = valid(); value['reviews'][0]['decision'] = 'exclude'; invalid.append(value)
        value = valid(); value['ids'] = []; invalid.append(value)
        value = valid(); value['ids'] = [1, 1]; invalid.append(value)
        value = valid(); value['reviews'][0]['status'] = 'successful'; invalid.append(value)
        for fields in invalid:
            with self.subTest(fields=fields):
                self.seed('Исходный факт.')
                _, phase, reply = await self.run_agent('Факт?', [scope(),
                    decision(sufficient=True, ids=[1]), decision(**fields)])
                self.assertIn('error_type', phase)
                self.assertIn('Поиск неполный', reply)
                self.assertNotIn('Подтверждено.', reply)
                self.store = Store(':memory:')
                self.addCleanup(self.store.db.close)

    async def test_stale_original_cannot_override_current_note(self):
        self.seed('Код объекта исправлен на BN94.')
        self.store.db.execute("UPDATE notes SET original='Код объекта BN49.'")
        _, phase, _ = await self.run_agent('Объект?', [scope(), decision(sufficient=True, ids=[1]),
            decision(answer='BN49.', ids=[1], reviews=[dict(id=1, decision='include',
                quote='Код объекта BN49.', relation='Состояние', status='completed')])])
        self.assertEqual(phase['error_type'], 'ValueError')

    async def test_no_included_facts_discards_answer_and_reports_uncertainty(self):
        self.seed('Зонт выбросил.')
        _, phase, reply = await self.run_agent('Зонт починил?', [scope('collection'),
            decision(sufficient=True, ids=[1]), verification('Зонт успешно починил.', [], uncertain=[1])])
        self.assertEqual(phase['included_ids'], [])
        self.assertNotIn('успешно', reply)
        self.assertIn('Связь части', reply)

    async def test_unknown_ids_and_write_calls_are_rejected(self):
        for decisions in ([scope(), decision(sufficient=True, ids=[999])],
                          [answer(call('create_note', {'content': 'Injected'}, 'bad'))]):
            with self.subTest(decisions=decisions):
                self.seed('Фильтр купил.')
                _, phase, reply = await self.run_agent('Фильтр?', decisions)
                self.assertIn('error_type', phase)
                self.assertIn('Поиск неполный', reply)
                self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
                self.store = Store(':memory:')
                self.addCleanup(self.store.db.close)

    async def test_scan_limit_is_reported_in_final_answer(self):
        self.seed(*(f'Предмет {i}' for i in range(90)))
        _, phase, reply = await self.run_agent('Какие предметы?', [
            scope('collection', query=''),
            decision(sufficient=True, ids=[90]),
            verification('Предмет 89.', [90])])
        self.assertEqual(phase['pages'], 16)
        self.assertFalse(phase['scan_complete'])
        self.assertIn('Поиск неполный', reply)

    async def test_resume_does_not_repeat_plan_or_search(self):
        self.seed('Свет в саду.', 'Электроэнергия — 500 рублей.')
        question = 'Сколько стоит свет?'
        self.store.enqueue(update(question), NOW)
        first = CollectingModel([answer(call('search_notes', {'query': question}, 'first')),
                                 answer(text='Черновик')], [
            scope('collection', query='электроэнергия'),
            RuntimeError('Interrupted')])
        with self.assertRaises(RuntimeError):
            await Agent(self.store, first, clock=lambda: NOW).process(self.store.next_job())
        searches = self.store.db.execute("SELECT count(*) FROM tool_results WHERE key LIKE '%alternative'").fetchone()[0]
        resumed = CollectingModel(decisions=[decision(sufficient=True, ids=[2]),
                                            verification('500 рублей.', [2])])
        await Agent(self.store, resumed, clock=lambda: NOW).process(self.store.next_job())
        self.assertEqual(resumed.inputs, [])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM tool_results WHERE key LIKE '%alternative'").fetchone()[0], searches)
        self.assertEqual(len(resumed.retrieval_inputs), 2)
        self.assertEqual(self.store.history()[-1]['content'], '500 рублей.')

    async def test_session_change_during_retrieval_cannot_publish_stale_answer(self):
        self.seed('Фильтр купил.')
        self.store.enqueue(update('Фильтр?'), NOW)
        store = self.store
        class Cancelled(CollectingModel):
            async def retrieve(self, messages):
                store.set_meta('session', 1)
                return scope('single', query='')
        model = Cancelled([answer(call('search_notes', {'query': 'фильтр'}, 's')),
                           answer(text='Черновик')])
        await Agent(store, model, clock=lambda: NOW).process(store.next_job())
        self.assertEqual(store.history(), [])
        self.assertEqual(store.db.execute('SELECT count(*) FROM outbox').fetchone()[0], 0)


class NarrowTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_request_has_no_general_tool_catalog(self):
        metrics = []
        def handler(request):
            body = json.loads(request.content)
            self.assertNotIn('tools', body)
            self.assertEqual(body['response_format'], {'type': 'json_object'})
            self.assertEqual(body['thinking'], {'type': 'disabled'})
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': decision(answer='Факт', ids=[1])}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await DeepSeek(client, 'fake', on_call=metrics.append).retrieve([{'role': 'user', 'content': 'synthetic'}])
        self.assertEqual(metrics[0]['purpose'], 'retrieval')
