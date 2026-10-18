import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import os

import httpx
from personal_assistent.agent import Agent, DeepSeek, SYSTEM
from personal_assistent.context import Budget, Context, ContextFull
from personal_assistent.config import Config
from personal_assistent.runtime import Runtime
from personal_assistent.storage import Store, dump
from personal_assistent.tools import ToolRunner, local_time, definitions
from test_assistant import update, call, answer, Model, NOW, OWNER


class SummaryModel(Model):
    def __init__(self, *responses, summaries=()):
        super().__init__(*responses)
        self.summaries = list(summaries)
        self.summary_inputs = []

    async def summarize(self, messages):
        self.summary_inputs.append(messages)
        response = self.summaries.pop(0)
        if isinstance(response, BaseException):
            raise response
        return answer(text=response)


def seed(store, pairs=10):
    with store.db:
        for i in range(pairs):
            user = 'Решили выбрать SQLite' if i == 0 else ('Изменение: выбираем PostgreSQL' if i == 5 else f'Реплика {i}')
            store.db.execute('INSERT INTO history(role,content) VALUES(?,?)', ('user', user))
            store.db.execute('INSERT INTO history(role,content) VALUES(?,?)', ('assistant', 'Обсудили'))


def state(store, text='Продолжим'):
    return {'context_version': 1, 'session': store.get_meta('session', 0),
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]}


class ContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store(':memory:')

    def tearDown(self):
        self.store.db.close()

    async def test_budget_includes_tools_output_reserve_and_utf8_margin(self):
        budget = Budget()
        messages = state(self.store)['messages']
        self.assertGreater(budget.estimate(messages), budget.estimate(messages, tools=False))
        self.assertGreater(budget.estimate(messages), len(dump({'messages': messages, 'tools': definitions()}).encode()))
        exact = Budget(budget.estimate(messages) + 2000, 2000)
        self.assertTrue(exact.fits(messages))
        self.assertFalse(Budget(exact.total - 1, 2000).fits(messages))
        self.assertFalse(Budget(exact.total, 2001).fits(messages))

    async def test_oversized_current_request_does_not_call_any_model_or_tools(self):
        seed(self.store)
        self.store.enqueue(update('x' * 100000), NOW)
        model = SummaryModel(summaries=['Не должно вызываться'])
        await Agent(self.store, model, clock=lambda: NOW).process(self.store.next_job())
        self.assertEqual(model.inputs, [])
        self.assertEqual(model.summary_inputs, [])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM history').fetchone()[0], 20)
        self.assertIn('бюджет', self.store.next_outbox(NOW)['text'])
        self.assertIsNone(self.store.next_job())
        self.assertIn('Локальная оценка', self.store.stats_text())

    async def test_budget_after_large_result_preserves_action_and_replay(self):
        self.store.enqueue(update('Сохрани вымышленный факт'), NOW)
        job = self.store.next_job()
        model = Model(answer(call('create_note', {'content': 'x' * 12000})))
        # Enough for the request and tool definitions, not the resulting tool exchange.
        budget = Budget(60000, 2000)
        agent = Agent(self.store, model, clock=lambda: NOW, budget=budget)
        initial = state(self.store)['messages']
        # The actual time/pending system context adds some overhead.
        agent.context.budget = Budget(budget.estimate(agent.context.with_profile(initial)) + 2800, 2000)
        await agent.process(job)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
        self.assertIn('бюджет', self.store.next_outbox(NOW)['text'])
        await Agent(self.store, Model(), clock=lambda: NOW).process(job)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_short_history_is_summarized_when_it_blocks_a_small_task(self):
        seed(self.store, pairs=4)
        with self.store.db:
            self.store.db.execute('UPDATE history SET content=content || ?', ('x' * 900,))
        model = SummaryModel(summaries=['Ранее обсуждали выбор базы данных.'])
        context = Context(self.store, model, Budget())
        current = state(self.store, 'Запомни день рождения Оли')
        base = context.budget.estimate(context.with_profile(current['messages']))
        context.budget = Budget(base + 7500, 2000)
        self.assertFalse(context.budget.fits(context.compose(current)))
        messages = await context.prepare(current)
        self.assertTrue(context.budget.fits(messages))
        self.assertEqual(len(model.summary_inputs), 1)
        self.assertEqual(len(self.store.history()), 2)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM history').fetchone()[0], 8)

    async def test_static_context_above_watermark_does_not_compress_short_history(self):
        seed(self.store, pairs=5)
        model = SummaryModel()
        context = Context(self.store, model, Budget())
        current = state(self.store, 'покажи профиль основной')
        base = context.budget.estimate(context.with_profile(current['messages']))
        context.budget = Budget(base + 7000, 2000)
        self.assertGreater(base, context.budget.total * .75)
        messages = await context.prepare(current)
        await context.prepare(current)
        self.assertTrue(context.budget.fits(messages))
        self.assertEqual(model.summary_inputs, [])
        self.assertEqual(len(self.store.history_rows()), 10)

    async def test_legacy_overlong_summary_and_short_history_recover_small_request(self):
        seed(self.store, pairs=3)
        self.store.set_meta('summary', {'text': 'Сводка о прежней беседе. ' * 110, 'through': 0})
        model = SummaryModel(summaries=['Обсуждали выбор базы; текущие настройки проверять инструментами.'])
        context = Context(self.store, model, Budget())
        current = state(self.store, 'покажи профиль основной')
        base = context.budget.estimate(context.with_profile(current['messages']))
        context.budget = Budget(base + 7000, 2000)
        self.assertFalse(context.budget.fits(context.compose(current)))
        messages = await context.prepare(current)
        self.assertTrue(context.budget.fits(messages))
        self.assertEqual(len(model.summary_inputs), 1)
        self.assertEqual(self.store.get_meta('summary')['through'], 4)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM history').fetchone()[0], 6)

    async def test_existing_summary_can_shrink_after_archive_is_exhausted(self):
        seed(self.store, pairs=1)
        previous = {'text': 'Длинная предыдущая сводка. ' * 60, 'through': 2}
        self.store.set_meta('summary', previous)
        model = SummaryModel(summaries=['Сохранён день рождения Оли; действует напоминание в сам день.'])
        context = Context(self.store, model, Budget())
        current = state(self.store, 'Убери напоминание за неделю')
        base = context.budget.estimate(context.with_profile(current['messages']))
        context.budget = Budget(base + 3500, 2000)
        self.assertFalse(context.budget.fits(context.compose(current)))
        self.assertEqual(self.store.history_rows(), [])
        messages = await context.prepare(current)
        self.assertTrue(context.budget.fits(messages))
        self.assertEqual(len(model.summary_inputs), 1)
        request = json.loads(model.summary_inputs[0][1]['content'])
        self.assertEqual(request['archive'], [])
        self.assertEqual(request['previous_summary'], previous['text'])
        self.assertEqual(self.store.get_meta('summary')['through'], 2)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM history').fetchone()[0], 2)

    async def test_failed_summary_recompression_preserves_text_and_boundary(self):
        seed(self.store, pairs=1)
        previous = {'text': 'Длинная предыдущая сводка. ' * 60, 'through': 2}
        self.store.set_meta('summary', previous)
        for response in (RuntimeError('summary unavailable'), previous['text']):
            with self.subTest(response=type(response).__name__):
                model = SummaryModel(summaries=[response])
                context = Context(self.store, model, Budget())
                current = state(self.store, 'Убери напоминание за неделю')
                base = context.budget.estimate(context.with_profile(current['messages']))
                context.budget = Budget(base + 3500, 2000)
                with self.assertRaises(ContextFull):
                    await context.prepare(current)
                self.assertTrue(current['summary_failed'])
                self.assertEqual(self.store.get_meta('summary'), previous)
                self.assertEqual(len(model.summary_inputs), 1)

    async def test_recompression_enforces_smaller_requested_character_limit(self):
        seed(self.store, pairs=1)
        previous = {'text': 'x' * 3000, 'through': 2}
        self.store.set_meta('summary', previous)
        model = SummaryModel(summaries=['a' * 500])
        context = Context(self.store, model, Budget())
        current = state(self.store)
        base = context.budget.estimate(context.with_profile(current['messages']))
        context.budget = Budget(base + 3000, 2000)
        with self.assertRaises(ContextFull):
            await context.prepare(current)
        self.assertTrue(current['summary_failed'])
        self.assertEqual(self.store.get_meta('summary'), previous)
        self.assertEqual(len(model.summary_inputs), 1)

    async def test_birthday_can_finish_after_tools_with_short_history_at_default_budget(self):
        seed(self.store, pairs=4)
        with self.store.db:
            # Keep enough pressure to exercise compaction even with lean tool schemas.
            self.store.db.execute('UPDATE history SET content=content || ?', ('x' * 900,))
        self.store.enqueue(update('У Оли день рождения 12 октября 1995 года, запомни'), NOW)
        model = SummaryModel(
            answer(call('prepare_request', {'action': 'create_event', 'parameters': {'data': {
                'kind': 'birthday', 'person': 'Оля', 'start_date': '1995-10-12',
                'birth_year': 1995}}}, 'birthday_draft')),
            answer(call('execute_request', {'request_id': 1}, 'birthday_saved')),
            answer(text=dump({'kind': 'actions', 'tool_call_ids': ['birthday_saved']})),
            summaries=['Обсуждали выбор базы данных.'])
        await Agent(self.store, model, clock=lambda: NOW).process(self.store.next_job())
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(len(model.summary_inputs), 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM events').fetchone()[0], 1)
        reply = self.store.next_outbox(NOW)['text']
        self.assertIn('12.10.1995', reply)
        self.assertNotIn('бюджет', reply)
        self.assertIsNone(self.store.next_job())

    async def test_summary_contains_early_decision_and_change_recent_turns_raw(self):
        seed(self.store)
        model = SummaryModel(summaries=['Задача: база. Сначала SQLite, затем PostgreSQL. Открыт вопрос миграции.'])
        context = Context(self.store, model, Budget())
        messages = await context.prepare(state(self.store))
        archive = json.loads(model.summary_inputs[0][1]['content'])['archive']
        self.assertIn('SQLite', archive[0]['content'])
        self.assertIn('PostgreSQL', archive[-2]['content'])
        self.assertEqual(len(self.store.history()), 8)
        self.assertEqual(messages[-2]['content'], '[Время сообщения: неизвестно]\nОбсудили')
        self.assertIn('PostgreSQL', messages[1]['content'])
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM history').fetchone()[0], 20)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)

    async def test_failed_empty_oversize_summary_preserves_previous_boundary(self):
        for response in (RuntimeError('network'), '', ' ' * 4, 'Я' * 4000, 'a' * 1501, 'Я' * 1501):
            with self.subTest(response_type=type(response).__name__):
                self.store.db.execute('DELETE FROM history')
                self.store.set_meta('summary', {'text': 'Старая сводка', 'through': 0})
                seed(self.store)
                model = SummaryModel(summaries=[response])
                await Context(self.store, model, Budget()).prepare(state(self.store))
                self.assertEqual(self.store.get_meta('summary'), {'text': 'Старая сводка', 'through': 0})
                self.assertEqual(len(model.summary_inputs), 1)

    async def test_summary_failure_over_budget_refuses_without_work_request(self):
        seed(self.store, 16)
        with self.store.db:
            self.store.db.execute("UPDATE history SET content=content || ?", ('a' * 500,))
        model = SummaryModel(summaries=[RuntimeError('network')])
        base = Budget().estimate(Context(self.store, model, Budget()).with_profile(state(self.store)['messages']))
        budget = Budget(base + 6000, 2000)
        with self.assertRaises(ContextFull):
            await Context(self.store, model, budget).prepare(state(self.store))
        self.assertEqual(len(model.summary_inputs), 1)
        self.assertEqual(model.inputs, [])
        self.assertIsNone(self.store.get_meta('summary'))

    async def test_long_archive_is_processed_in_bounded_chunks_with_previous_summary(self):
        seed(self.store, 32)
        with self.store.db:
            self.store.db.execute('UPDATE history SET content=content || ?', ('x' * 1500,))
        model = SummaryModel(summaries=['Решение и изменения'] * 8)
        context = Context(self.store, model, Budget())
        base = Budget().estimate(context.with_profile(state(self.store)['messages']))
        context.budget = Budget(base + 20000, 2000)
        await context.prepare(state(self.store))
        self.assertGreater(len(model.summary_inputs), 1)
        for request in model.summary_inputs:
            self.assertTrue(context.budget.fits(request, tools=False))
        second = json.loads(model.summary_inputs[1][1]['content'])
        self.assertEqual(second['previous_summary'], 'Решение и изменения')
        all_ids = [r['id'] for req in model.summary_inputs for r in json.loads(req[1]['content'])['archive']]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    async def test_summary_uses_original_message_time_not_compression_time(self):
        self.store.enqueue(update('Вчера решили использовать SQLite'), NOW)
        self.store.finish(1, 'Вчера решили использовать SQLite', 'Принято', NOW + 3600)
        seed(self.store, 9)
        model = SummaryModel(summaries=['Решили использовать SQLite 15 октября 2026'])
        await Context(self.store, model, Budget()).prepare(state(self.store))
        archive = json.loads(model.summary_inputs[0][1]['content'])['archive']
        self.assertEqual(archive[0]['source_time'], local_time(NOW))
        self.assertEqual(archive[1]['source_time'], local_time(NOW + 3600))
        self.assertIsNone(archive[2]['source_time'])
        self.assertEqual(self.store.history()[0]['content'], 'Изменение: выбираем PostgreSQL')

    async def test_failed_summary_is_not_retried_after_restart_during_work_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'restart.db'
            first = Store(path)
            try:
                seed(first)
                first.enqueue(update('Продолжим'), NOW)
                model = SummaryModel(asyncio.CancelledError(), summaries=[RuntimeError('network')])
                with self.assertRaises(asyncio.CancelledError):
                    await Agent(first, model, clock=lambda: NOW).process(first.next_job())
                self.assertEqual(len(model.summary_inputs), 1)
            finally:
                first.db.close()
            second = Store(path)
            try:
                model = SummaryModel(answer(text='Продолжаем'), summaries=[RuntimeError('must not retry')])
                await Agent(second, model, clock=lambda: NOW).process(second.next_job())
                self.assertEqual(model.summary_inputs, [])
                self.assertEqual(second.history()[-1]['content'], 'Продолжаем')
            finally:
                second.db.close()

    async def test_page_marks_truncated_text_as_incomplete_even_for_one_record(self):
        self.store.enqueue(update(), NOW)
        runner = ToolRunner(self.store)
        job = self.store.next_job()
        runner.run(call('create_note', {'content': 'а' * 2000}), job, NOW)
        result = runner.run(call('list_notes', {}, 'list'), job, NOW)
        self.assertEqual(len(result['items']), 1)
        self.assertTrue(result['incomplete'])
        self.assertIsNone(result['next_offset'])
        self.assertEqual(result['items'][0]['next_content_offset'], 1000)

    async def test_json_escaped_text_page_makes_progress_and_can_be_read_fully(self):
        request = update('\x01' * 1000)
        self.store.enqueue(request, NOW)
        job = self.store.next_job()
        runner = ToolRunner(self.store)
        content = '\x01' * 1000
        runner.run(call('create_note', {'content': content}), job, NOW)
        page = runner.run(call('list_notes', {}, 'list'), job, NOW)
        self.assertEqual(len(page['items']), 1)
        self.assertLessEqual(len(dump(page['items']).encode('utf-8')), 12000)
        self.assertTrue(page['incomplete'])
        self.assertIsNone(page['next_offset'])
        chunks, offset = [], 0
        while True:
            part = runner.run(call('get_note', {'id': 1, 'content_offset': offset}, f'part{offset}'), job, NOW)
            self.assertLessEqual(len(dump(part).encode('utf-8')), 12000)
            chunks.append(part['content'])
            next_offset = part['next_content_offset']
            if next_offset is None:
                break
            self.assertGreater(next_offset, offset)
            offset = next_offset
        self.assertEqual(''.join(chunks), content)

    async def test_failed_summary_is_not_retried_on_next_tool_step(self):
        seed(self.store)
        model = SummaryModel(summaries=[RuntimeError('network')])
        context = Context(self.store, model, Budget())
        current = state(self.store)
        await context.prepare(current)
        await context.prepare(current)
        self.assertEqual(len(model.summary_inputs), 1)

    async def test_impossible_request_preserves_pending_and_pairs(self):
        self.store.enqueue(update('x' * 100000), NOW)
        pending = {'question': 'Когда?', 'origin': update()['message']}
        with self.store.db:
            self.store.set_meta('pending', pending)
        await Agent(self.store, Model(), clock=lambda: NOW).process(self.store.next_job())
        self.assertEqual(self.store.get_meta('pending'), pending)
        self.assertEqual(self.store.history(), [])

    async def test_summary_and_tool_checkpoint_survive_restart_without_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.db'
            first = Store(path)
            seed(first)
            await Context(first, SummaryModel(summaries=['Раннее решение; позднее изменение']), Budget()).prepare(state(first))
            summary = first.get_meta('summary')
            first.enqueue(update('Сохрани тест'), NOW)
            crashing = Model(answer(call('create_note', {'content': 'Тест'})), asyncio.CancelledError())
            with self.assertRaises(asyncio.CancelledError):
                await Agent(first, crashing, clock=lambda: NOW).process(first.next_job())
            first.db.close()
            second = Store(path)
            model = SummaryModel(answer(text='Сохранено'))
            await Agent(second, model, clock=lambda: NOW).process(second.next_job())
            self.assertEqual(second.get_meta('summary'), summary)
            self.assertEqual(model.summary_inputs, [])
            self.assertEqual(second.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
            self.assertEqual(second.db.execute('SELECT count(*) FROM tool_results').fetchone()[0], 1)
            second.db.close()

    async def test_large_tool_pages_are_valid_and_all_text_can_be_read(self):
        self.store.enqueue(update(), NOW)
        job = self.store.next_job()
        runner = ToolRunner(self.store)
        for i in range(12):
            runner.run(call('create_note', {'content': 'Я' * 12000}, str(i)), job, NOW)
        page = runner.run(call('list_notes', {'limit': 50}, 'page'), job, NOW)
        self.assertTrue(page['incomplete'])
        self.assertGreater(page['next_offset'], 0)
        self.assertLess(len(dump(page).encode()), 13000)
        texts = []
        offset = 0
        while True:
            result = runner.run(call('get_note', {'id': 1, 'content_offset': offset}, f'read{offset}'), job, NOW)
            self.assertEqual(result['id'], 1)
            texts.append(result['content'])
            offset = result['next_content_offset']
            if offset is None:
                break
        self.assertEqual(''.join(texts), 'Я' * 12000)

    async def test_truncated_summary_http_reply_is_failed_and_usage_is_separate(self):
        seed(self.store)
        self.store.set_meta('summary', {'text': 'Прежнее решение', 'through': 0})
        self.store.db.commit()
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            is_summary = 'tools' not in payload
            return httpx.Response(200, json={'choices': [{'finish_reason': 'length' if is_summary else 'stop',
                'message': answer(text='Неполный' if is_summary else 'Ответ')}],
                **({'usage': {'prompt_tokens': 12, 'completion_tokens': 5}} if is_summary else {})})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeek(client, 'fake', on_call=self.store.record_llm_call)
            await Context(self.store, model, Budget()).prepare(state(self.store))
            await model.complete(state(self.store)['messages'])
        self.assertEqual(self.store.get_meta('summary'), {'text': 'Прежнее решение', 'through': 0})
        rows = self.store.db.execute('SELECT * FROM llm_calls ORDER BY id').fetchall()
        self.assertEqual([(r['purpose'], r['status']) for r in rows], [('summary', 'failed'), ('work', 'ok')])
        self.assertIsNone(rows[1]['prompt_tokens'])
        stats = self.store.stats_text()
        self.assertIn('summary: 1 вызовов; вход 12', stats)
        self.assertIn('work: 1 вызовов; вход нет данных', stats)
        self.assertNotIn('tools', requests[0])
        self.assertEqual(requests[0]['max_tokens'], 2000)

    async def test_summary_output_reserve_is_checked_independently_of_work_reserve(self):
        seed(self.store)
        previous = {'text': 'x' * 50000, 'through': 0}
        self.store.set_meta('summary', previous)
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': answer(text='Сводка')}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeek(client, 'fake', max_tokens=128)
            request = Context.summary_request(previous, self.store.history_rows()[:2])
            estimate = Budget().estimate(request, tools=False)
            context = Context(self.store, model, Budget(estimate + 1500, 1000))
            self.assertTrue(context.budget.fits(request, tools=False))
            self.assertFalse(context.summary_budget().fits(request, tools=False))
            current = state(self.store)
            with self.assertRaises(ContextFull):
                await context.prepare(current)
        self.assertEqual(requests, [])
        self.assertEqual(context.summary_budget().reserve, 2000)


class NewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.now = NOW
        self.number = 0
        seed(self.store, 2)
        with self.store.db:
            self.store.set_meta('pending', {'question': 'Когда?', 'origin': update()['message']})
            self.store.set_meta('summary', {'text': 'Обсуждение', 'through': 2})
        self.model = Model(answer(text='Обычный ответ'))
        self.runtime = Runtime(self.store, None, Agent(self.store, self.model, clock=lambda: self.now), OWNER, lambda: self.now)

    def tearDown(self):
        self.store.db.close()

    async def send(self, text):
        self.number += 1
        self.store.enqueue(update(text, self.number), self.now)
        await self.runtime.process_once()

    async def test_new_immediately_resets_context_without_model_call(self):
        await self.send('/new')
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertIsNone(self.store.get_meta('pending'))
        self.assertEqual(self.store.get_meta('summary'), {})
        self.assertEqual(self.store.history(), [])
        self.assertEqual(self.model.inputs, [])
        self.assertIn('Новая беседа начата', self.store.next_outbox(NOW)['text'])
        await self.send('/new')
        self.assertEqual(self.store.get_meta('session'), 2)
        await self.send('Новая тема')
        model_input = dump(self.model.inputs[0])
        self.assertNotIn('Обсуждение', model_input)
        self.assertNotIn('Решили выбрать SQLite', model_input)
        self.assertNotIn('Когда?', model_input)

    async def test_legacy_arguments_do_not_reset_or_cancel_new_session(self):
        await self.send('/new')
        await self.send('/stats')
        await self.send('Новая тема')
        for command in ('/new confirm', '/new cancel', '/new wrong'):
            await self.send(command)
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertEqual(len(self.store.history()), 2)
        self.assertEqual(len(self.model.inputs), 1)

    async def test_new_resets_only_context_and_deduplicates(self):
        self.store.enqueue(update('Служебная запись', 100), NOW)
        tooljob = dict(self.store.db.execute('SELECT * FROM jobs WHERE id=100').fetchone())
        runner = ToolRunner(self.store)
        runner.run(call('create_note', {'content': 'Факт'}, 'note'), tooljob, NOW)
        r = runner.run(call('create_reminder', {'content': 'Напомнить', 'due_at': local_time(NOW+100)}, 'reminder'), tooljob, NOW)
        self.store.prepare_reminders(NOW+100, lambda r: 'Уведомление')
        self.store.sent(self.store.next_outbox(NOW+100)['key'], 1, NOW+100)
        runner.run(call('create_reminder', {'content': 'Активное', 'due_at': local_time(NOW+900)}, 'active'), tooljob, NOW)
        runner.run(call('create_reminder', {'content': 'В очереди', 'due_at': local_time(NOW+150)}, 'queued'), tooljob, NOW)
        self.store.prepare_reminders(NOW+160, lambda r: 'В очереди уведомлений')
        notifications = [tuple(row) for row in self.store.db.execute('SELECT * FROM outbox WHERE reminder_id IS NOT NULL')]
        with self.store.db:
            self.store.queue_message('other', 'Сохранить очередь', now=NOW)
            self.store.set_meta('setting', 42)
            self.store.record_llm_call({'created_at': NOW, 'duration': 1, 'status': 'ok', 'error_type': None,
                'finish_reason': 'stop', 'prompt_tokens': None, 'completion_tokens': None, 'cached_tokens': None})
            self.store.db.execute("UPDATE jobs SET status='done' WHERE id=100")
        tables = ('notes', 'reminders', 'tool_results', 'llm_calls')
        before = {t: [tuple(r) for r in self.store.db.execute('SELECT * FROM ' + t)] for t in tables}
        await self.send('/new')
        self.assertEqual(self.store.history(), [])
        self.assertEqual(self.store.get_meta('summary'), {})
        self.assertIsNone(self.store.get_meta('pending'))
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM history').fetchone()[0], 4)
        self.assertEqual(self.store.get_meta('setting'), 42)
        self.assertEqual(notifications, [tuple(row) for row in self.store.db.execute('SELECT * FROM outbox WHERE reminder_id IS NOT NULL')])
        self.assertEqual(before, {t: [tuple(r) for r in self.store.db.execute('SELECT * FROM ' + t)] for t in tables})
        self.assertTrue(self.store.db.execute("SELECT 1 FROM outbox WHERE key LIKE 'other:%'").fetchone())
        self.assertIn('Отложено', self.store.snooze(r['id'], NOW+200))
        await self.send('Новая тема')
        self.store.enqueue(update('/new', self.number-1), NOW)
        self.assertFalse(await self.runtime.process_once())
        await self.send('/new confirm')
        self.assertEqual(self.store.get_meta('session'), 1)
        self.assertEqual(len(self.store.history()), 2)

    async def test_session_persists_restart_and_unauthorized_new_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bot.db'
            store = Store(path)
            store.set_meta('new_confirmation', {'session': 0, 'owner': OWNER, 'expires': NOW+300})
            store.enqueue(update('/new'), NOW)
            await Runtime(store, None, None, OWNER, lambda: NOW).process_once()
            store.db.close()
            store = Store(path)
            try:
                self.assertEqual(store.get_meta('session'), 1)
                self.assertIsNone(store.get_meta('new_confirmation'))
                unauthorized = update('/new', 2)
                unauthorized['message']['from']['id'] = OWNER+1
                store.enqueue(unauthorized, NOW)
                await Runtime(store, None, None, OWNER, lambda: NOW).process_once()
                self.assertEqual(store.get_meta('session'), 1)
                self.assertEqual(store.db.execute('SELECT status FROM jobs WHERE id=2').fetchone()[0], 'ignored')
            finally:
                store.db.close()

    async def test_old_checkpoint_is_never_executed_after_new_session_and_queue_order(self):
        # Queue a reset, a new message and an old checkpoint before processing.
        self.store.enqueue(update('/new', 2), NOW)
        self.store.enqueue(update('Новая тема', 3), NOW)
        self.store.enqueue(update('Старая задача', 4), NOW)
        old = state(self.store)
        old['rounds'] = 1
        old['messages'].append(answer(call('create_note', {'content': 'Не создавать'})))
        self.store.checkpoint(4, old)
        await self.runtime.process_once()
        await self.runtime.process_once()
        self.assertEqual(len(self.store.history()), 2)
        self.assertFalse(await self.runtime.process_once())
        job = dict(self.store.db.execute('SELECT * FROM jobs WHERE id=4').fetchone())
        await Agent(self.store, Model()).process(job)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
        self.assertEqual(self.store.get_meta('session'), 1)


class MigrationTests(unittest.TestCase):
    def test_old_database_upgrade_keeps_archive_pending_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bot.db'
            db = sqlite3.connect(path)
            db.executescript('''
                CREATE TABLE jobs(id INTEGER PRIMARY KEY,payload TEXT NOT NULL,received_at REAL NOT NULL,status TEXT NOT NULL DEFAULT 'pending',checkpoint TEXT);
                CREATE TABLE llm_calls(id INTEGER PRIMARY KEY,created_at REAL NOT NULL,duration REAL NOT NULL,status TEXT NOT NULL,error_type TEXT,finish_reason TEXT,prompt_tokens INTEGER,completion_tokens INTEGER,cached_tokens INTEGER);
                INSERT INTO llm_calls VALUES(1,100,1,'ok',NULL,'stop',12,2,NULL);
                CREATE TABLE history(id INTEGER PRIMARY KEY,role TEXT NOT NULL,content TEXT NOT NULL);
                INSERT INTO history VALUES(1,'user','Старый вопрос'),(2,'assistant','Старый ответ');
                CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                INSERT INTO meta VALUES('pending','{"question":"Когда?"}');
            ''')
            db.close()
            store = Store(path)
            self.assertEqual(store.history()[0]['content'], 'Старый вопрос')
            self.assertEqual(store.get_meta('pending'), {'question': 'Когда?'})
            self.assertEqual(store.db.execute('SELECT purpose FROM llm_calls').fetchone()[0], 'work')
            self.assertIn('Входные токены: 12.', store.stats_text())
            store.db.close()

    def test_context_settings_reject_invalid_budget_and_reserve(self):
        base = {'TELEGRAM_BOT_TOKEN': 'fake', 'TELEGRAM_OWNER_ID': '42', 'DEEPSEEK_API_KEY': 'fake'}
        for settings in ({'ASSISTANT_CONTEXT_BUDGET': '2000'}, {'ASSISTANT_CONTEXT_RESERVE': '1999'},
                         {'ASSISTANT_CONTEXT_BUDGET': 'nan'}, {'ASSISTANT_CONTEXT_BUDGET': '1000001'}):
            with self.subTest(settings=settings), patch('personal_assistent.config.load_dotenv'), patch.dict(os.environ, {**base, **settings}, clear=True):
                with self.assertRaises(ValueError):
                    Config.load()

    def test_small_generation_limit_keeps_existing_configuration_usable(self):
        base = {'TELEGRAM_BOT_TOKEN': 'fake', 'TELEGRAM_OWNER_ID': '42',
                'DEEPSEEK_API_KEY': 'fake', 'DEEPSEEK_MAX_TOKENS': '128'}
        with patch('personal_assistent.config.load_dotenv'), patch.dict(os.environ, base, clear=True):
            config = Config.load()
        self.assertEqual(config.max_tokens, 128)
        self.assertEqual(config.context_reserve, 1000)
