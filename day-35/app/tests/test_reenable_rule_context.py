"""Two completed rule actions must fit without losing their durable results."""
import json
import unittest

from personal_assistent.agent import Agent
from personal_assistent.context import Budget, Context
from personal_assistent.events import Events
from personal_assistent.storage import dump
from test_assistant import answer, call, update
from test_context import SummaryModel
import test_events as fixtures


class ReenableRuleContextTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    async def test_two_rule_actions_fit_default_budget_and_preserve_full_snapshots(self):
        event = self.execute('create_event', data={'kind': 'birthday', 'person': 'Оля',
            'start_date': '1995-10-12', 'birth_year': 1995})['result']['id']
        for offset in (3, 0):
            self.execute('set_event_rule', id=event, scope='future', from_position=0,
                         rule={'kind': 'calendar', 'offset': offset})
        self.execute('change_event', id=event, operation='disable', scope='future', from_position=0)
        self.store.set_meta('profile', {'notification_time': '11:30'})
        with self.store.db:
            for text, reply in (
                ('У Оли день рождения 12 октября 1995 года, запомни',
                 'День рождения сохранён: "Оля". Дата рождения: 12.10.1995. Повтор: ежегодно.'),
                ('Напоминай за 3 дня каждый год', 'Правило напоминания сохранено: "День рождения: Оля".'),
                ('Ещё напоминай в сам день рождения', 'Правило напоминания сохранено: "День рождения: Оля".'),
                ('Больше не напоминай о дне рождения Оли', 'Уведомления отключены: "День рождения: Оля".')):
                for role, content in (('user', text), ('assistant', reply)):
                    self.store.db.execute('INSERT INTO history(role,content,source_time) VALUES(?,?,?)',
                                          (role, content, fixtures.NOW))
        first = 'call_0123456789abcdef01234567_saved_first'
        second = 'call_0123456789abcdef01234567_saved_second'
        model = SummaryModel(
            answer(call('search_events', {'query': 'Оля'}, 'call_0123456789abcdef01234567_find')),
            answer(call('get_event', {'id': event}, 'call_0123456789abcdef01234567_read')),
            answer(call('prepare_request', {'action': 'set_event_rule', 'parameters': {
                'id': event, 'scope': 'future', 'from_position': 0,
                'rule': {'kind': 'calendar', 'offset': 3}}}, 'call_0123456789abcdef01234567_draft_first')),
            answer(call('execute_request', {'request_id': 5}, first)),
            answer(call('prepare_request', {'action': 'set_event_rule', 'parameters': {
                'id': event, 'scope': 'future', 'from_position': 0,
                'rule': {'kind': 'calendar', 'offset': 0}}}, 'call_0123456789abcdef01234567_draft_second')),
            answer(call('execute_request', {'request_id': 6}, second)),
            answer(text=dump({'kind': 'actions', 'tool_call_ids': [first, second]})),
            summaries=['Сохранён день рождения Оли, 12 октября 1995 года. Оба напоминания были отключены.'] * 3)
        self.store.enqueue(update('Снова напоминай за 3 дня и в сам день рождения Оли', number=50), fixtures.NOW)
        job = self.store.next_job()
        budget = Budget()
        agent = Agent(self.store, model, clock=lambda: fixtures.NOW, budget=budget)
        await agent.process(job)
        self.assertEqual(len(model.inputs), 7)
        self.assertTrue(all(budget.fits(m) for m in model.inputs))
        reply = self.store.next_outbox(fixtures.NOW)['text']
        self.assertNotIn('бюджет', reply)
        self.assertEqual(reply.count('Правило напоминания сохранено'), 2)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM events').fetchone()[0], 1)
        rules = Events(self.store).view(event)['rules']
        self.assertEqual(sorted(r['offset'] for r in rules), [0, 3])
        self.assertTrue(all(r['time_mode'] == 'default' and r['due_time'] is None for r in rules))
        # Storage and replay must retain full action snapshots, including rules/segments.
        saved = {r['id']: json.loads(r['result']) for r in self.store.db.execute('SELECT id,result FROM requests WHERE id IN (5,6)')}
        self.assertEqual(len(saved[5]['event']['rules']), 1)
        self.assertEqual(len(saved[6]['event']['rules']), 2)
        checkpoint = json.loads(self.store.db.execute('SELECT checkpoint FROM jobs WHERE id=50').fetchone()[0])
        tools = {m['tool_call_id']: json.loads(m['content']) for m in checkpoint['messages'] if m['role'] == 'tool'}
        self.assertEqual(tools[first]['result'], saved[5])
        self.assertEqual(tools[second]['result'], saved[6])
        api_tools = {m['tool_call_id']: json.loads(m['content']) for m in model.inputs[-1] if m['role'] == 'tool'}
        for identity in (first, second):
            self.assertEqual(api_tools[identity]['request_id'], tools[identity]['request_id'])
            self.assertEqual(api_tools[identity]['status'], 'done')
            self.assertEqual(api_tools[identity]['result']['rule_id'], tools[identity]['result']['rule_id'])
            self.assertEqual(api_tools[identity]['result']['warnings'], tools[identity]['result']['warnings'])
            self.assertEqual(api_tools[identity]['result']['event']['id'], event)
            self.assertEqual(api_tools[identity]['result']['event_details_tool'], 'get_event')
        read = 'call_0123456789abcdef01234567_read'
        self.assertEqual(api_tools[read], tools[read])
        cached = [json.loads(r[0]) for r in self.store.db.execute('SELECT result FROM tool_results')]
        for identity in (5, 6):
            result = next(r for r in cached if r.get('request_id') == identity and r.get('status') == 'done')
            self.assertEqual(result['result'], saved[identity])
        self.assertTrue(all(json.loads(r['payload'])['event']['segments'] for r in self.store.db.execute(
            "SELECT payload FROM action_receipts WHERE job_id=50 AND outcome='done'")))
        # Full checkpoints from earlier versions must fit too; preparing them
        # must not rewrite their complete tool results or call the model again.
        resumed = {'messages': checkpoint['messages'][:-1]}
        original = dump(resumed)
        prepared = await Context(self.store, SummaryModel(), budget).prepare(resumed)
        self.assertTrue(budget.fits(prepared))
        self.assertEqual(dump(resumed), original)
        before = [tuple(r) for r in self.store.db.execute('SELECT * FROM reminders ORDER BY id')]
        self.store.prepare_reminders(fixtures.NOW, lambda r: r['content'])
        await agent.process(job)
        self.assertEqual(len(model.inputs), 7)
        self.assertEqual([tuple(r) for r in self.store.db.execute('SELECT * FROM reminders ORDER BY id')], before)
        self.assertIsNone(self.store.next_job())

    def test_projection_keeps_warnings_and_never_changes_failures_or_reads(self):
        event = {'id': 1, 'kind': 'birthday', 'content': 'День рождения Оли',
                 'rules': [{'id': 2}], 'segments': [{'start_date': '2026-10-12'}]}
        success = {'request_id': 5, 'status': 'done', 'result': {
            'event': event, 'rule_id': 2, 'warnings': ['Срок уже прошёл; уведомление пропущено.']}}
        failure = {'request_id': 6, 'status': 'failed', 'result': {
            'event': event, 'rule_id': 2, 'error': {'code': 'delivery_in_progress'}}}
        messages = [answer(call('execute_request', {'request_id': 5}, 'success'),
                           call('execute_request', {'request_id': 6}, 'failure'),
                           call('get_event', {'id': 1}, 'read')),
                    {'role': 'tool', 'tool_call_id': 'success', 'content': dump(success)},
                    {'role': 'tool', 'tool_call_id': 'failure', 'content': dump(failure)},
                    {'role': 'tool', 'tool_call_id': 'read', 'content': dump(event)},
                    {'role': 'assistant', 'content': 'Готово', 'tool_calls': None}]
        original = dump(messages)
        projected = Context.model_messages(messages)
        self.assertEqual(json.loads(projected[1]['content'])['result']['warnings'], success['result']['warnings'])
        self.assertEqual(projected[2:], messages[2:])
        self.assertEqual(dump(messages), original)
