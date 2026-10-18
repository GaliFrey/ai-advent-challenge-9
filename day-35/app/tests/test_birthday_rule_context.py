"""Live-reported budget regression when adding a rule to an existing birthday."""
import unittest
from unittest.mock import patch

from pydantic import BaseModel, Field

from personal_assistent.agent import Agent
from personal_assistent.context import Budget
from personal_assistent.storage import Store, dump
from personal_assistent.tools import definitions, SPECS
from test_assistant import NOW, update, answer, call
from test_context import SummaryModel


class BirthdayRuleContextTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_schema_keeps_title_fields_constraints_and_default_data(self):
        class TitledInput(BaseModel):
            title: str = Field(min_length=1, max_length=40)
            metadata: dict = {'title': 'User data', 'nested': {'title': 'Keep this'}}

        with patch.dict(SPECS, {'test_titled_input': (TitledInput, 'Test tool contract')}):
            definition = next(t['function'] for t in definitions() if t['function']['name'] == 'test_titled_input')
        schema = definition['parameters']
        self.assertIn('title', schema['required'])
        self.assertEqual(schema['properties']['title']['minLength'], 1)
        self.assertEqual(schema['properties']['title']['maxLength'], 40)
        self.assertEqual(schema['properties']['metadata']['default'],
                         {'title': 'User data', 'nested': {'title': 'Keep this'}})

    async def test_search_read_prepare_execute_fits_default_budget(self):
        store = Store(':memory:')
        self.addCleanup(store.db.close)
        store.enqueue(update('У Оли день рождения 12 октября 1995 года, запомни'), NOW)
        model = SummaryModel(
            answer(call('prepare_request', {'action': 'create_event', 'parameters': {'data': {
                'kind': 'birthday', 'person': 'Оля', 'start_date': '1995-10-12',
                'birth_year': 1995}}}, 'call_0123456789abcdef01234567_birthday_draft')),
            answer(call('execute_request', {'request_id': 1}, 'call_0123456789abcdef01234567_birthday_saved')),
            answer(text=dump({'kind': 'actions', 'tool_call_ids': ['call_0123456789abcdef01234567_birthday_saved']})),
            answer(call('search_events', {'query': 'Оля'}, 'call_0123456789abcdef01234567_find')),
            answer(call('get_event', {'id': 1}, 'call_0123456789abcdef01234567_read')),
            answer(call('prepare_request', {'action': 'set_event_rule', 'parameters': {
                'id': 1, 'scope': 'future', 'from_position': 0,
                'rule': {'kind': 'calendar', 'offset': 3}}}, 'call_0123456789abcdef01234567_draft')),
            answer(call('execute_request', {'request_id': 2}, 'call_0123456789abcdef01234567_saved')),
            answer(text=dump({'kind': 'actions', 'tool_call_ids': ['call_0123456789abcdef01234567_saved']})),
            summaries=['Пользователь попросил сохранить день рождения Оли: 12 октября 1995 года. '
                       'Ассистент сохранил одно ежегодное событие на весь день. '
                       'Отдельных напоминаний при создании пользователь не просил.'])
        budget = Budget()
        agent = Agent(store, model, clock=lambda: NOW, budget=budget)
        await agent.process(store.next_job())
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(store.db.execute('SELECT count(*) FROM events').fetchone()[0], 1)
        self.assertEqual(store.db.execute('SELECT count(*) FROM event_reminder_rules').fetchone()[0], 0)
        self.assertEqual(store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
        store.enqueue(update('Напоминай за 3 дня каждый год', number=2), NOW)
        job = store.next_job()
        await agent.process(job)
        self.assertEqual(len(model.inputs), 8)
        self.assertTrue(all(budget.fits(messages) for messages in model.inputs))
        reply = '\n'.join(r[0] for r in store.db.execute('SELECT text FROM outbox ORDER BY rowid'))
        self.assertNotIn('бюджет', reply)
        self.assertNotIn('?', reply)
        self.assertIn('Правило напоминания сохранено', reply)
        self.assertEqual(store.db.execute('SELECT count(*) FROM events').fetchone()[0], 1)
        rules = store.db.execute('SELECT * FROM event_reminder_rules').fetchall()
        self.assertEqual(len(rules), 1)
        self.assertEqual((rules[0]['event_id'], rules[0]['offset'], rules[0]['time_mode']),
                         (1, 3, 'default'))
        self.assertIsNone(store.next_job())
        await agent.process(job)
        self.assertEqual(len(model.inputs), 8)
        self.assertEqual(store.db.execute('SELECT count(*) FROM event_reminder_rules').fetchone()[0], 1)
