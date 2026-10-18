"""Selective rule cancellation must preserve other event notifications."""
import unittest
from datetime import date
from unittest.mock import patch

from personal_assistent.events import Events
from personal_assistent.agent import Agent
from personal_assistent.context import Budget
from personal_assistent.receipts import confirmed_text
from personal_assistent.storage import dump
from test_assistant import answer, call, update
from test_context import SummaryModel
import test_events as fixtures


class DisableEventRuleTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    def birthday(self):
        event = self.execute('create_event', data={'kind': 'birthday', 'person': 'Оля',
            'start_date': '1995-10-12', 'birth_year': 1995})['result']
        weekly = self.execute('set_event_rule', id=event['id'], scope='future', from_position=0,
            rule={'kind': 'calendar', 'offset': 7})['result']['rule_id']
        day = self.execute('set_event_rule', id=event['id'], scope='future', from_position=0,
            rule={'kind': 'calendar', 'offset': 0})['result']['rule_id']
        with self.store.atomic():
            Events(self.store).generate(fixtures.NOW, event['id'], through=date(2028, 12, 31))
        return event['id'], weekly, day

    def notifications(self, rule_id):
        return [dict(r) for r in self.store.db.execute(
            'SELECT * FROM reminders WHERE event_rule_id=? ORDER BY id', (rule_id,))]

    def disable(self, event, rule_id, **scope):
        return self.execute('change_event', id=event, operation='disable_rule', rule_id=rule_id,
                            **({'scope': 'future', 'from_position': 0} | scope))

    def test_disable_weekly_preserves_event_same_day_and_replay(self):
        event, weekly, day = self.birthday()
        day_rule = dict(self.store.db.execute('SELECT * FROM event_reminder_rules WHERE id=?', (day,)).fetchone())
        day_notifications = self.notifications(day)
        segments = [tuple(r) for r in self.store.db.execute('SELECT * FROM event_segments')]
        result = self.disable(event, weekly)
        self.assertEqual(result['status'], 'done', result)
        self.assertEqual(self.notifications(day), day_notifications)
        self.assertEqual(dict(self.store.db.execute('SELECT * FROM event_reminder_rules WHERE id=?', (day,)).fetchone()), day_rule)
        self.assertEqual([tuple(r) for r in self.store.db.execute('SELECT * FROM event_segments')], segments)
        self.assertEqual(self.store.db.execute('SELECT deleted FROM events WHERE id=?', (event,)).fetchone()[0], 0)
        self.assertFalse(any(r['status'] in ('pending', 'queued') for r in self.notifications(weekly)))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM event_reminder_rules').fetchone()[0], 2)
        visible = Events(self.store).view(event, limit=1)
        self.assertEqual([r['id'] for r in visible['rules']], [day])
        self.assertEqual(visible['rules_total'], 1)
        self.assertIsNone(visible['next_rules_offset'])
        before = [tuple(r) for r in self.store.db.execute('SELECT * FROM reminders ORDER BY id')]
        with self.store.atomic():
            Events(self.store).generate(fixtures.NOW, event, through=date(2028, 12, 31))
        self.tool('execute_request', request_id=result['request_id'])
        self.assertEqual([tuple(r) for r in self.store.db.execute('SELECT * FROM reminders ORDER BY id')], before)
        self.assertIn('Правило напоминания отключено', confirmed_text(self.store, self.job['id']))

    def test_missing_rule_requires_selection_instead_of_disabling_all(self):
        event, weekly, day = self.birthday()
        before = self.notifications(weekly) + self.notifications(day)
        result = self.tool('prepare_request', action='change_event', parameters={
            'id': event, 'operation': 'disable_rule', 'scope': 'future', 'from_position': 0})
        self.assertEqual(result['status'], 'waiting', result)
        self.assertIn('rule_id', result['missing_fields'])
        self.assertEqual(self.notifications(weekly) + self.notifications(day), before)

    def test_one_occurrence_keeps_rule_for_following_year(self):
        event, weekly, day = self.birthday()
        future = self.store.db.execute("SELECT * FROM event_occurrences WHERE event_id=? AND starts_at>? ORDER BY position", (event, fixtures.NOW)).fetchall()
        self.assertGreaterEqual(len(future), 2)
        day_before = self.notifications(day)
        result = self.disable(event, weekly, scope='one', occurrence_id=future[0]['id'])
        self.assertEqual(result['status'], 'done', result)
        rows = self.notifications(weekly)
        self.assertTrue(all(r['status'] == 'cancelled' for r in rows if r['event_occurrence_id'] == future[0]['id']))
        self.assertTrue(any(r['status'] == 'pending' for r in rows if r['event_occurrence_id'] == future[1]['id']))
        self.assertEqual(self.notifications(day), day_before)

    def test_rule_from_another_event_is_rejected(self):
        event, weekly, day = self.birthday()
        other = self.execute('create_event', data={'content': 'Другая встреча',
            'start_date': '2026-11-01', 'start_time': '12:00'})['result']['id']
        before = self.notifications(weekly)
        result = self.disable(other, weekly)
        self.assertEqual(result['error']['code'], 'rule_not_found', result)
        self.assertEqual(self.notifications(weekly), before)

    def test_pending_snooze_cancelled_but_sent_history_kept(self):
        event, weekly, day = self.birthday()
        row = next(r for r in self.notifications(weekly) if r['status'] == 'pending')
        with self.store.db:
            self.store.db.execute("UPDATE reminders SET status='sent' WHERE id=?", (row['id'],))
        due = row['due_at']
        self.assertEqual(self.store.snooze(row['id'], due), 'Отложено на 10 минут.')
        day_before = self.notifications(day)
        result = self.disable(event, weekly)
        self.assertEqual(result['status'], 'done', result)
        rows = self.notifications(weekly)
        self.assertEqual(next(r['status'] for r in rows if r['id'] == row['id']), 'sent')
        self.assertTrue(any(r['is_snooze'] and r['status'] == 'cancelled' for r in rows))
        self.assertEqual(self.notifications(day), day_before)
        self.assertNotEqual(self.store.snooze(row['id'], due + 60), 'Отложено на 10 минут.')

    def grouped_notifications(self):
        event = self.execute('create_event', data={'content': 'Встреча',
            'start_date': '2026-10-20', 'start_time': '12:00'})['result']['id']
        first = self.execute('set_event_rule', id=event, scope='future', from_position=0,
            rule={'kind': 'calendar', 'offset': 0})['result']['rule_id']
        second = self.execute('set_event_rule', id=event, scope='future', from_position=0,
            rule={'kind': 'interval', 'offset': 10800})['result']['rule_id']
        due = fixtures.ts('2026-10-20T09:00')
        self.store.prepare_reminders(due, lambda r: r['content'])
        return event, first, second, due

    def test_unstarted_group_keeps_other_rule_for_delivery(self):
        event, first, second, due = self.grouped_notifications()
        surviving_id = self.notifications(second)[0]['id']
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notification_groups').fetchone()[0], 1)
        result = self.disable(event, first)
        self.assertEqual(result['status'], 'done', result)
        self.assertEqual(self.notifications(first)[0]['status'], 'cancelled')
        self.assertEqual(self.notifications(second)[0]['status'], 'pending')
        self.store.prepare_reminders(due, lambda r: r['content'])
        outbox = self.store.next_outbox(due)
        self.assertEqual(outbox['reminder_id'], surviving_id)
        self.assertEqual(self.notifications(second)[0]['status'], 'queued')

    def test_started_group_rejects_disable_without_partial_changes(self):
        event, first, second, due = self.grouped_notifications()
        outbox = self.store.next_outbox(due)
        self.store.start_delivery(outbox['key'])
        before = self.notifications(first) + self.notifications(second)
        rules = [tuple(r) for r in self.store.db.execute('SELECT * FROM event_reminder_rules')]
        revision = self.store.db.execute('SELECT revision FROM events WHERE id=?', (event,)).fetchone()[0]
        result = self.disable(event, first)
        self.assertEqual(result['error']['code'], 'delivery_in_progress', result)
        self.assertEqual(self.notifications(first) + self.notifications(second), before)
        self.assertEqual([tuple(r) for r in self.store.db.execute('SELECT * FROM event_reminder_rules')], rules)
        self.assertEqual(self.store.db.execute('SELECT revision FROM events WHERE id=?', (event,)).fetchone()[0], revision)

    def test_receipt_failure_rolls_back_rule_and_notifications(self):
        event, weekly, day = self.birthday()
        before = self.notifications(weekly) + self.notifications(day)
        rule = dict(self.store.db.execute('SELECT * FROM event_reminder_rules WHERE id=?', (weekly,)).fetchone())
        draft = self.tool('prepare_request', action='change_event', parameters={'id': event,
            'operation': 'disable_rule', 'rule_id': weekly, 'scope': 'future', 'from_position': 0})
        with patch.object(self.runner.requests, '_receipt', side_effect=RuntimeError('receipt unavailable')):
            with self.assertRaises(RuntimeError):
                self.tool('execute_request', request_id=draft['request_id'])
        self.assertEqual(self.notifications(weekly) + self.notifications(day), before)
        self.assertEqual(dict(self.store.db.execute('SELECT * FROM event_reminder_rules WHERE id=?', (weekly,)).fetchone()), rule)

    def test_rule_id_is_not_accepted_for_disabling_all_notifications(self):
        event, weekly, day = self.birthday()
        before = self.notifications(weekly) + self.notifications(day)
        result = self.tool('prepare_request', action='change_event', parameters={'id': event,
            'operation': 'disable', 'rule_id': weekly, 'scope': 'future', 'from_position': 0})
        self.assertIn('error', result)
        self.assertEqual(self.notifications(weekly) + self.notifications(day), before)


class DisableRuleAgentTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    async def test_live_sequence_can_finish_selective_disable_with_default_budget(self):
        event = self.execute('create_event', data={'kind': 'birthday', 'person': 'Оля',
            'start_date': '1995-10-12', 'birth_year': 1995})['result']['id']
        first = self.execute('set_event_rule', id=event, scope='future', from_position=0,
            rule={'kind': 'calendar', 'offset': 3})['result']['rule_id']
        day = self.execute('set_event_rule', id=event, scope='future', from_position=0,
            rule={'kind': 'calendar', 'offset': 0})['result']['rule_id']
        weekly = self.execute('set_event_rule', id=event, scope='future', from_position=0,
            replace_rule_id=first, rule={'kind': 'calendar', 'offset': 7})['result']['rule_id']
        with self.store.db:
            for text in ('У Оли день рождения 12 октября 1995 года, запомни',
                         'Напоминай за 3 дня каждый год', 'Ещё напоминай в сам день рождения',
                         'Вместо напоминания за 3 дня напоминай за неделю'):
                self.store.db.execute('INSERT INTO history(role,content,source_time) VALUES(?,?,?)',
                                      ('user', text, fixtures.NOW))
                self.store.db.execute('INSERT INTO history(role,content,source_time) VALUES(?,?,?)',
                                      ('assistant', 'Событие и выбранные правила сохранены.', fixtures.NOW))
        summary = 'Сохранён день рождения Оли: 12 октября 1995 года, ежегодно. Действуют два правила: за неделю и в сам день, общее время 09:00.'
        model = SummaryModel(
            answer(call('search_events', {'query': 'Оля'}, 'call_0123456789abcdef01234567_find')),
            answer(call('get_event', {'id': event}, 'call_0123456789abcdef01234567_read')),
            answer(call('prepare_request', {'action': 'change_event', 'parameters': {
                'id': event, 'operation': 'disable_rule', 'rule_id': weekly,
                'scope': 'future', 'from_position': 0}}, 'call_0123456789abcdef01234567_draft')),
            answer(call('execute_request', {'request_id': 5}, 'call_0123456789abcdef01234567_saved')),
            answer(text=dump({'kind': 'actions', 'tool_call_ids': ['call_0123456789abcdef01234567_saved']})),
            summaries=[summary] * 3)
        self.store.enqueue(update('Убери напоминание за неделю, оставь только в день рождения', number=50), fixtures.NOW)
        budget = Budget()
        await Agent(self.store, model, clock=lambda: fixtures.NOW, budget=budget).process(self.store.next_job())
        self.assertEqual(len(model.inputs), 5)
        self.assertTrue(all(budget.fits(m) for m in model.inputs))
        reply = self.store.next_outbox(fixtures.NOW)['text']
        self.assertIn('Правило напоминания отключено', reply)
        self.assertNotIn('бюджет', reply)
        upcoming = self.store.db.execute("SELECT event_rule_id FROM reminders WHERE status='pending'").fetchall()
        self.assertTrue(upcoming)
        self.assertTrue(all(r[0] == day for r in upcoming))
        self.assertIsNone(self.store.next_job())
