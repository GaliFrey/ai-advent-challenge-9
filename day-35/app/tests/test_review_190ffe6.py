"""Requirement regressions found by independent review of 190ffe6."""
import json
import unittest
from datetime import date

from personal_assistent.events import EventData
import test_events as fixtures
from test_events import NOW, ts


class BirthdayCorrectionTests(unittest.TestCase):
    setUp = fixtures.EventTests.setUp
    create = fixtures.EventTests.create
    change = fixtures.EventTests.change
    occs = fixtures.EventTests.occs

    def test_birth_year_in_correction_does_not_create_decades_of_history(self):
        event = self.create(kind='birthday', person='Оля', start_time=None,
                            start_date='1990-10-20', birth_year=1990)
        first = self.occs(event)[0]['id']
        self.change(event, scope='future', data={
            'content': 'День рождения', 'kind': 'birthday', 'person': 'Оля',
            'start_date': '1990-11-05', 'birth_year': 1990})
        self.assertEqual(self.occs(event)[0]['id'], first)
        self.assertEqual([o['start_date'] for o in self.occs(event)], ['2026-11-05', '2027-11-05'])
        self.assertEqual(self.e.event(event['id'])['birth_year'], 1990)

    def test_correction_to_passed_month_starts_next_year(self):
        event = self.create(kind='birthday', person='Оля', start_time=None)
        self.change(event, scope='future', data={
            'content': 'День рождения', 'kind': 'birthday', 'person': 'Оля', 'start_date': '1990-05-05'})
        self.assertEqual(self.occs(event)[0]['start_date'], '2027-05-05')

    def test_correction_respects_new_count_or_until(self):
        for limit in ({'count': 3}, {'until': '2028-11-05'}):
            with self.subTest(limit=limit):
                event = self.create(kind='birthday', person=str(limit), start_time=None)
                self.change(event, scope='future', data={
                    'content': 'День рождения', 'kind': 'birthday', 'person': str(limit),
                    'start_date': '2026-11-05', 'recurrence': {
                        'frequency': 'yearly', 'start_date': '2026-11-05',
                        'dates': [{'month': 11, 'day': 5}], **limit}})
                self.e.materialize(event['id'], NOW, through=date(2040, 1, 1))
                self.assertEqual([o['start_date'] for o in self.occs(event)],
                                 ['2026-11-05', '2027-11-05', '2028-11-05'])


class ToolBoundaryTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    def test_partial_standalone_delivery_blocks_update_and_cancel(self):
        reminder = self.execute('create_reminder', content='X' * 2500, due_date='2026-10-18')['result']
        due = ts('2026-10-18T09:00')
        self.store.prepare_reminders(due, lambda r: r['content'])
        first = self.store.next_outbox(due)
        self.store.start_delivery(first['key'])
        self.store.sent(first['key'], 1, due)
        remaining = [dict(r) for r in self.store.db.execute('SELECT * FROM outbox WHERE sent_at IS NULL')]
        self.assertTrue(remaining)
        for action, parameters in (
            ('update_reminder', {'content': 'Изменено', 'due_at': '2026-10-19T09:00+03:00'}),
            ('cancel_reminder', {}),
        ):
            with self.subTest(action=action):
                result = self.execute(action, id=reminder['id'], **parameters)
                self.assertEqual(result.get('result', result)['error']['code'], 'delivery_in_progress', result)
                self.assertEqual([dict(r) for r in self.store.db.execute('SELECT * FROM outbox WHERE sent_at IS NULL')], remaining)

    def test_partial_recurring_delivery_blocks_schedule_mutations(self):
        schedule = self.execute('create_schedule', content='X' * 2500, recurrence={
            'frequency': 'daily', 'start_date': '2026-10-18'})['result']
        due = ts('2026-10-18T09:00')
        self.store.prepare_reminders(due, lambda r: r['content'])
        first = self.store.next_outbox(due)
        self.store.start_delivery(first['key'])
        self.store.sent(first['key'], 1, due)
        remaining = [dict(r) for r in self.store.db.execute('SELECT * FROM outbox WHERE sent_at IS NULL')]
        for action, parameters in (
            ('update_schedule', {'content': 'Изменено'}),
            ('cancel_schedule', {}),
            ('skip_schedule', {}),
        ):
            with self.subTest(action=action):
                result = self.execute(action, id=schedule['id'], **parameters)
                self.assertEqual(result['error']['code'], 'delivery_in_progress', result)
                self.assertEqual([dict(r) for r in self.store.db.execute('SELECT * FROM outbox WHERE sent_at IS NULL')], remaining)

    def test_get_event_finds_distant_next_occurrence(self):
        event = self.execute('create_event', data={
            'content': 'Встреча', 'start_date': '2028-10-20', 'start_time': '12:00',
            'recurrence': {'frequency': 'yearly', 'start_date': '2028-10-20',
                           'interval': 5, 'dates': [{'month': 10, 'day': 20}]}})['result']
        result = self.tool('get_event', id=event['id'])
        self.assertIsNotNone(result['next_occurrence'])
        self.assertEqual(result['next_occurrence']['start_date'], '2028-10-20')
        selected = result['next_occurrence']['id']
        changed = self.execute('change_event', id=event['id'], operation='move', scope='one',
                               occurrence_id=selected, data={
                                   'content': 'Встреча', 'start_date': '2029-10-20', 'start_time': '12:00'})
        self.assertEqual(changed['status'], 'done')

    def test_manual_far_move_does_not_hide_nearer_unmaterialized_repeat(self):
        event = self.execute('create_event', data={
            'content': 'Встреча', 'start_date': '2026-10-20', 'start_time': '12:00',
            'recurrence': {'frequency': 'yearly', 'start_date': '2026-10-20',
                           'interval': 5, 'dates': [{'month': 10, 'day': 20}]}})['result']
        first = self.store.db.execute('SELECT id FROM event_occurrences WHERE event_id=?', (event['id'],)).fetchone()[0]
        self.execute('change_event', id=event['id'], operation='move', scope='one', occurrence_id=first,
                     data={'content': 'Встреча', 'start_date': '2090-10-20', 'start_time': '12:00'})
        result = self.tool('get_event', id=event['id'])
        self.assertEqual(result['next_occurrence']['start_date'], '2031-10-20')

    def test_frequency_change_keeps_new_explicit_limit(self):
        for limit in ({'count': 7}, {'until': '2027-12-31'}):
            with self.subTest(limit=limit):
                schedule = self.execute('create_schedule', content='Вода', recurrence={
                    'frequency': 'daily', 'start_date': '2026-10-20', 'count': 3})['result']
                changed = self.execute('update_schedule', id=schedule['id'], recurrence={
                    'frequency': 'weekly', 'weekdays': [1], **limit})
                self.assertEqual(changed['status'], 'done', changed)
                rule = json.loads(self.store.db.execute('SELECT rule FROM schedules WHERE id=?', (schedule['id'],)).fetchone()[0])
                self.assertEqual(rule.get('count'), limit.get('count'))
                self.assertEqual(rule.get('until'), limit.get('until'))

    def test_explicit_count_replaces_previous_until(self):
        schedule = self.execute('create_schedule', content='Вода', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20', 'until': '2026-10-22'})['result']
        changed = self.execute('update_schedule', id=schedule['id'], recurrence={'count': 7})
        self.assertEqual(changed['status'], 'done', changed)
        rule = json.loads(self.store.db.execute('SELECT rule FROM schedules WHERE id=?', (schedule['id'],)).fetchone()[0])
        self.assertEqual(rule['count'], 7)
        self.assertIsNone(rule['until'])
