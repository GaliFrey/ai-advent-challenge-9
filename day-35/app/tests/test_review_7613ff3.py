"""Requirement regressions from independent review of 7613ff3."""
import unittest

from personal_assistent.storage import dump
import test_events as fixtures
from test_events import NOW, ts


class ScheduleClockTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    def execute_at(self, request_id, now):
        self.n += 1
        return self.runner.run({'id': str(self.n), 'function': {
            'name': 'execute_request', 'arguments': dump({'request_id': request_id})}}, self.job, now)

    def test_create_finite_default_schedule_uses_current_clock_for_validation(self):
        ready = self.tool('prepare_request', action='create_schedule', parameters={
            'content': 'Вода', 'recurrence': {'frequency': 'daily', 'start_date': '2026-10-21', 'count': 1}})
        self.assertEqual(ready['status'], 'ready', ready)
        self.store.set_meta('profile', {'notification_time': '11:00'})
        result = self.execute_at(ready['request_id'], ts('2026-10-21T10:00'))
        self.assertEqual(result.get('status'), 'done', result)
        self.assertEqual(self.store.db.execute('SELECT next_due_at FROM schedules').fetchone()[0], ts('2026-10-21T11:00'))

    def test_elapsed_explicit_schedule_does_not_adopt_default_clock(self):
        ready = self.tool('prepare_request', action='create_schedule', parameters={
            'content': 'Вода', 'due_time': '09:00',
            'recurrence': {'frequency': 'daily', 'start_date': '2026-10-21', 'count': 1}})
        self.assertEqual(ready['status'], 'ready', ready)
        self.store.set_meta('profile', {'notification_time': '11:00'})
        result = self.execute_at(ready['request_id'], ts('2026-10-21T10:00'))
        self.assertEqual(result['error']['code'], 'schedule_out_of_range', result)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM schedules').fetchone()[0], 0)

    def test_schedule_text_edit_preserves_queued_overdue_occurrence(self):
        schedule = self.execute('create_schedule', content='Вода', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20', 'count': 3})['result']
        now = ts('2026-10-20T10:00')
        self.store.prepare_reminders(now, lambda item: item['content'])
        before = dict(self.store.db.execute('SELECT * FROM reminders').fetchone())
        cursor = self.store.db.execute('SELECT next_due_at FROM schedules').fetchone()[0]
        ready = self.tool('prepare_request', action='update_schedule', parameters={'id': schedule['id'], 'content': 'Чай'})
        result = self.execute_at(ready['request_id'], now)
        self.assertEqual(result.get('status'), 'done', result)
        self.store.prepare_reminders(now, lambda item: item['content'])
        after = dict(self.store.db.execute('SELECT * FROM reminders').fetchone())
        self.assertEqual((after['id'], after['due_at'], after['planned_due_at'], after['status'], after['content']),
                         (before['id'], before['due_at'], before['planned_due_at'], 'queued', 'Чай'))
        self.assertEqual(self.store.db.execute('SELECT next_due_at FROM schedules').fetchone()[0], cursor)
        self.assertEqual(self.store.next_outbox(now)['text'], 'Чай')

    def test_schedule_text_edit_preserves_unprocessed_overdue_cursor(self):
        schedule = self.execute('create_schedule', content='Вода', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20', 'count': 3})['result']
        ready = self.tool('prepare_request', action='update_schedule', parameters={'id': schedule['id'], 'content': 'Чай'})
        now = ts('2026-10-20T10:00')
        result = self.execute_at(ready['request_id'], now)
        self.assertEqual(result.get('status'), 'done', result)
        self.assertEqual(self.store.db.execute('SELECT next_due_at FROM schedules').fetchone()[0], ts('2026-10-20T09:00'))
        self.store.prepare_reminders(now, lambda item: item['content'])
        row = self.store.db.execute('SELECT content,due_at,status FROM reminders').fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(tuple(row), ('Чай', ts('2026-10-20T09:00'), 'queued'))
