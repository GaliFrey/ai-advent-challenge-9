"""Requirement regressions from the independent review of 5c6553d."""
import unittest
from datetime import date

from personal_assistent.events import EventData, Events, EventError
import test_events as fixtures
from test_events import NOW, ts


class FutureSeriesTests(unittest.TestCase):
    setUp = fixtures.EventTests.setUp
    create = fixtures.EventTests.create
    change = fixtures.EventTests.change
    occs = fixtures.EventTests.occs
    rule = fixtures.EventTests.rule

    def test_move_series_starting_beyond_lookahead(self):
        event = self.create(start_date='2028-10-20', recurrence={
            'frequency': 'yearly', 'start_date': '2028-10-20',
            'dates': [{'month': 10, 'day': 20}], 'interval': 2, 'count': 3})
        self.change(event, scope='future', data={
            'content': 'Позже', 'start_date': '2028-10-21', 'start_time': '12:00',
            'recurrence': {'frequency': 'yearly', 'start_date': '2028-10-21',
                           'dates': [{'month': 10, 'day': 21}], 'interval': 2}})
        self.assertEqual(self.occs(event)[0]['start_date'], '2028-10-21')
        self.e.materialize(event['id'], NOW, through=date(2035, 1, 1))
        self.assertEqual([o['start_date'] for o in self.occs(event)],
                         ['2028-10-21', '2030-10-21', '2032-10-21'])

    def test_move_after_past_instance_with_distant_next_repeat(self):
        event = self.create(start_date='2025-10-20', recurrence={
            'frequency': 'yearly', 'start_date': '2025-10-20',
            'dates': [{'month': 10, 'day': 20}], 'interval': 5, 'count': 2})
        past = self.occs(event)[0]
        self.change(event, scope='future', data={
            'content': 'Позже', 'start_date': '2030-10-21', 'start_time': '12:00',
            'recurrence': {'frequency': 'yearly', 'start_date': '2030-10-21',
                           'dates': [{'month': 10, 'day': 21}], 'interval': 5}})
        self.assertEqual(self.occs(event)[0], past)
        self.assertEqual(self.occs(event)[1]['position'], 1)
        self.e.materialize(event['id'], NOW, through=date(2040, 1, 1))
        self.assertEqual(len(self.occs(event)), 2)

    def test_exhausted_series_still_rejects_move(self):
        event = self.create(start_date='2025-10-20', recurrence={
            'frequency': 'daily', 'start_date': '2025-10-20', 'count': 2})
        with self.assertRaises(EventError) as error:
            self.change(event, scope='future', data={
                'content': 'Позже', 'start_date': '2028-10-21', 'start_time': '12:00'})
        self.assertEqual(error.exception.code, 'series_exhausted')

    def test_move_restores_relative_notification_previously_after_start(self):
        event = self.create(start_time='08:00')
        self.rule(event, offset=0)
        self.assertEqual(self.db.execute('SELECT reason FROM reminders').fetchone()[0], 'after_event')
        self.change(event, data={'content': 'Позже', 'start_date': '2026-10-20', 'start_time': '10:00'})
        self.assertEqual(tuple(self.db.execute('SELECT status,due_at FROM reminders').fetchone()),
                         ('pending', ts('2026-10-20T09:00')))

    def test_future_change_preserves_old_manual_move_without_shifting_regular_start(self):
        event = self.create(start_date='2026-10-18', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-18', 'count': 5})
        self.change(event, data={'content': 'Исключение', 'start_date': '2026-10-30', 'start_time': '12:00'})
        before = self.occs(event)
        with self.store.atomic():
            self.e.change(event['id'], 'move', ts('2026-10-20T13:00'), scope='future', from_position=0,
                          data=EventData(content='Новый срок', start_date='2026-10-25', start_time='12:00'))
        actual = self.occs(event)
        self.assertEqual(actual[:3], before[:3])
        self.assertEqual([o['start_date'] for o in actual[3:]], ['2026-10-25', '2026-10-26'])

    def test_future_cancel_includes_old_instance_moved_into_future(self):
        event = self.create(start_date='2026-10-18', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-18', 'count': 5})
        self.change(event, data={'content': 'Исключение', 'start_date': '2026-10-30', 'start_time': '12:00'})
        with self.store.atomic():
            self.e.change(event['id'], 'cancel', ts('2026-10-20T13:00'), scope='future', from_position=0)
        self.assertEqual([o['status'] for o in self.occs(event)],
                         ['cancelled', 'active', 'active', 'cancelled', 'cancelled'])


class ReminderEditingTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    def test_text_edit_keeps_default_clock_dependency(self):
        reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20')['result']
        self.execute('update_reminder', id=reminder['id'], content='Вода исправлено')
        self.store.set_meta('profile', {'notification_time': '11:00'})
        with self.store.atomic():
            Events(self.store).profile_changed(NOW)
        row = self.store.db.execute('SELECT content,due_at,is_manual FROM reminders').fetchone()
        self.assertEqual(tuple(row), ('Вода исправлено', ts('2026-10-20T11:00'), 0))

    def test_new_default_date_updates_source_and_follows_profile(self):
        reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20', due_time='08:00')['result']
        self.execute('update_reminder', id=reminder['id'], due_date='2026-10-21',
                     due_time=None)
        self.store.set_meta('profile', {'notification_time': '11:00'})
        with self.store.atomic():
            Events(self.store).profile_changed(NOW)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T11:00'))

    def test_text_edit_of_recurring_instance_keeps_default_clock(self):
        self.execute('create_schedule', content='Вода', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20', 'count': 2})
        self.store.prepare_reminders(ts('2026-10-20T09:00'), lambda r: r['content'])
        identity = self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.execute('update_reminder', id=identity, content='Вода исправлено')
        self.store.set_meta('profile', {'notification_time': '11:00'})
        with self.store.atomic():
            Events(self.store).profile_changed(ts('2026-10-20T09:01'))
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders WHERE id=?', (identity,)).fetchone()[0],
                         ts('2026-10-20T11:00'))

    def test_ordinal_monthly_schedule_can_change_frequency(self):
        schedule = self.execute('create_schedule', content='Вода', recurrence={
            'frequency': 'monthly', 'start_date': '2026-10-20',
            'ordinal_weekdays': [{'ordinal': 2, 'weekday': 1}]})['result']
        result = self.execute('update_schedule', id=schedule['id'], recurrence={
            'frequency': 'weekly', 'weekdays': [1]})
        self.assertEqual(result['status'], 'done', result)

    def test_ordinal_schedule_frequency_change_after_selecting_target(self):
        schedule = self.execute('create_schedule', content='Вода', recurrence={
            'frequency': 'monthly', 'start_date': '2026-10-20',
            'ordinal_weekdays': [{'ordinal': 2, 'weekday': 1}]})['result']
        draft = self.tool('prepare_request', action='update_schedule', parameters={
            'recurrence': {'frequency': 'weekly', 'weekdays': [1]}})
        self.assertEqual(draft['status'], 'waiting', draft)
        self.job['pending'] = {'selection': {'record_type': 'schedule', 'id': schedule['id']}}
        ready = self.tool('prepare_request', continue_request=True, parameters={})
        self.assertEqual(ready['status'], 'ready', ready)
        result = self.tool('execute_request', request_id=ready['request_id'])
        self.assertEqual(result['status'], 'done', result)
