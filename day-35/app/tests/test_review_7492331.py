"""Requirement regressions from independent review of 7492331."""
import unittest
import tempfile
from pathlib import Path

from personal_assistent.events import Events, EventData, EventError
from personal_assistent.storage import Store, dump
from personal_assistent.migrate import migrate_database, requires_migration
import test_events as fixtures
from test_events import NOW, ts


class ScheduleSelectionTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    def select_update(self, schedule, parameters):
        draft = self.tool('prepare_request', action='update_schedule', parameters=parameters)
        self.assertEqual(draft['status'], 'waiting', draft)
        self.job['pending'] = {'selection': {'record_type': 'schedule', 'id': schedule['id']}}
        ready = self.tool('prepare_request', continue_request=True, parameters={})
        self.assertEqual(ready['status'], 'ready', ready)
        result = self.tool('execute_request', request_id=ready['request_id'])
        self.assertEqual(result['status'], 'done', result)
        return result

    def test_selected_text_edit_preserves_explicit_clock(self):
        schedule = self.execute('create_schedule', content='Вода', due_time='08:00', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20'})['result']
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        self.select_update(schedule, {'content': 'Чай'})
        self.assertEqual(tuple(self.store.db.execute('SELECT content,due_time,timezone,time_mode,next_due_at FROM schedules').fetchone()),
                         ('Чай', '08:00:00', 'Europe/Kirov', 'explicit', ts('2026-10-20T08:00')))

    def test_selected_reset_to_default_follows_profile_clock(self):
        schedule = self.execute('create_schedule', content='Вода', due_time='08:00', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20'})['result']
        self.select_update(schedule, {'due_time': None})
        self.store.set_meta('profile', {'notification_time': '11:00'})
        with self.store.atomic():
            Events(self.store).profile_changed(NOW)
        self.assertEqual(tuple(self.store.db.execute('SELECT time_mode,next_due_at FROM schedules').fetchone()),
                         ('default', ts('2026-10-20T11:00')))

    def test_selected_explicit_clock_remains_explicit(self):
        schedule = self.execute('create_schedule', content='Вода', due_time='08:00', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20'})['result']
        self.select_update(schedule, {'due_time': '07:00'})
        self.assertEqual(tuple(self.store.db.execute('SELECT time_mode,next_due_at FROM schedules').fetchone()),
                         ('explicit', ts('2026-10-20T07:00')))

    def test_calendar_draft_can_switch_to_relative_deadline(self):
        draft = self.tool('prepare_request', action='create_reminder', parameters={'due_date': '2026-10-20'})
        self.assertEqual(draft['status'], 'waiting', draft)
        ready = self.tool('prepare_request', continue_request=True,
                          parameters={'content': 'Вода', 'after_seconds': 600})
        self.assertEqual(ready['status'], 'ready', ready)
        result = self.tool('execute_request', request_id=ready['request_id'])
        self.assertEqual(result['status'], 'done', result)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], NOW + 600)

    def test_absolute_or_relative_draft_can_switch_to_default_calendar_clock(self):
        for initial in ({'after_seconds': 600}, {'due_at': '2026-10-20T15:00+03:00'}):
            with self.subTest(initial=initial):
                draft = self.tool('prepare_request', action='create_reminder', parameters=initial)
                self.assertEqual(draft['status'], 'waiting', draft)
                ready = self.tool('prepare_request', continue_request=True, parameters={
                    'content': 'Вода', 'due_date': '2026-10-22', 'due_time': None})
                self.assertEqual(ready['status'], 'ready', ready)
                result = self.tool('execute_request', request_id=ready['request_id'])
                self.assertEqual(result['status'], 'done', result)
                self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders WHERE id=?',
                                 (result['result']['id'],)).fetchone()[0], ts('2026-10-22T09:00'))

    def test_default_deadline_at_execution_uses_current_profile_clock(self):
        reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-25', due_time='08:00')['result']
        ready = self.tool('prepare_request', action='update_reminder', parameters={
            'id': reminder['id'], 'due_date': '2026-10-21', 'due_time': None})
        self.assertEqual(ready['status'], 'ready', ready)
        self.store.set_meta('profile', {'notification_time': '11:00'})
        self.n += 1
        result = self.runner.run({'id': str(self.n), 'function': {
            'name': 'execute_request', 'arguments': dump({'request_id': ready['request_id']})}},
            self.job, ts('2026-10-21T10:00'))
        self.assertEqual(result['status'], 'done', result)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T11:00'))


class EventReviewTests(unittest.TestCase):
    setUp = fixtures.EventTests.setUp
    create = fixtures.EventTests.create
    change = fixtures.EventTests.change
    rule = fixtures.EventTests.rule
    occs = fixtures.EventTests.occs

    def test_duplicate_birthday_without_optional_year_keeps_existing_year(self):
        event = self.create(kind='birthday', person='Оля', start_time=None, birth_year=1990)
        duplicate = self.create(kind='birthday', person='Оля', start_time=None)
        self.assertEqual(event['id'], duplicate['id'])
        self.assertEqual(self.e.event(event['id'])['birth_year'], 1990)

    def test_expanding_shortened_plan_restores_only_system_excluded_instances(self):
        event = self.create(recurrence={'frequency': 'daily', 'start_date': '2026-10-20', 'count': 4})
        self.rule(event)
        before = self.occs(event)
        self.change(event, operation='cancel', occurrence_id=before[1]['id'])
        for count in (1, 4):
            self.change(event, scope='future', data={'content': 'Встреча', 'start_date': '2026-10-20',
                'start_time': '12:00', 'recurrence': {'frequency': 'daily', 'start_date': '2026-10-20', 'count': count}})
        actual = self.occs(event)
        self.assertEqual([o['id'] for o in actual], [o['id'] for o in before])
        self.assertEqual([o['status'] for o in actual], ['active', 'cancelled', 'active', 'active'])
        self.assertEqual([r[0] for r in self.db.execute('SELECT status FROM reminders ORDER BY id')],
                         ['pending', 'cancelled', 'pending', 'pending'])

    def test_restored_interval_still_blocks_silent_all_day_conversion(self):
        event = self.create(recurrence={'frequency': 'daily', 'start_date': '2026-10-20', 'count': 3})
        self.rule(event, kind='interval', offset=3600, from_position=1)
        self.change(event, scope='future', data={'content': 'Встреча', 'start_date': '2026-10-20',
            'start_time': '12:00', 'recurrence': {'frequency': 'daily', 'start_date': '2026-10-20', 'count': 1}})
        with self.assertRaises(EventError) as error:
            self.change(event, scope='future', data={'content': 'Встреча', 'start_date': '2026-10-20',
                'all_day': True, 'recurrence': {'frequency': 'daily', 'start_date': '2026-10-20', 'count': 3}})
        self.assertEqual(error.exception.code, 'all_day_reference_required')


class MigrationReviewTests(unittest.TestCase):
    def test_version_two_upgrade_preserves_unknown_cancellation_and_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path, backup = Path(directory) / 'database.sqlite', Path(directory) / 'before.sqlite'
            store = Store(path)
            events = Events(store)
            with store.atomic():
                event = events.create(EventData(content='Встреча', start_date='2026-10-20', start_time='12:00'), NOW, 1)
                occurrence = store.db.execute('SELECT id FROM event_occurrences').fetchone()[0]
                events.change(event['id'], 'cancel', NOW, scope='one', occurrence_id=occurrence)
                store.db.execute('ALTER TABLE event_occurrences DROP COLUMN cancel_reason')
                store.set_meta('events_schema_version', 2)
            previous = dict(store.db.execute('SELECT * FROM event_occurrences').fetchone())
            store.db.close()
            self.assertTrue(requires_migration(path))
            migrate_database(path, backup)
            store = Store(path)
            try:
                current = dict(store.db.execute('SELECT * FROM event_occurrences').fetchone())
                self.assertIsNone(current.pop('cancel_reason'))
                self.assertEqual(current, previous)
                self.assertEqual(store.get_meta('events_schema_version'), 3)
                self.assertIsNone(store.db.execute('PRAGMA foreign_key_check').fetchone())
            finally:
                store.db.close()
            self.assertTrue(requires_migration(backup))
