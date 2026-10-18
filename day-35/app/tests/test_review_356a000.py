"""Requirement regressions from independent review of 356a000."""
import unittest

from personal_assistent.storage import dump
from personal_assistent.events import EventError
from personal_assistent.events import Events
import test_events as fixtures
from test_events import NOW, ts


class ReminderRequestTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    def at(self, now, name, **arguments):
        self.n += 1
        return self.runner.run({'id': str(self.n), 'function': {
            'name': name, 'arguments': dump(arguments)}}, self.job, now)

    def test_date_only_move_keeps_default_clock_and_pinned_zone(self):
        reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20')['result']
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        moved = self.execute('update_reminder', id=reminder['id'], due_date='2026-10-21')
        self.assertEqual(moved['status'], 'done', moved)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T09:00'))
        self.assertEqual(self.store.db.execute('SELECT timezone,time_mode FROM schedules').fetchone()[:],
                         ('Europe/Kirov', 'default'))

    def test_date_only_move_keeps_explicit_clock_and_pinned_zone(self):
        reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20', due_time='08:00')['result']
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        moved = self.execute('update_reminder', id=reminder['id'], due_date='2026-10-21')
        self.assertEqual(moved['status'], 'done', moved)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T08:00'))

    def test_text_only_edit_of_overdue_unstarted_notification_keeps_due(self):
        reminder = self.execute('create_reminder', content='Вода', due_at='2026-10-17T10:01+03:00')['result']
        now = NOW + 61
        self.store.prepare_reminders(now, lambda item: item['content'])
        ready = self.at(now, 'prepare_request', action='update_reminder', parameters={
            'id': reminder['id'], 'content': 'Чай'})
        self.assertEqual(ready['status'], 'ready', ready)
        changed = self.at(now, 'execute_request', request_id=ready['request_id'])
        self.assertEqual(changed['status'], 'done', changed)
        self.assertEqual(tuple(self.store.db.execute('SELECT content,due_at FROM reminders').fetchone()),
                         ('Чай', NOW + 60))
        self.store.prepare_reminders(now, lambda item: item['content'])
        self.assertEqual(self.store.next_outbox(now)['text'], 'Чай')

    def test_explicit_move_to_past_is_still_rejected(self):
        reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20')['result']
        draft = self.tool('prepare_request', action='update_reminder', parameters={
            'id': reminder['id'], 'due_at': '2026-10-17T09:59+03:00'})
        self.assertEqual(draft['status'], 'waiting', draft)
        self.assertEqual(draft['error'][0]['code'], 'past_due')

    def test_date_only_move_of_recurring_instance_uses_source_zone(self):
        self.execute('create_schedule', content='Вода', due_time='08:00', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20'})
        self.store.prepare_reminders(ts('2026-10-20T08:00'), lambda item: item['content'])
        identity = self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        moved = self.execute('update_reminder', id=identity, due_date='2026-10-21')
        self.assertEqual(moved['status'], 'done', moved)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T08:00'))

    def test_local_move_of_legacy_unknown_zone_records_current_zone(self):
        # The migration creates this technical parent without a known local date.
        with self.store.atomic():
            identity = self.store.db.execute('INSERT INTO reminders(content,due_at,created_at,source_message_id) VALUES(?,?,?,?)',
                                             ('Старое', ts('2026-10-20T09:00'), NOW, 1)).lastrowid
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        changed = self.execute('update_reminder', id=identity, due_date='2026-10-21', due_time='08:00')
        self.assertEqual(changed['status'], 'done', changed)
        self.assertEqual(tuple(self.store.db.execute('SELECT timezone,local_date FROM schedules').fetchone()),
                         ('Asia/Tokyo', '2026-10-21'))

    def test_enabling_digest_before_first_day_clock_waits_until_due(self):
        self.execute('create_event', data={'content': 'Оля', 'kind': 'birthday',
                     'person': 'Оля', 'start_date': '2026-11-20'})
        now = ts('2026-11-01T08:00')
        text = 'Включи сводку дней рождения'
        self.job['payload'] = dump({'message': {'message_id': 2, 'date': now, 'text': text}})
        changed = self.at(now, 'update_profile', birthday_digest=True, source_text=text)
        self.assertNotIn('error', changed, changed)
        self.assertIsNone(self.store.next_outbox(now))
        due = ts('2026-11-01T09:00')
        self.store.prepare_reminders(due, lambda item: item['content'])
        self.assertIn('Оля', self.store.next_outbox(due)['text'])

    def test_enabling_digest_midmonth_still_delivers_immediately(self):
        self.execute('create_event', data={'content': 'Оля', 'kind': 'birthday',
                     'person': 'Оля', 'start_date': '2026-11-20'})
        now = ts('2026-11-15T08:00')
        text = 'Включи сводку дней рождения'
        self.job['payload'] = dump({'message': {'message_id': 2, 'date': now, 'text': text}})
        changed = self.at(now, 'update_profile', birthday_digest=True, source_text=text)
        self.assertNotIn('error', changed, changed)
        self.assertIn('Оля', self.store.next_outbox(now)['text'])

    def test_date_only_move_after_selection_uses_chosen_once_clock_and_zone(self):
        for hour in (None, '08:00'):
            with self.subTest(hour=hour):
                self.store.set_meta('profile', {'timezone': 'Europe/Kirov'})
                parameters = {'due_time': hour} if hour else {}
                reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20', **parameters)['result']
                self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
                self.job['pending'] = None
                draft = self.tool('prepare_request', action='update_reminder', parameters={'due_date': '2026-10-21'})
                self.assertEqual(draft['status'], 'waiting', draft)
                self.job['pending'] = {'selection': {'record_type': 'reminder', 'id': reminder['id']}}
                ready = self.tool('prepare_request', continue_request=True, parameters={})
                self.assertEqual(ready['status'], 'ready', ready)
                expected = ts('2026-10-21T' + (hour or '09:00'))
                self.assertEqual(ts(ready['parameters']['due_at']), expected)
                result = self.tool('execute_request', request_id=ready['request_id'])
                self.assertEqual(result['status'], 'done', result)
                self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders WHERE id=?', (reminder['id'],)).fetchone()[0], expected)
                self.job['pending'] = None

    def test_recurring_instance_move_after_selection_uses_chosen_clock_and_zone(self):
        self.execute('create_schedule', content='Вода', due_time='08:00', recurrence={
            'frequency': 'daily', 'start_date': '2026-10-20'})
        self.store.prepare_reminders(ts('2026-10-20T08:00'), lambda item: item['content'])
        identity = self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        draft = self.tool('prepare_request', action='update_reminder', parameters={'due_date': '2026-10-21'})
        self.assertEqual(draft['status'], 'waiting', draft)
        self.job['pending'] = {'selection': {'record_type': 'reminder', 'id': identity}}
        ready = self.tool('prepare_request', continue_request=True, parameters={})
        self.assertEqual(ready['status'], 'ready', ready)
        result = self.tool('execute_request', request_id=ready['request_id'])
        self.assertEqual(result['status'], 'done', result)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T08:00'))

    def test_selection_keeps_explicit_reset_to_default_clock_dynamic(self):
        reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20', due_time='08:00')['result']
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        draft = self.tool('prepare_request', action='update_reminder', parameters={
            'due_date': '2026-10-21', 'due_time': None})
        self.assertEqual(draft['status'], 'waiting', draft)
        self.job['pending'] = {'selection': {'record_type': 'reminder', 'id': reminder['id']}}
        ready = self.tool('prepare_request', continue_request=True, parameters={})
        self.assertEqual(ready['status'], 'ready', ready)
        result = self.tool('execute_request', request_id=ready['request_id'])
        self.assertEqual(result['status'], 'done', result)
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo', 'notification_time': '11:00'})
        with self.store.atomic():
            Events(self.store).profile_changed(NOW)
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T11:00'))

    def test_selection_can_replace_calendar_deadline_with_absolute_and_back(self):
        for initial, patch in (
            ({'due_date': '2026-10-21'}, {'due_at': '2026-10-22T15:00+03:00'}),
            ({'due_at': '2026-10-21T12:00+03:00'}, {'due_date': '2026-10-22', 'due_time': '15:00'}),
            ({'due_at': '2026-10-21T12:00+03:00'}, {'due_time': '15:00'}),
        ):
            with self.subTest(initial=initial, patch=patch):
                self.job['pending'] = None
                reminder = self.execute('create_reminder', content='Вода', due_date='2026-10-20', due_time='08:00')['result']
                draft = self.tool('prepare_request', action='update_reminder', parameters=initial)
                self.assertEqual(draft['status'], 'waiting', draft)
                self.job['pending'] = {'selection': {'record_type': 'reminder', 'id': reminder['id']}}
                ready = self.tool('prepare_request', continue_request=True, parameters=patch)
                self.assertEqual(ready['status'], 'ready', ready)
                changed = self.tool('execute_request', request_id=ready['request_id'])
                self.assertEqual(changed['status'], 'done', changed)
                expected_day = '22' if 'due_at' in patch or 'due_date' in patch else '21'
                actual = self.store.db.execute('SELECT due_at FROM reminders WHERE id=?', (reminder['id'],)).fetchone()[0]
                self.assertEqual(actual, ts('2026-10-' + expected_day + 'T15:00'))
                self.job['pending'] = None


class EventScopeTests(unittest.TestCase):
    setUp = fixtures.EventTests.setUp
    create = fixtures.EventTests.create
    change = fixtures.EventTests.change
    rule = fixtures.EventTests.rule
    occs = fixtures.EventTests.occs

    def test_later_interval_does_not_block_all_day_conversion_of_earlier_instance(self):
        event = self.create(recurrence={'frequency': 'daily', 'start_date': '2026-10-20', 'count': 2})
        self.rule(event, kind='interval', offset=3600, from_position=1)
        changed = self.change(event, data={'content': 'Встреча', 'start_date': '2026-10-20', 'all_day': True})
        self.assertIsNone(changed['occurrence']['start_time'])
        self.assertEqual(self.occs(event)[1]['start_time'], '12:00:00')

    def test_excluded_interval_does_not_block_all_day_conversion(self):
        event = self.create()
        interval = self.rule(event, kind='interval', offset=3600)
        self.rule(event, scope='one', occurrence_id=self.occs(event)[0]['id'],
                  replace_rule_id=interval['rule_id'], offset=1)
        changed = self.change(event, data={'content': 'Встреча', 'start_date': '2026-10-20', 'all_day': True})
        self.assertIsNone(changed['occurrence']['start_time'])

    def test_interval_allowed_when_finite_remaining_occurrence_is_manually_timed(self):
        event = self.create(start_time=None, all_day=True,
                            recurrence={'frequency': 'daily', 'start_date': '2026-10-20', 'count': 2})
        self.change(event, occurrence_id=self.occs(event)[1]['id'],
                    data={'content': 'Встреча', 'start_date': '2026-10-21', 'start_time': '12:00'})
        added = self.rule(event, kind='interval', offset=3600, from_position=1)
        self.assertIn('rule_id', added)
        self.assertEqual(self.db.execute('SELECT due_at FROM reminders').fetchone()[0], ts('2026-10-21T11:00'))

    def test_unmaterialized_all_day_repetitions_still_require_reference(self):
        event = self.create(start_time=None, all_day=True,
                            recurrence={'frequency': 'daily', 'start_date': '2026-10-20'})
        with self.assertRaises(EventError) as error:
            self.rule(event, kind='interval', offset=3600, from_position=1000)
        self.assertEqual(error.exception.code, 'all_day_reference_required')

    def test_future_all_day_conversion_checks_unmaterialized_interval_scope(self):
        event = self.create(recurrence={'frequency': 'daily', 'start_date': '2026-10-20'})
        self.rule(event, kind='interval', offset=3600, from_position=1000)
        with self.assertRaises(EventError) as error:
            self.change(event, scope='future', data={
                'content': 'Встреча', 'start_date': '2026-10-20', 'all_day': True})
        self.assertEqual(error.exception.code, 'all_day_reference_required')
        self.assertEqual(self.occs(event)[0]['start_time'], '12:00:00')

    def test_interval_outside_shortened_plan_does_not_block_all_day_conversion(self):
        event = self.create(recurrence={'frequency': 'daily', 'start_date': '2026-10-20', 'count': 2})
        self.rule(event, kind='interval', offset=3600, from_position=1)
        self.change(event, scope='future', data={
            'content': 'Встреча', 'start_date': '2026-10-20', 'all_day': True,
            'recurrence': {'frequency': 'daily', 'start_date': '2026-10-20', 'count': 1}})
        self.assertIsNone(self.occs(event)[0]['start_time'])
        self.assertEqual(self.occs(event)[1]['status'], 'cancelled')

    def test_future_conversion_preserves_manual_timed_instance_and_its_interval(self):
        event = self.create(recurrence={'frequency': 'daily', 'start_date': '2026-10-20', 'count': 2})
        first = self.occs(event)[0]['id']
        self.change(event, data={'content': 'Исключение', 'start_date': '2026-10-20', 'start_time': '14:00'})
        self.rule(event, scope='one', occurrence_id=first, kind='interval', offset=3600)
        self.change(event, scope='future', data={
            'content': 'Встреча', 'start_date': '2026-10-21', 'all_day': True})
        actual = self.occs(event)
        self.assertEqual(actual[0]['start_time'], '14:00:00')
        self.assertIsNone(actual[1]['start_time'])
        self.assertEqual(self.db.execute('SELECT due_at,status FROM reminders').fetchone()[:],
                         (ts('2026-10-20T13:00'), 'pending'))
