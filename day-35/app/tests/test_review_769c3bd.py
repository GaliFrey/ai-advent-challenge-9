"""Requirement regressions from independent review of 769c3bd."""
import unittest

from personal_assistent.storage import dump
import test_events as fixtures


class DeletedEventProfileTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool
    execute = fixtures.EventToolTests.execute

    def test_deleted_event_does_not_block_clock_change_or_restore_notifications(self):
        for offset, hour in ((5, '12:00'), (0, '08:00'), (1, '12:00')):
            with self.subTest(offset=offset, hour=hour):
                event = self.execute('create_event', data={
                    'content': 'Встреча', 'start_date': '2026-10-20', 'start_time': hour})['result']
                self.execute('set_event_rule', id=event['id'], scope='future', from_position=0,
                             rule={'kind': 'calendar', 'offset': offset})
                deleted = self.execute('change_event', id=event['id'], operation='delete',
                                       scope='future', from_position=0)
                self.assertEqual(deleted['status'], 'done', deleted)
                before = [dict(row) for row in self.store.db.execute('SELECT * FROM reminders ORDER BY id')]
                text = 'Поставь общее время уведомлений 11:00'
                self.job['payload'] = dump({'message': {'text': text, 'message_id': 2}})
                changed = self.tool('update_profile', notification_time='11:00', source_text=text)
                self.assertNotIn('error', changed, changed)
                self.assertEqual(changed['profile']['notification_time'], '11:00')
                self.assertEqual([dict(row) for row in self.store.db.execute('SELECT * FROM reminders ORDER BY id')], before)
                # Reset just the profile clock so each scenario exercises a change.
                self.store.set_meta('profile', {'notification_time': '09:00'})
