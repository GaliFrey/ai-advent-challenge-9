"""Requirement regressions from independent review of e6f7ba9."""
import unittest

from personal_assistent.events import EventData, Events, EventError
from personal_assistent.storage import dump
import test_events as fixtures
from test_events import NOW


class EventSearchReviewTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool

    def test_search_long_content_makes_progress_and_full_text_remains_readable(self):
        for suffix in ('😀' * 3990, '\x01' * 3990):
            with self.subTest(suffix=suffix[:1]):
                content = 'Встреча ' + suffix
                with self.store.atomic():
                    event = Events(self.store).create(EventData(content=content,
                        start_date='2026-10-20', start_time='12:00'), NOW, 1)
                page = self.tool('search_events', query='встреча', offset=event['id'] - 1, limit=1)
                self.assertEqual(len(page['items']), 1, page)
                found = page['items'][0]
                self.assertEqual(found['id'], event['id'])
                self.assertTrue(found['incomplete'])
                self.assertTrue(page['next_offset'] is None or page['next_offset'] > event['id'] - 1)
                self.assertLessEqual(len(dump(page).encode('utf-8')), 12000)
                restored, offset = '', 0
                while True:
                    detail = self.tool('get_event', id=event['id'], content_offset=offset)
                    restored += detail['content']
                    if detail['next_content_offset'] is None:
                        break
                    self.assertGreater(detail['next_content_offset'], offset)
                    offset = detail['next_content_offset']
                self.assertEqual(restored, content)


class RuleReplacementReviewTests(unittest.TestCase):
    setUp = fixtures.EventTests.setUp
    create = fixtures.EventTests.create
    rule = fixtures.EventTests.rule
    occs = fixtures.EventTests.occs

    def test_replacing_already_excluded_rule_is_rejected_without_extra_notification(self):
        event = self.create()
        occurrence = self.occs(event)[0]['id']
        original = self.rule(event, offset=1)
        replacement = self.rule(event, scope='one', occurrence_id=occurrence,
                                replace_rule_id=original['rule_id'], offset=2)
        before = [dict(row) for row in self.db.execute('SELECT * FROM reminders ORDER BY id')]
        with self.assertRaises(EventError) as error:
            self.rule(event, scope='one', occurrence_id=occurrence,
                      replace_rule_id=original['rule_id'], offset=0)
        self.assertEqual(error.exception.code, 'rule_scope_mismatch')
        self.assertEqual([dict(row) for row in self.db.execute('SELECT * FROM reminders ORDER BY id')], before)
        current = self.rule(event, scope='one', occurrence_id=occurrence,
                            replace_rule_id=replacement['rule_id'], offset=0)
        pending = self.db.execute("SELECT event_rule_id FROM reminders WHERE status='pending'").fetchall()
        self.assertEqual([row[0] for row in pending], [current['rule_id']])
