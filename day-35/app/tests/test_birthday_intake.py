"""Regressions for the birthday creation reported from Telegram."""
import unittest

from personal_assistent.receipts import confirmed_text
import test_events as fixtures


class BirthdayIntakeTests(unittest.TestCase):
    setUp = fixtures.EventToolTests.setUp
    tool = fixtures.EventToolTests.tool

    def test_name_and_date_are_sufficient_without_an_event_title(self):
        draft = self.tool('prepare_request', action='create_event', parameters={'data': {
            'kind': 'birthday', 'person': 'Оля Пушная',
            'start_date': '1995-10-12', 'birth_year': 1995}})
        self.assertEqual(draft['status'], 'ready', draft)
        saved = self.tool('execute_request', request_id=draft['request_id'])
        self.assertEqual(saved['status'], 'done', saved)
        event = self.store.db.execute('SELECT * FROM events').fetchone()
        self.assertEqual((event['kind'], event['person'], event['birth_year']),
                         ('birthday', 'Оля Пушная', 1995))
        self.assertIn('Оля Пушная', event['content'])
        reply = confirmed_text(self.store, self.job['id'])
        self.assertIn('12.10.1995', reply)
        self.assertIn('ежегодно', reply)
        self.assertNotIn('2026-10-12', reply)

    def test_unknown_year_and_explicit_title_are_preserved(self):
        draft = self.tool('prepare_request', action='create_event', parameters={'data': {
            'kind': 'birthday', 'person': 'Оля', 'start_date': '2026-10-12',
            'content': 'День рождения подруги'}})
        saved = self.tool('execute_request', request_id=draft['request_id'])
        self.assertEqual(saved['status'], 'done', saved)
        event = self.store.db.execute('SELECT * FROM events').fetchone()
        self.assertEqual(event['content'], 'День рождения подруги')
        self.assertIsNone(event['birth_year'])
        reply = confirmed_text(self.store, self.job['id'])
        self.assertIn('12.10.', reply)
        self.assertNotIn('2026', reply)

    def test_ordinary_event_still_needs_a_title(self):
        draft = self.tool('prepare_request', action='create_event', parameters={'data': {
            'start_date': '2026-10-20', 'start_time': '12:00'}})
        self.assertEqual(draft['status'], 'waiting', draft)
        self.assertIn('data.content', draft['missing_fields'])
