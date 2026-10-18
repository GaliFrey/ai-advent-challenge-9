import json
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from personal_assistent.recurrence import Rule, RuleDraft, occurrence
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner
from personal_assistent.runtime import Runtime
from personal_assistent.agent import Agent
from personal_assistent.receipts import confirmed_text
from test_assistant import Model, FakeTelegram, answer, call, update, OWNER


def stamp(value, zone='Europe/Kirov'):
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo(zone)).timestamp()


NOW = stamp('2026-10-17T12:00:00')


class CalendarTests(unittest.TestCase):
    def rule(self, frequency, **kw):
        return Rule(frequency=frequency, start_date='2026-01-01', **kw)

    def next(self, rule, reference, previous=False, at='09:00', zone='Europe/Kirov'):
        value = occurrence(rule, at, zone, stamp(reference, zone), previous=previous)
        return datetime.fromtimestamp(value, ZoneInfo(zone)).isoformat() if value is not None else None

    def test_daily_and_weekdays_keep_wall_time(self):
        self.assertEqual(self.next(self.rule('daily'), '2026-10-17T09:01'), '2026-10-18T09:00:00+03:00')
        rule = self.rule('weekly', weekdays=[0, 1, 2, 3, 4])
        self.assertEqual(self.next(rule, '2026-10-16T09:00'), '2026-10-19T09:00:00+03:00')
        self.assertEqual(self.next(rule, '2026-10-18T22:00', previous=True), '2026-10-16T09:00:00+03:00')

    def test_month_end_does_not_drift(self):
        rule = self.rule('monthly', days=[31])
        self.assertEqual(self.next(rule, '2026-01-31T09:00'), '2026-02-28T09:00:00+03:00')
        self.assertEqual(self.next(rule, '2026-02-28T09:00'), '2026-03-31T09:00:00+03:00')
        self.assertEqual(self.next(rule, '2026-03-31T09:00'), '2026-04-30T09:00:00+03:00')

    def test_three_times_monthly_and_clamped_duplicates(self):
        rule = self.rule('monthly', days=[1, 10, 20])
        self.assertEqual(self.next(rule, '2026-01-10T09:00'), '2026-01-20T09:00:00+03:00')
        self.assertEqual(self.next(rule, '2026-01-20T09:00'), '2026-02-01T09:00:00+03:00')
        rule = self.rule('monthly', days=[28, 30, 31])
        self.assertEqual(self.next(rule, '2026-02-28T09:00'), '2026-03-28T09:00:00+03:00')

    def test_yearly_leap_day_returns_to_february_29(self):
        rule = self.rule('yearly', dates=[{'month': 2, 'day': 29}])
        self.assertEqual(self.next(rule, '2026-02-28T09:00'), '2027-02-28T09:00:00+03:00')
        self.assertEqual(self.next(rule, '2027-02-28T09:00'), '2028-02-29T09:00:00+03:00')

    def test_four_dates_yearly(self):
        rule = self.rule('yearly', dates=[{'month': m, 'day': 15} for m in (1, 4, 7, 10)])
        self.assertEqual(self.next(rule, '2026-07-15T09:00'), '2026-10-15T09:00:00+03:00')
        self.assertEqual(self.next(rule, '2026-10-15T09:00'), '2027-01-15T09:00:00+03:00')

    def test_intervals_have_a_stable_anchor(self):
        for frequency, kw, reference, expected in [
            ('daily', {}, '2026-01-02T12:00', '2026-01-04T09:00:00+03:00'),
            ('weekly', {'weekdays': [0]}, '2026-01-05T12:00', '2026-01-19T09:00:00+03:00'),
            ('monthly', {'days': [31]}, '2026-03-15T12:00', '2026-04-30T09:00:00+03:00'),
            ('yearly', {'dates': [{'month': 6, 'day': 1}]}, '2027-06-01T12:00', '2029-06-01T09:00:00+03:00'),
        ]:
            with self.subTest(frequency=frequency):
                self.assertEqual(self.next(self.rule(frequency, interval=3, **kw), reference), expected)

    def test_start_date_and_long_outage(self):
        rule = Rule(frequency='monthly', start_date='2026-10-17', days=[1, 20])
        self.assertIsNone(self.next(rule, '2026-10-16T12:00', previous=True))
        self.assertEqual(self.next(rule, '2026-01-01T12:00'), '2026-10-20T09:00:00+03:00')
        self.assertEqual(self.next(rule, '2046-12-18T12:00', previous=True), '2046-12-01T09:00:00+03:00')

    def test_clock_changes_shift_gap_and_use_first_fold(self):
        rule = Rule(frequency='daily', start_date='2026-01-01')
        self.assertEqual(self.next(rule, '2026-03-28T12:00', at='02:30', zone='Europe/Berlin'), '2026-03-29T03:30:00+02:00')
        self.assertEqual(self.next(rule, '2026-03-29T03:30', at='02:30', zone='Europe/Berlin'), '2026-03-30T02:30:00+02:00')
        self.assertEqual(self.next(rule, '2026-10-24T12:00', at='02:30', zone='Europe/Berlin'), '2026-10-25T02:30:00+02:00')

    def test_reject_invalid_or_incomplete_rules(self):
        for values in ({'frequency': 'monthly'}, {'frequency': 'weekly', 'weekdays': [7]},
                       {'frequency': 'daily', 'days': [1]}, {'frequency': 'monthly', 'days': [1, 1]},
                       {'frequency': 'daily', 'interval': 0}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Rule.model_validate({'start_date': '2026-01-01', **values})
        self.assertIsNone(RuleDraft().frequency)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.store.enqueue(update('Каждый день напоминай проверить фильтр'), NOW)
        self.job = self.store.next_job()
        self.runner = ToolRunner(self.store)
        self.counter = 0

    def tearDown(self):
        self.store.db.close()

    def tool(self, name, args=None, now=NOW, job=None):
        self.counter += 1
        return self.runner.run(call(name, args or {}, str(self.counter)), job or self.job, now)

    def prepare(self, action='create_schedule', params=None, continued=False, **kw):
        return self.tool('prepare_request', {'action': action, 'parameters': params or {}, 'continue_request': continued}, **kw)

    def create(self, **kw):
        ready = self.prepare(params={'content': 'Проверить фильтр', 'due_time': '09:00', 'recurrence': {'frequency': 'daily'}, **kw})
        self.assertEqual(ready['status'], 'ready', ready)
        done = self.tool('execute_request', {'request_id': ready['request_id']})
        self.assertEqual(done['status'], 'done', done)
        return done['result']

    def run_action(self, action, params, now=NOW):
        ready = self.prepare(action, params, now=now)
        self.assertEqual(ready['status'], 'ready', ready)
        return self.tool('execute_request', {'request_id': ready['request_id']}, now=now)

    def tick(self, at):
        self.store.prepare_reminders(at, lambda r: r['content'])

    def test_clarification_preserves_frequency_dates_and_original_source(self):
        draft = self.prepare(params={'content': 'Проверить фильтр', 'recurrence': {'frequency': 'monthly'}})
        self.assertEqual(set(draft['missing_fields']), {'days'})
        self.assertIn('По каким числам', draft['question'])
        self.assertEqual(self.tool('execute_request', {'request_id': draft['request_id']})['error']['code'], 'request_incomplete')
        ready = self.prepare(None, {'recurrence': {'days': [1, 10, 20]}, 'due_time': '10:00'}, True)
        self.assertEqual(ready['request_id'], draft['request_id'])
        done = self.tool('execute_request', {'request_id': ready['request_id']})
        self.assertEqual(done['result']['recurrence']['days'], [1, 10, 20])
        self.assertEqual(done['result']['source_message_id'], 1)
        again = self.tool('execute_request', {'request_id': ready['request_id']})
        self.assertEqual(again, done)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM schedules').fetchone()[0], 1)
        self.assertIn('Повторяющееся напоминание создано', confirmed_text(self.store, self.job['id']))

    def test_yearly_clarification_does_not_choose_dates(self):
        draft = self.prepare(params={'content': 'Проверка', 'due_time': '09:00', 'recurrence': {'frequency': 'yearly'}})
        self.assertEqual(draft['missing_fields'], ['dates'])
        self.assertEqual(draft['question'], 'В какие даты года напоминать?')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM schedules').fetchone()[0], 0)

    def test_long_schedule_text_has_progressing_pages(self):
        content = '🛠' * 4000
        dates = [{'month': m, 'day': d} for m in (1, 4, 7, 10) for d in range(1, 26)]
        schedule = self.create(content=content, recurrence={'frequency': 'yearly', 'dates': dates})
        page = self.tool('list_schedules')
        self.assertEqual(len(page['items']), 1)
        self.assertTrue(page['items'][0]['incomplete'])
        parts, offset = [], 0
        while True:
            item = self.tool('get_schedule', {'id': schedule['id'], 'content_offset': offset})
            self.assertNotIn('error', item)
            parts.append(item['content'])
            if item['next_content_offset'] is None:
                break
            self.assertGreater(item['next_content_offset'], offset)
            offset = item['next_content_offset']
        self.assertEqual(''.join(parts), content)

    def test_outage_only_creates_latest_and_next_tick_is_idempotent(self):
        schedule = self.create()
        now = stamp('2026-10-25T12:00')
        self.tick(now)
        self.tick(now)
        rows = list(self.store.db.execute('SELECT * FROM reminders'))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['due_at'], stamp('2026-10-25T09:00'))
        self.assertEqual(rows[0]['schedule_id'], schedule['id'])
        self.assertEqual(self.tool('get_schedule', {'id': schedule['id']})['next_due_at'], '2026-10-26T09:00:00+03:00')

    def test_queued_unstarted_old_occurrence_is_replaced_after_outage(self):
        self.create()
        self.tick(stamp('2026-10-18T09:00'))
        self.tick(stamp('2026-10-25T12:00'))
        rows = list(self.store.db.execute('SELECT status,due_at FROM reminders ORDER BY id'))
        self.assertEqual([r['status'] for r in rows], ['cancelled', 'queued'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM outbox').fetchone()[0], 1)

    def test_uncertain_delivery_is_preserved_and_blocks_schedule_cancellation(self):
        schedule = self.create()
        self.tick(stamp('2026-10-18T09:00'))
        out = self.store.next_outbox(stamp('2026-10-18T09:01'))
        self.store.start_delivery(out['key'])
        self.tick(stamp('2026-10-25T12:00'))
        result = self.run_action('cancel_schedule', {'id': schedule['id']}, stamp('2026-10-25T12:01'))
        self.assertEqual(result['error']['code'], 'delivery_in_progress')
        self.assertEqual(self.tool('get_schedule', {'id': schedule['id']})['status'], 'active')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM outbox').fetchone()[0], 2)

    def test_snooze_chain_preserves_link_and_does_not_move_schedule(self):
        schedule = self.create()
        now = stamp('2026-10-18T09:00')
        self.tick(now)
        out = self.store.next_outbox(now)
        self.store.sent(out['key'], 123, now)
        self.store.snooze(out['reminder_id'], now + 60)
        self.store.snooze(out['reminder_id'], now + 120)
        self.tick(now + 660)
        out = self.store.next_outbox(now + 660)
        self.store.sent(out['key'], 124, now + 660)
        self.store.snooze(out['reminder_id'], now + 700)
        rows = list(self.store.db.execute('SELECT * FROM reminders ORDER BY id'))
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['is_snooze'] for r in rows], [0, 1, 1])
        self.assertEqual(rows[2]['due_at'], now + 1300)
        self.assertTrue(all(r['schedule_id'] == schedule['id'] for r in rows))
        self.assertEqual(self.tool('get_schedule', {'id': schedule['id']})['next_due_at'], '2026-10-19T09:00:00+03:00')

    def test_cancel_removes_instances_and_prevents_snoozing_old_notification(self):
        schedule = self.create()
        now = stamp('2026-10-18T09:00')
        self.tick(now)
        out = self.store.next_outbox(now)
        self.store.sent(out['key'], 123, now)
        self.store.snooze(out['reminder_id'], now + 10)
        done = self.run_action('cancel_schedule', {'id': schedule['id']}, now + 20)
        self.assertEqual(done['status'], 'done')
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM reminders WHERE status IN ('pending','queued')").fetchone()[0], 0)
        # A different delivered occurrence may still have an unused button.
        self.store.db.execute('UPDATE reminders SET snoozed_to=NULL WHERE id=?', (out['reminder_id'],))
        self.assertIn('отменено', self.store.snooze(out['reminder_id'], now + 30))
        self.tick(stamp('2026-11-01T12:00'))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 2)
        self.assertIn('Повторяющееся напоминание отменено', confirmed_text(self.store, self.job['id']))

    def test_update_only_time_keeps_rule_and_timezone(self):
        schedule = self.create(recurrence={'frequency': 'monthly', 'days': [1, 10, 20]})
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        done = self.run_action('update_schedule', {'id': schedule['id'], 'due_time': '10:00'})
        self.assertEqual(done['result']['due_time'], '10:00:00')
        self.assertEqual(done['result']['timezone'], 'Europe/Kirov')
        self.assertEqual(done['result']['recurrence']['days'], [1, 10, 20])
        self.assertIn('Europe/Kirov', confirmed_text(self.store, self.job['id']))

    def test_update_after_ambiguous_target_selection_inherits_its_rule(self):
        schedule = self.create(recurrence={'frequency': 'monthly', 'interval': 3, 'days': [1, 10, 20]})
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        draft = self.prepare('update_schedule', {'due_time': '10:00'})
        self.assertIn('id', draft['missing_fields'])
        self.assertNotIn('timezone', draft['parameters'])
        ready = self.prepare(None, {'id': schedule['id']}, True)
        self.assertEqual(ready['status'], 'ready', ready)
        result = self.tool('execute_request', {'request_id': ready['request_id']})['result']
        self.assertEqual(result['recurrence']['days'], [1, 10, 20])
        self.assertEqual(result['recurrence']['interval'], 3)
        self.assertEqual(result['timezone'], 'Europe/Kirov')

    def test_partial_rule_update_before_target_selection_merges_explicit_changes(self):
        schedule = self.create(recurrence={'frequency': 'monthly', 'interval': 3, 'days': [1, 10, 20]})
        self.prepare('update_schedule', {'recurrence': {'days': [5, 15, 25]}})
        ready = self.prepare(None, {'id': schedule['id']}, True)
        self.assertEqual(ready['status'], 'ready', ready)
        result = self.tool('execute_request', {'request_id': ready['request_id']})['result']
        self.assertEqual(result['recurrence']['days'], [5, 15, 25])
        self.assertEqual(result['recurrence']['interval'], 3)
        self.assertEqual(result['due_time'], '09:00:00')

    def test_changed_frequency_removes_previous_days_and_interval(self):
        schedule = self.create(recurrence={'frequency': 'monthly', 'interval': 3, 'days': [31]})
        result = self.run_action('update_schedule', {'id': schedule['id'], 'recurrence': {'frequency': 'weekly', 'weekdays': [0, 4]}})['result']
        self.assertEqual(result['recurrence']['frequency'], 'weekly')
        self.assertEqual(result['recurrence']['interval'], 1)
        self.assertEqual(result['recurrence']['days'], [])
        self.assertEqual(result['recurrence']['weekdays'], [0, 4])

    def test_skip_one_future_occurrence_and_replay(self):
        schedule = self.create()
        ready = self.prepare('skip_schedule', {'id': schedule['id']})
        first = self.tool('execute_request', {'request_id': ready['request_id']})
        again = self.tool('execute_request', {'request_id': ready['request_id']})
        self.assertEqual(first, again)
        self.assertEqual(first['result']['next_due_at'], '2026-10-19T09:00:00+03:00')
        self.tick(stamp('2026-10-18T12:00'))
        self.assertEqual(tuple(self.store.db.execute('SELECT status,reason FROM reminders').fetchone()),('cancelled','user_skipped'))
        self.tick(stamp('2026-10-19T09:00'))
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM reminders WHERE status='queued'").fetchone()[0], 1)
        self.assertIn('Срабатывание пропущено', confirmed_text(self.store, self.job['id']))

    def test_skip_queued_occurrence_keeps_next_calendar_date(self):
        schedule = self.create()
        now = stamp('2026-10-18T09:00')
        self.tick(now)
        done = self.run_action('skip_schedule', {'id': schedule['id']}, now + 10)
        self.assertEqual(done['result']['next_due_at'], '2026-10-19T09:00:00+03:00')
        self.assertIsNone(self.store.next_outbox(now + 10))
        self.tick(now + 20)
        self.assertEqual(self.store.db.execute('SELECT status FROM reminders').fetchone()[0], 'cancelled')

    def test_one_explicitly_moved_occurrence_survives_next_calendar_tick(self):
        self.create()
        now = stamp('2026-10-18T09:00')
        self.tick(now)
        rid = self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.tool('update_reminder', {'id': rid, 'content': 'Проверить фильтр позже', 'due_at': '2026-10-21T10:00:00+03:00'}, now=now)
        self.tick(stamp('2026-10-19T09:00'))
        rows = list(self.store.db.execute('SELECT * FROM reminders ORDER BY id'))
        self.assertEqual(rows[0]['status'], 'pending')
        self.assertEqual(rows[0]['due_at'], stamp('2026-10-21T10:00'))
        self.assertEqual(rows[1]['status'], 'queued')

    def test_schedule_update_removes_old_queue_but_preserves_original_anchor(self):
        schedule = self.create()
        now = stamp('2026-10-18T09:00')
        self.tick(now)
        done = self.run_action('update_schedule', {'id': schedule['id'], 'due_time': '10:00'}, now + 10)
        self.assertEqual(done['result']['next_due_at'], '2026-10-18T10:00:00+03:00')
        self.assertEqual(done['result']['recurrence']['start_date'], '2026-10-17')
        self.assertIsNone(self.store.next_outbox(now + 10))
        self.tick(stamp('2026-10-18T10:00'))
        # Updating the time on the same calendar day must produce its new slot.
        rows = list(self.store.db.execute('SELECT status,due_at FROM reminders ORDER BY id'))
        self.assertEqual([r['status'] for r in rows], ['cancelled', 'queued'])
        self.assertEqual(rows[1]['due_at'], stamp('2026-10-18T10:00'))

    def test_timezone_is_fixed_for_delivery_after_profile_change(self):
        self.create()
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        runtime = Runtime(self.store, None, None, OWNER, clock=lambda: stamp('2026-10-18T09:00'))
        runtime.schedule_once()
        text = self.store.next_outbox(stamp('2026-10-18T09:00'))['text']
        self.assertIn('09:00:00+03:00', text)
        self.assertNotIn('+09:00', text)

    def test_moving_one_instance_confirms_its_fixed_timezone(self):
        self.create()
        now = stamp('2026-10-18T09:00')
        self.tick(now)
        self.store.set_meta('profile', {'timezone': 'Asia/Tokyo'})
        rid = self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.tool('update_reminder', {'id': rid, 'content': 'Фильтр', 'due_at': '2026-10-18T10:00:00+03:00'}, now=now)
        text = confirmed_text(self.store, self.job['id'])
        self.assertIn('18.10.2026 10:00', text)
        self.assertNotIn('Asia/Tokyo', text)

    def test_execution_rejects_schedule_changed_since_preparation(self):
        schedule = self.create()
        ready = self.prepare('cancel_schedule', {'id': schedule['id']})
        self.tick(stamp('2026-10-18T09:00'))
        done = self.tool('execute_request', {'request_id': ready['request_id']}, now=stamp('2026-10-18T09:01'))
        self.assertEqual(done['error']['code'], 'request_stale')
        self.assertEqual(self.tool('get_schedule', {'id': schedule['id']})['status'], 'active')

    def test_cancel_one_occurrence_does_not_cancel_series(self):
        schedule = self.create()
        now = stamp('2026-10-18T09:00')
        self.tick(now)
        rid = self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.tool('cancel_reminder', {'id': rid}, now=now)
        self.tick(stamp('2026-10-19T09:00'))
        self.assertEqual(self.tool('get_schedule', {'id': schedule['id']})['status'], 'active')
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM reminders WHERE status='queued'").fetchone()[0], 1)

    def test_failed_materialization_rolls_back_cursor_and_occurrence(self):
        schedule = self.create()
        with patch.object(self.store, 'queue_message', side_effect=RuntimeError('crash')):
            with self.assertRaises(RuntimeError):
                self.tick(stamp('2026-10-18T09:00'))
        self.assertEqual(self.tool('get_schedule', {'id': schedule['id']})['next_due_at'], schedule['next_due_at'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
        self.tick(stamp('2026-10-18T09:00'))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 1)

    def test_numbered_choice_is_typed_and_retained(self):
        first = self.create()
        second = self.create(content='Проверить масло')
        choices = [{'record_type': 'schedule', 'id': r['id']} for r in (first, second)]
        result = self.tool('ask_clarification', {'question': 'Какое напоминание отменить?', 'kind': 'other', 'choices': choices})
        self.assertIn('каждый день', result['question'])
        job = {**self.job, 'payload': json.dumps(update('2')), 'pending': {'selection': choices[1]}}
        bad = self.prepare('cancel_schedule', {'id': first['id']}, job=job)
        self.assertEqual(bad['error']['code'], 'selection_mismatch')

    def test_migration_and_restart_keep_oneoffs_and_calendar_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assistant.db'
            db = sqlite3.connect(path)
            db.execute('CREATE TABLE reminders(id INTEGER PRIMARY KEY,content TEXT NOT NULL,due_at REAL NOT NULL,created_at REAL NOT NULL,status TEXT NOT NULL DEFAULT \'pending\',source_message_id INTEGER NOT NULL,snoozed_to INTEGER)')
            db.execute("INSERT INTO reminders VALUES(1,'Старое',?,?, 'pending',1,NULL)", (NOW + 20, NOW))
            db.commit()
            db.close()
            store = Store(path)
            self.assertIsNone(store.db.execute('SELECT schedule_id FROM reminders').fetchone()[0])
            original = self.store
            self.store = store
            self.runner = ToolRunner(store)
            store.enqueue(update('Каждый день'), NOW)
            self.job = store.next_job()
            self.create()
            self.tick(stamp('2026-10-18T12:00'))
            count = store.db.execute('SELECT count(*) FROM reminders').fetchone()[0]
            store.db.close()
            self.store = Store(path)
            self.runner = ToolRunner(self.store)
            self.tick(stamp('2026-10-18T12:00'))
            self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], count)
            self.store.db.close()
            self.store = original


class ScheduleAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_draft_followup_execute_delivery_without_real_services(self):
        store = Store(':memory:')
        self.addCleanup(store.db.close)
        model = Model(answer(call('prepare_request', {'action': 'create_schedule', 'parameters': {
            'content': 'Проверить фильтр', 'recurrence': {'frequency': 'monthly'}, 'due_time': '09:00'}}, 'draft')),
            answer(call('prepare_request', {'continue_request': True, 'parameters': {'recurrence': {'days': [1, 10, 20]}}}, 'dates')),
            answer(call('execute_request', {'request_id': 1}, 'execute')),
            answer(text='Сделано'))
        clock = [NOW]
        tg = FakeTelegram()
        runtime = Runtime(store, tg, Agent(store, model, clock=lambda: clock[0]), OWNER, clock=lambda: clock[0])
        store.enqueue(update('Три раза в месяц в 9 напоминай проверить фильтр'), NOW)
        await runtime.process_once()
        self.assertIn('По каким числам', store.next_outbox(NOW)['text'])
        store.enqueue(update('1, 10 и 20 числа', 2), NOW)
        await runtime.process_once()
        self.assertEqual(store.db.execute('SELECT count(*) FROM schedules').fetchone()[0], 1)
        clock[0] = stamp('2026-10-20T09:00')
        runtime.schedule_once()
        while await runtime.deliver_once():
            pass
        self.assertTrue(any('⏰ Напоминание' in str(item) for item in tg.sent))
        self.assertEqual(store.db.execute('SELECT status FROM reminders').fetchone()[0], 'sent')
