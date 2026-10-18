import json
import sqlite3
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

from personal_assistent.events import Events, EventData, EventError, ReminderRuleData
from personal_assistent.event_schema import migrate
from personal_assistent.recurrence import Rule, iter_dates, occurrence
from personal_assistent.storage import Store, dump
from personal_assistent.tools import ToolRunner
from personal_assistent.receipts import confirmed_text


def ts(value):
    return datetime.fromisoformat(value if '+' in value or value.endswith('Z') else value+'+03:00').timestamp()


NOW=ts('2026-10-17T10:00')


class EventTests(unittest.TestCase):
    def setUp(self):
        self.store=Store(':memory:')
        self.e=Events(self.store)
        self.addCleanup(self.store.db.close)
        self.db=self.store.db

    def create(self, **kwargs):
        with self.store.atomic():
            return self.e.create(EventData.model_validate({'content':'Встреча','start_date':'2026-10-20','start_time':'12:00',**kwargs}),NOW,1)

    def occs(self, event):
        return [dict(r) for r in self.db.execute('SELECT * FROM event_occurrences WHERE event_id=? ORDER BY position',(event['id'],))]

    def rule(self, event, now=NOW, scope='future', **kwargs):
        controls={k:kwargs.pop(k) for k in ('occurrence_id','from_position','replace_rule_id') if k in kwargs}
        if scope=='future':
            controls.setdefault('from_position',0)
        with self.store.atomic():
            return self.e.add_rule(event['id'],ReminderRuleData.model_validate({'kind':'calendar','offset':1,**kwargs}),now,scope=scope,**controls)

    def change(self,event, operation='move', scope='one', data=None, **kwargs):
        if scope=='one':
            kwargs.setdefault('occurrence_id',self.occs(event)[0]['id'])
        else:
            kwargs.setdefault('from_position',0)
        with self.store.atomic():
            return self.e.change(event['id'],operation,NOW,scope=scope,data=EventData.model_validate(data) if data else None,**kwargs)

    def tick(self, now):
        self.store.prepare_reminders(ts(now),lambda r:r['content'])

    def test_event_without_notification_and_no_note(self):
        event=self.create()
        self.assertEqual(len(self.occs(event)),1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM reminders').fetchone()[0],0)
        self.assertEqual(self.db.execute('SELECT count(*) FROM notes').fetchone()[0],0)

    def test_meeting_requires_hour_all_day_has_no_implicit_hour(self):
        with self.assertRaises(ValueError):
            self.create(start_time=None)
        self.create(start_time=None,all_day=True)

    def test_calendar_day_differs_from_24_hours_across_dst(self):
        event=self.create(start_date='2026-03-29',start_time='10:00',timezone='Europe/Berlin')
        # Rules intentionally created before these deadlines.
        now=ts('2026-03-01T10:00')
        self.rule(event,now=now,kind='calendar',offset=1,due_time='10:00')
        self.rule(event,now=now,kind='interval',offset=86400)
        due=[r[0] for r in self.db.execute('SELECT due_at FROM reminders ORDER BY id')]
        self.assertEqual(abs(due[0]-due[1]),3600)

    def test_all_day_exact_interval_requires_reference(self):
        event=self.create(start_time=None,all_day=True)
        with self.assertRaises(EventError) as caught:
            self.rule(event,kind='interval',offset=3600)
        self.assertEqual(caught.exception.code,'all_day_reference_required')

    def test_move_one_stable_id_relative_changes_fixed_stays(self):
        event=self.create()
        occ=self.occs(event)[0]
        self.rule(event,kind='interval',offset=3600)
        self.rule(event,scope='one',occurrence_id=occ['id'],kind='fixed',fixed_due_at='2026-10-19T08:00+03:00',offset=0)
        before=[dict(r) for r in self.db.execute('SELECT * FROM reminders ORDER BY id')]
        self.change(event,data={'content':'Встреча','start_date':'2026-10-21','start_time':'14:00'})
        after=[dict(r) for r in self.db.execute('SELECT * FROM reminders ORDER BY id')]
        self.assertEqual([r['id'] for r in before],[r['id'] for r in after])
        self.assertEqual(after[0]['due_at'],ts('2026-10-21T13:00'))
        self.assertEqual(after[1]['due_at'],before[1]['due_at'])
        self.assertEqual(self.occs(event)[0]['id'],occ['id'])
        self.assertEqual(self.occs(event)[0]['original_date'],'2026-10-20')

    def test_fixed_after_moved_event_is_cancelled_with_warning(self):
        event=self.create(); occ=self.occs(event)[0]
        self.rule(event,scope='one',occurrence_id=occ['id'],kind='fixed',fixed_due_at='2026-10-20T10:00+03:00')
        result=self.change(event,data={'content':'Встреча','start_date':'2026-10-19','start_time':'09:00'})
        self.assertTrue(result['warnings'])
        self.assertEqual(self.db.execute('SELECT reason FROM reminders').fetchone()[0],'after_event')

    def test_replace_one_does_not_replace_series(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':3})
        original=self.rule(event)
        self.rule(event,scope='one',occurrence_id=self.occs(event)[0]['id'],kind='interval',offset=3600,replace_rule_id=original['rule_id'])
        self.assertEqual(self.db.execute("SELECT count(*) FROM reminders WHERE event_rule_id=? AND status='pending'",(original['rule_id'],)).fetchone()[0],2)

    def test_move_nearest_does_not_shift_future(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':3})
        self.change(event,data={'content':'Встреча','start_date':'2026-10-22','start_time':'15:00'})
        self.assertEqual([r['start_date'] for r in self.occs(event)],['2026-10-22','2026-10-21','2026-10-22'])

    def test_future_move_preserves_history_manual_exception_and_count(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':4})
        before=self.occs(event)
        self.change(event,occurrence_id=before[2]['id'],data={'content':'Встреча','start_date':'2026-10-28','start_time':'15:00'})
        self.change(event,scope='future',from_position=1,data={'content':'Встреча','start_date':'2026-10-25','start_time':'10:00','recurrence':{'frequency':'daily','start_date':'2026-10-25'}})
        after=self.occs(event)
        self.assertEqual(len(after),4)
        self.assertEqual([r['id'] for r in before],[r['id'] for r in after])
        self.assertEqual([r['start_date'] for r in after],['2026-10-20','2026-10-25','2026-10-28','2026-10-27'])
        self.assertEqual(self.db.execute('SELECT count(*) FROM event_segments').fetchone()[0],2)

    def test_cancelled_position_still_counts(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':2})
        self.change(event,operation='cancel')
        self.e.materialize(event['id'],NOW,date(2028,1,1))
        self.assertEqual(len(self.occs(event)),2)
        self.assertEqual(self.occs(event)[0]['status'],'cancelled')

    def test_group_same_occurrence_distinct_rules_different_events_separate(self):
        first=self.create();second=self.create(content='Другая встреча')
        self.rule(first);self.rule(first);self.rule(second)
        self.tick('2026-10-19T09:00')
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],2)
        self.assertEqual(self.db.execute('SELECT count(*) FROM notification_members').fetchone()[0],3)
        row=self.store.next_outbox(ts('2026-10-19T09:00'))
        self.store.sent(row['key'],123,ts('2026-10-19T09:01'))
        self.assertEqual(self.db.execute("SELECT count(*) FROM reminders WHERE status='sent'").fetchone()[0],2)

    def test_started_group_blocks_mutation_and_rolls_back(self):
        event=self.create();self.rule(event);self.rule(event)
        self.tick('2026-10-19T09:00')
        row=self.store.next_outbox(ts('2026-10-19T09:00'));self.store.start_delivery(row['key'])
        with self.assertRaises(EventError):
            self.change(event,operation='cancel')
        self.assertEqual(self.occs(event)[0]['status'],'active')
        self.assertIsNotNone(self.store.next_outbox(ts('2026-10-19T09:00')))

    def test_recovery_latest_before_start_only(self):
        event=self.create();self.rule(event,offset=2);self.rule(event,offset=1)
        self.tick('2026-10-19T10:00')
        states=[tuple(r) for r in self.db.execute('SELECT status,reason FROM reminders ORDER BY due_at')]
        self.assertEqual(states,[('cancelled','superseded'),('queued',None)])
        self.tick('2026-10-20T12:00')
        self.assertIsNone(self.store.next_outbox(ts('2026-10-20T12:00')))

    def test_all_day_recovery_until_end_of_local_day(self):
        event=self.create(start_time=None,all_day=True)
        self.rule(event,offset=0)
        self.tick('2026-10-20T23:59')
        self.assertIsNotNone(self.store.next_outbox(ts('2026-10-20T23:59')))
        self.tick('2026-10-21T00:00')
        self.assertIsNone(self.store.next_outbox(ts('2026-10-21T00:00')))

    def test_rule_created_after_deadline_no_immediate_delivery(self):
        event=self.create()
        result=self.rule(event,offset=10)
        self.assertTrue(result['warnings'])
        self.tick('2026-10-19T10:00')
        self.assertIsNone(self.store.next_outbox(ts('2026-10-19T10:00')))

    def test_snooze_survives_move_and_repeated_button_is_safe(self):
        event=self.create();self.rule(event)
        self.tick('2026-10-19T09:00');row=self.store.next_outbox(ts('2026-10-19T09:00'));self.store.sent(row['key'],1,NOW)
        identity=row['reminder_id']
        self.assertEqual(self.store.snooze(identity,ts('2026-10-19T09:00')),'Отложено на 10 минут.')
        self.assertIn('уже отложено',self.store.snooze(identity,ts('2026-10-19T09:01')))
        self.change(event,data={'content':'Встреча','start_date':'2026-10-21','start_time':'12:00'})
        child=self.db.execute('SELECT * FROM reminders WHERE parent_reminder_id=?',(identity,)).fetchone()
        self.assertEqual(child['due_at'],ts('2026-10-19T09:10'))
        self.assertEqual(child['status'],'pending')

    def test_snooze_after_event_not_created_and_disable_cancels_existing(self):
        event=self.create();self.rule(event)
        self.tick('2026-10-19T09:00');row=self.store.next_outbox(ts('2026-10-19T09:00'));self.store.sent(row['key'],1,NOW)
        self.assertIn('после начала',self.store.snooze(row['reminder_id'],ts('2026-10-20T11:55')))
        self.store.snooze(row['reminder_id'],ts('2026-10-19T09:00'))
        self.change(event,operation='disable',scope='future')
        self.assertEqual(self.db.execute("SELECT count(*) FROM reminders WHERE status='pending'").fetchone()[0],0)
        self.assertEqual(self.e.event(event['id'])['deleted'],0)

    def test_profile_time_only_unsent_default_and_zone_pinned(self):
        event=self.create();self.rule(event);self.rule(event,due_time='08:00')
        self.store.set_meta('profile',{'notification_time':'11:00','timezone':'Asia/Tokyo'})
        self.e.profile_changed(NOW)
        rows=list(self.db.execute('SELECT due_at FROM reminders ORDER BY id'))
        self.assertEqual(rows[0][0],ts('2026-10-19T11:00'))
        self.assertEqual(rows[1][0],ts('2026-10-19T08:00'))
        self.assertEqual(self.e.event(event['id'])['timezone'],'Europe/Kirov')

    def test_profile_time_in_past_skipped_with_warning(self):
        event=self.create();self.rule(event)
        self.store.set_meta('profile',{'notification_time':'07:00'})
        self.assertTrue(self.e.profile_changed(ts('2026-10-19T08:00')))
        self.assertEqual(self.db.execute('SELECT reason FROM reminders').fetchone()[0],'profile_time_in_past')

    def test_birthday_year_metadata_no_historical_generation_duplicate_and_ambiguity(self):
        event=self.create(content='День рождения Оли',kind='birthday',person='Оля',start_date='1990-10-20',start_time=None,birth_year=1990)
        self.assertEqual(self.occs(event)[0]['start_date'],'2026-10-20')
        duplicate=self.create(content='День рождения Оли',kind='birthday',person='Оля',start_date='1990-10-20',start_time=None,birth_year=1990)
        self.assertEqual(event['id'],duplicate['id'])
        with self.assertRaises(EventError):
            self.create(kind='birthday',person='Оля',start_date='2026-11-20',start_time=None)
        self.rule(event)
        self.assertIn('36 лет',self.db.execute('SELECT content FROM reminders ORDER BY id LIMIT 1').fetchone()[0])

    def test_digest_optin_includes_disabled_birthdays_and_no_replay(self):
        event=self.create(kind='birthday',person='Оля',start_date='2026-10-10',start_time=None)
        self.e.digest(NOW);self.assertEqual(self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],0)
        self.change(event,operation='disable',scope='future')
        self.store.set_meta('profile',{'birthday_digest':True})
        self.e.digest(NOW,immediate=True)
        row=self.store.next_outbox(NOW);self.assertIn('уже прошёл',row['text'])
        self.store.sent(row['key'],1,NOW)
        self.assertEqual(self.db.execute('SELECT state FROM digest_runs').fetchone()[0],'sent')
        self.store.set_meta('profile',{'birthday_digest':False});self.e.digest(NOW)
        self.store.set_meta('profile',{'birthday_digest':True});self.e.digest(NOW,immediate=True)
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)

    def test_empty_digest_recorded_no_past_month_catchup_and_deleted_excluded(self):
        event=self.create(kind='birthday',person='Оля',start_time=None)
        self.change(event,operation='delete',scope='future')
        self.store.set_meta('profile',{'birthday_digest':True})
        self.e.digest(ts('2026-11-15T10:00'))
        self.assertEqual(tuple(self.db.execute('SELECT month,state FROM digest_runs').fetchone()),(11,'empty'))
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],0)
        self.assertEqual(self.e.calendar(date(2026,10,1),date(2026,10,31),NOW),[])

    def test_database_rejects_mixed_event_sources(self):
        event=self.create();self.rule(event)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute('UPDATE reminders SET standalone_id=123')


class CalendarRuleTests(unittest.TestCase):
    def test_ordinal_weekdays_skip_absent_fifth_and_limits(self):
        rule=Rule(frequency='monthly',start_date='2026-01-01',ordinal_weekdays=[{'ordinal':5,'weekday':0}],count=3)
        self.assertEqual([d.isoformat() for _,d in iter_dates(rule)],['2026-03-30','2026-06-29','2026-08-31'])
        last=Rule(frequency='monthly',start_date='2026-01-01',ordinal_weekdays=[{'ordinal':-1,'weekday':4}],until='2026-02-27')
        self.assertEqual([d.isoformat() for _,d in iter_dates(last)],['2026-01-30','2026-02-27'])

    def test_clamp_leap_dates_and_dedupe_and_inclusive_until(self):
        rule=Rule(frequency='yearly',start_date='2024-02-29',dates=[{'month':2,'day':29},{'month':2,'day':28}],until='2025-02-28')
        self.assertEqual([d.isoformat() for _,d in iter_dates(rule)],['2024-02-29','2025-02-28'])
        rule=Rule(frequency='monthly',start_date='2026-01-31',days=[30,31],count=3)
        self.assertEqual([d.isoformat() for _,d in iter_dates(rule)],['2026-01-31','2026-02-28','2026-03-30'])
        self.assertIsNone(occurrence(rule,'09:00','UTC',ts('2030-01-01T00:00')))


class EventToolTests(unittest.TestCase):
    def setUp(self):
        self.store=Store(':memory:');self.addCleanup(self.store.db.close)
        self.runner=ToolRunner(self.store);self.n=0
        self.job={'id':1,'payload':dump({'message':{'text':'Запомни событие','message_id':1,'date':NOW}})}

    def tool(self,name,**args):
        self.n+=1
        return self.runner.run({'id':str(self.n),'function':{'name':name,'arguments':dump(args)}},self.job,NOW)

    def execute(self,action,**params):
        draft=self.tool('prepare_request',action=action,parameters=params)
        self.assertEqual(draft.get('status'),'ready',draft)
        return self.tool('execute_request',request_id=draft['request_id'])

    def test_incomplete_event_persisted_hour_followup_and_replay_receipt(self):
        draft=self.tool('prepare_request',action='create_event',parameters={'data':{'content':'Встреча','start_date':'2026-10-20'}})
        self.assertEqual(draft['missing_fields'],['data.start_time'])
        self.job['pending']=self.store.get_meta('pending')
        ready=self.tool('prepare_request',continue_request=True,parameters={'data':{'start_time':'12:00'}})
        result=self.tool('execute_request',request_id=ready['request_id'])
        self.assertEqual(result['status'],'done')
        self.assertIn('Событие сохранено',confirmed_text(self.store,1))
        self.tool('execute_request',request_id=ready['request_id'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM events').fetchone()[0],1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0],0)

    def test_rule_requires_scope_and_ambiguous_target_not_chosen(self):
        event=self.execute('create_event',data={'content':'Встреча','start_date':'2026-10-20','start_time':'12:00'})['result']
        draft=self.tool('prepare_request',action='set_event_rule',parameters={'id':event['id'],'rule':{'kind':'calendar','offset':1}})
        self.assertEqual(draft['missing_fields'],['scope'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0],0)

    def test_calendar_and_search_birthdays_without_notes(self):
        self.execute('create_event',data={'content':'День рождения Оли','kind':'birthday','person':'Оля','start_date':'2026-10-20'})
        self.assertEqual(self.tool('search_events',query='Оля')['total'],1)
        page=self.tool('calendar',start='2026-10-01',end='2026-10-31')
        self.assertEqual(page['total'],1)
        self.assertEqual(page['items'][0]['notifications'],[])

    def test_elapsed_standalone_from_message_sent_with_original_due(self):
        self.job['payload']=dump({'message':{'text':'Через 15 минут','message_id':1,'date':NOW-3600}})
        result=self.execute('create_reminder',content='Вода',after_seconds=900)
        row=self.store.db.execute('SELECT * FROM reminders').fetchone()
        self.assertEqual(row['due_at'],NOW-2700)
        self.store.prepare_reminders(NOW,lambda r:str(r['due_at']))
        self.assertEqual(self.store.next_outbox(NOW)['text'],str(NOW-2700))
        self.assertEqual(result['status'],'done')

    def test_default_standalone_recurring_dynamic_and_explicit_unchanged(self):
        first=self.execute('create_schedule',content='Вода',recurrence={'frequency':'daily'})
        self.execute('create_schedule',content='Фильтр',recurrence={'frequency':'daily'},due_time='08:00')
        self.assertEqual(first['status'],'done')
        self.store.set_meta('profile',{'notification_time':'11:00'})
        Events(self.store).profile_changed(NOW)
        rows=list(self.store.db.execute('SELECT next_due_at,time_mode FROM schedules ORDER BY id'))
        self.assertEqual(rows[0]['next_due_at'],ts('2026-10-17T11:00'))
        self.assertEqual(rows[0]['time_mode'],'default')
        self.assertEqual(rows[1]['next_due_at'],ts('2026-10-18T08:00'))

    def test_default_oneoff_tracks_time_but_not_profile_zone(self):
        self.execute('create_reminder',content='Вода',due_date='2026-10-20')
        self.store.set_meta('profile',{'notification_time':'11:00','timezone':'Asia/Tokyo'})
        Events(self.store).profile_changed(NOW)
        row=self.store.db.execute('SELECT * FROM reminders').fetchone()
        self.assertEqual(row['due_at'],ts('2026-10-20T11:00'))

    def test_action_receipt_failure_rolls_back_event_and_request(self):
        ready=self.tool('prepare_request',action='create_event',parameters={'data':{'content':'Встреча','start_date':'2026-10-20','start_time':'12:00'}})
        with patch.object(self.store,'record_receipt',side_effect=RuntimeError('crash')):
            with self.assertRaises(RuntimeError):
                self.tool('execute_request',request_id=ready['request_id'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM events').fetchone()[0],0)
        self.assertEqual(self.store.db.execute('SELECT status FROM requests').fetchone()[0],'ready')


class MigrationTests(unittest.TestCase):
    def legacy(self,path):
        store=Store.__new__(Store)
        store.db=sqlite3.connect(path)
        store.db.row_factory=sqlite3.Row
        store.db.execute('PRAGMA foreign_keys=ON')
        store.db.executescript((Path(__file__).parent/'fixtures'/'legacy_calendar.sql').read_text())
        return store

    def test_all_states_queue_ids_callback_cache_and_reopen_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.sqlite';old=self.legacy(path)
            rule=Rule(frequency='daily',start_date='2026-10-01')
            old.db.execute("INSERT INTO schedules VALUES(7,'Вода',?,'08:00','Europe/Berlin',?,'active',1,1,1)",(rule.model_dump_json(),NOW+100))
            for i,state in enumerate(['pending','queued','sent','cancelled'],1):
                old.db.execute('INSERT INTO reminders(id,content,due_at,created_at,status,source_message_id,schedule_id,occurrence_at,is_snooze) VALUES(?,?,?,?,?,?,?,?,?)',(i,'Текст',NOW+i*100,1,state,1,7 if i<3 else None,NOW+i*50 if i<3 else None,0))
            old.db.execute('UPDATE reminders SET snoozed_to=4 WHERE id=3')
            old.queue_message('legacy','Часть',2,now=NOW)
            key=old.db.execute('SELECT key FROM outbox').fetchone()[0]
            old.db.execute('UPDATE outbox SET delivery_started=1,attempts=3 WHERE key=?',(key,))
            old.db.execute("INSERT INTO tool_results VALUES('1:call','{\"id\":3}')")
            old.db.execute("INSERT INTO notes(content,original,event_date,created_at,updated_at,source_message_id) VALUES('Факт','Факт','2000-10-20',1,1,1)")
            before=[dict(r) for r in old.db.execute('SELECT * FROM reminders ORDER BY id')]
            queue=[dict(r) for r in old.db.execute('SELECT * FROM outbox')]
            old.db.commit();old.db.close()
            for _ in range(2):
                new=Store(path)
                for original in before:
                    actual=dict(new.db.execute('SELECT * FROM reminders WHERE id=?',(original['id'],)).fetchone())
                    self.assertEqual({k:actual[k] for k in original},original)
                self.assertEqual([dict(r) for r in new.db.execute('SELECT * FROM outbox')],queue)
                self.assertEqual(new.db.execute('SELECT count(*) FROM events').fetchone()[0],0)
                self.assertEqual(tuple(new.db.execute('SELECT due_time,timezone,next_due_at,time_mode FROM schedules WHERE id=7').fetchone()),('08:00','Europe/Berlin',NOW+100,'explicit'))
                self.assertEqual(new.db.execute('SELECT parent_reminder_id FROM reminders WHERE id=4').fetchone()[0],3)
                self.assertEqual(new.db.execute('PRAGMA foreign_key_check').fetchall(),[])
                new.db.close()

    def test_fault_in_migration_rolls_back_schema_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.sqlite';old=self.legacy(path)
            old.db.execute("INSERT INTO reminders(content,due_at,created_at,source_message_id) VALUES('Вода',100,1,1)")
            old.db.commit();before=[tuple(r) for r in old.db.execute('SELECT * FROM reminders')]
            original=old.set_meta
            def fail(key,value):
                if key=='events_schema_version':
                    raise RuntimeError('fault')
                original(key,value)
            with patch.object(old,'set_meta',side_effect=fail):
                with self.assertRaises(RuntimeError):
                    migrate(old)
            self.assertEqual([tuple(r) for r in old.db.execute('SELECT * FROM reminders')],before)
            self.assertIsNone(old.db.execute("SELECT name FROM sqlite_master WHERE name='events'").fetchone())
            old.db.close()
            new=Store(path);new.db.close()

class IntegrityReviewTests(unittest.TestCase):
    setUp=EventTests.setUp
    create=EventTests.create
    occs=EventTests.occs
    rule=EventTests.rule
    change=EventTests.change
    tick=EventTests.tick
    def test_group_snooze_keeps_all_rules_and_chain_replay(self):
        event=self.create();self.rule(event);self.rule(event)
        self.tick('2026-10-19T09:00')
        queued=self.store.next_outbox(ts('2026-10-19T09:00'))
        self.store.sent(queued['key'],1,ts('2026-10-19T09:00'))
        self.store.snooze(queued['reminder_id'],ts('2026-10-19T09:01'))
        children=[dict(r) for r in self.db.execute('SELECT * FROM reminders WHERE is_snooze=1')]
        self.assertEqual(len(children),2)
        self.assertEqual(len({r['event_rule_id'] for r in children}),2)
        self.store.snooze(queued['reminder_id'],ts('2026-10-19T09:02'))
        self.assertEqual(self.db.execute('SELECT count(*) FROM reminders').fetchone()[0],4)
        self.tick('2026-10-19T09:11');second=self.store.next_outbox(ts('2026-10-19T09:11'))
        self.store.sent(second['key'],2,ts('2026-10-19T09:11'))
        self.store.snooze(second['reminder_id'],ts('2026-10-19T09:12'))
        self.assertEqual(self.db.execute('SELECT count(*) FROM reminders WHERE is_snooze=1').fetchone()[0],4)

    def test_disabling_one_rule_preserves_other_unstarted_group_member(self):
        event=self.create();first=self.rule(event);self.rule(event)
        self.tick('2026-10-19T09:00')
        self.rule(event,kind='interval',offset=3600,replace_rule_id=first['rule_id'])
        # Replacement dissolves the group; surviving old rule will queue again.
        self.tick('2026-10-19T09:01')
        pending=[dict(r) for r in self.db.execute("SELECT * FROM reminders WHERE status='queued'")]
        self.assertEqual(len(pending),1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox WHERE sent_at IS NULL').fetchone()[0],1)

    def test_database_rejects_detaching_or_cross_linking_sources(self):
        event=self.create();self.rule(event)
        other=self.create(content='Другое')
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute('UPDATE reminders SET event_rule_id=NULL,event_occurrence_id=NULL')
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute('UPDATE reminders SET event_occurrence_id=?',(self.occs(other)[0]['id'],))

    def test_future_cancellation_includes_manually_moved_instance_and_snooze(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':3})
        self.rule(event);self.tick('2026-10-19T09:00')
        queued=self.store.next_outbox(ts('2026-10-19T09:00'));self.store.sent(queued['key'],1,NOW)
        self.store.snooze(queued['reminder_id'],ts('2026-10-19T09:00'))
        self.change(event,occurrence_id=self.occs(event)[1]['id'],data={'content':'Встреча','start_date':'2026-10-25','start_time':'12:00'})
        self.change(event,operation='cancel',scope='future')
        self.assertEqual({r['status'] for r in self.occs(event)},{'cancelled'})
        self.assertEqual(self.db.execute("SELECT count(*) FROM reminders WHERE status IN ('pending','queued')").fetchone()[0],0)


class SavedLegacyRequestTests(unittest.TestCase):
    legacy=MigrationTests.legacy
    def test_pre_migration_target_snapshot_can_execute_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.sqlite';old=self.legacy(path)
            old.db.execute("INSERT INTO reminders(id,content,due_at,created_at,status,source_message_id) VALUES(11,'Вода',?,1,'pending',1)",(NOW+300,))
            target=dict(old.db.execute('SELECT * FROM reminders WHERE id=11').fetchone())
            params={'id':11,'content':'Вода позже','due_at':'2026-10-20T09:00:00+03:00'}
            origin={'message_id':1,'text':'Перенеси','date':NOW}
            old.db.execute("INSERT INTO requests(id,session,action,parameters,origin,status,missing,target,created_at,updated_at) VALUES(3,0,'update_reminder',?,?,'ready','[]',?,1,1)",(dump(params),dump(origin),dump(target)))
            old.db.execute("UPDATE requests SET provided='[\"id\",\"content\",\"due_at\"]' WHERE id=3")
            old.db.commit();old.db.close()
            store=Store(path)
            runner=ToolRunner(store)
            job={'id':10,'payload':dump({'message':origin})}
            call={'id':'execute','function':{'name':'execute_request','arguments':dump({'request_id':3})}}
            result=runner.run(call,job,NOW)
            self.assertEqual(result['status'],'done',result)
            self.assertEqual(store.db.execute('SELECT id FROM reminders').fetchone()[0],11)
            self.assertEqual(runner.run(call,job,NOW),result)
            store.db.close()

class MigrationCommandTests(unittest.TestCase):
    def test_backup_restorable_and_running_bot_lock_prevents_migration(self):
        import fcntl
        from personal_assistent.migrate import migrate_database
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.sqlite';backup=Path(directory)/'snapshot.sqlite'
            store=MigrationTests().legacy(path)
            store.db.execute("INSERT INTO reminders(id,content,due_at,created_at,status,source_message_id) VALUES(17,'Вода',100,1,'sent',1)")
            store.db.commit();store.db.close()
            with open(str(path)+'.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    migrate_database(path,backup)
                self.assertFalse(backup.exists())
            migrate_database(path,backup)
            self.assertEqual(backup.stat().st_mode & 0o777,0o600)
            old=sqlite3.connect(backup)
            self.assertEqual(old.execute('SELECT id FROM reminders').fetchone()[0],17)
            self.assertIsNone(old.execute("SELECT name FROM sqlite_master WHERE name='events'").fetchone())
            self.assertEqual(old.execute('PRAGMA integrity_check').fetchone()[0],'ok')
            old.close()
            with self.assertRaises(FileExistsError):
                migrate_database(path,backup)
            current=Store(path);self.assertEqual(current.db.execute('SELECT id FROM reminders').fetchone()[0],17);current.db.close()

class ScenarioReviewTests(unittest.TestCase):
    setUp=EventTests.setUp
    create=EventTests.create
    occs=EventTests.occs
    rule=EventTests.rule
    change=EventTests.change
    tick=EventTests.tick

    def test_move_reactivates_previously_skipped_relative_deadline(self):
        event=self.create();self.rule(event,offset=5)
        self.assertEqual(self.db.execute('SELECT status FROM reminders').fetchone()[0],'cancelled')
        self.change(event,data={'content':'Встреча','start_date':'2026-10-30','start_time':'12:00'})
        self.assertEqual(tuple(self.db.execute('SELECT status,due_at FROM reminders').fetchone()),('pending',ts('2026-10-25T09:00')))

    def test_past_timed_event_does_not_recover_while_still_in_progress(self):
        event=self.create(end_date='2026-10-20',end_time='18:00');self.rule(event,offset=0,due_time='11:00')
        self.tick('2026-10-20T13:00')
        self.assertIsNone(self.store.next_outbox(ts('2026-10-20T13:00')))
        self.assertEqual(self.db.execute('SELECT reason FROM reminders').fetchone()[0],'event_passed')

    def test_moved_snooze_after_event_cancelled_with_explanation(self):
        event=self.create();self.rule(event)
        self.tick('2026-10-19T09:00');row=self.store.next_outbox(ts('2026-10-19T09:00'));self.store.sent(row['key'],1,NOW)
        self.store.snooze(row['reminder_id'],ts('2026-10-19T09:00'))
        result=self.change(event,data={'content':'Встреча','start_date':'2026-10-19','start_time':'09:05'})
        self.assertTrue(result['warnings'])
        self.assertEqual(self.db.execute('SELECT reason FROM reminders WHERE is_snooze=1').fetchone()[0],'after_event')

    def test_birthday_correction_changes_digest_date_but_not_year_metadata(self):
        event=self.create(kind='birthday',person='Оля',start_time=None,birth_year=1990)
        self.change(event,scope='future',data={'content':'День рождения Оли','kind':'birthday','person':'Оля','start_date':'2026-11-05','all_day':True})
        saved=self.e.event(event['id'])
        self.assertEqual((saved['birth_month'],saved['birth_day'],saved['birth_year']),(11,5,1990))
        self.assertEqual(self.occs(event)[0]['start_date'],'2026-11-05')

    def test_next_year_future_occurrence_does_not_justify_last_year_recovery(self):
        event=self.create(kind='birthday',person='Оля',start_time=None,start_date='2026-10-20')
        self.rule(event,offset=5)
        self.tick('2026-11-01T10:00')
        self.assertIsNone(self.store.next_outbox(ts('2026-11-01T10:00')))
        self.tick('2027-10-15T09:00')
        self.assertIsNotNone(self.store.next_outbox(ts('2027-10-15T09:00')))

    def test_restart_no_duplicate_event_notification_or_group(self):
        event=self.create();self.rule(event);self.tick('2026-10-19T09:00')
        self.tick('2026-10-19T09:01')
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM notification_groups').fetchone()[0],1)

    def test_profile_recompute_rolls_back_on_started_delivery(self):
        event=self.create();self.rule(event);self.tick('2026-10-19T09:00')
        row=self.store.next_outbox(ts('2026-10-19T09:00'));self.store.start_delivery(row['key'])
        before=self.db.execute('SELECT due_at FROM reminders').fetchone()[0]
        runner=ToolRunner(self.store)
        job={'id':1,'payload':dump({'message':{'text':'Установи время 11:00','message_id':1,'date':NOW}})}
        result=runner.run({'id':'p','function':{'name':'update_profile','arguments':dump({'source_text':'Установи время 11:00','notification_time':'11:00'})}},job,NOW)
        self.assertEqual(result['error']['code'],'delivery_in_progress')
        self.assertIsNone(self.store.get_meta('profile'))
        self.assertEqual(self.db.execute('SELECT due_at FROM reminders').fetchone()[0],before)

    def test_history_keeps_old_description_after_future_change(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':2})
        self.change(event,scope='future',from_position=1,data={'content':'Новое описание','start_date':'2026-10-21','start_time':'15:00'})
        self.assertEqual([r['content'] for r in self.occs(event)],['Встреча','Новое описание'])


class ToolScenarioReviewTests(unittest.TestCase):
    setUp=EventToolTests.setUp
    tool=EventToolTests.tool
    execute=EventToolTests.execute

    def test_default_request_uses_current_time_at_execution(self):
        ready=self.tool('prepare_request',action='create_reminder',parameters={'content':'Вода','due_date':'2026-10-20'})
        self.store.set_meta('profile',{'notification_time':'11:00'})
        self.tool('execute_request',request_id=ready['request_id'])
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0],ts('2026-10-20T11:00'))

    def test_oneoff_parents_not_listed_as_recurring_and_default_zone_rendered(self):
        self.execute('create_reminder',content='Вода',due_date='2026-10-20')
        self.assertEqual(self.tool('list_schedules')['total'],0)
        self.store.set_meta('profile',{'timezone':'Asia/Tokyo'})
        self.assertEqual(self.tool('get_reminder',id=1)['due_at'],'2026-10-20T09:00:00+03:00')

    def test_event_choices_survive_number_selection_and_mismatch_is_refused(self):
        first=self.execute('create_event',data={'content':'Встреча','start_date':'2026-10-20','start_time':'12:00'})['result']
        second=self.execute('create_event',data={'content':'Встреча','start_date':'2026-10-21','start_time':'12:00'})['result']
        choices=[{'record_type':'event','id':r['id']} for r in (first,second)]
        question=self.tool('ask_clarification',kind='other',question='Какая встреча?',choices=choices)
        self.assertIn('1. «Встреча»',question['question'])
        self.job['pending']={'selection':choices[0]}
        rejected=self.tool('get_event',id=second['id'])
        self.assertEqual(rejected['error']['code'],'selection_mismatch')

    def test_profile_moved_unmaterialized_default_slot_warns_and_skips(self):
        self.execute('create_schedule',content='Вода',recurrence={'frequency':'daily'})
        # At creation 09:00 has passed; next slot is tomorrow 09:00.
        self.store.set_meta('profile',{'notification_time':'07:00'})
        warnings=Events(self.store).profile_changed(ts('2026-10-18T08:00'))
        self.assertTrue(warnings)
        self.assertEqual(self.store.db.execute('SELECT next_due_at FROM schedules').fetchone()[0],ts('2026-10-19T07:00'))

class DigestSwitchTests(unittest.TestCase):
    setUp=EventTests.setUp
    create=EventTests.create

    def test_unstarted_digest_can_be_disabled_then_reenabled_once(self):
        self.create(kind='birthday',person='Оля',start_time=None)
        self.store.set_meta('profile',{'birthday_digest':True})
        self.e.digest(NOW)
        self.e.disable_digest()
        self.assertIsNone(self.store.next_outbox(NOW))
        self.assertEqual(self.db.execute('SELECT state FROM digest_runs').fetchone()[0],'cancelled')
        self.e.digest(NOW,immediate=True)
        row=self.store.next_outbox(NOW);self.store.sent(row['key'],1,NOW)
        self.e.disable_digest();self.e.digest(NOW,immediate=True)
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)

    def test_started_digest_is_protected(self):
        self.create(kind='birthday',person='Оля',start_time=None)
        self.store.set_meta('profile',{'birthday_digest':True});self.e.digest(NOW)
        row=self.store.next_outbox(NOW);self.store.start_delivery(row['key'])
        with self.assertRaises(EventError):
            self.e.disable_digest()
        self.assertIsNotNone(self.store.next_outbox(NOW))

class StartupSafetyTests(unittest.TestCase):
    def test_existing_database_requires_explicit_backup_migration_and_future_version_refused(self):
        from personal_assistent.migrate import requires_migration, migrate_database
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.sqlite'
            self.assertFalse(requires_migration(path))
            old=MigrationTests().legacy(path);old.db.close()
            self.assertTrue(requires_migration(path))
            migrate_database(path,Path(directory)/'backup.sqlite')
            self.assertFalse(requires_migration(path))
            new=Store(path)
            with new.db:
                new.set_meta('events_schema_version',999)
            new.db.close()
            with self.assertRaisesRegex(ValueError,'новее'):
                Store(path)

    def test_mixed_event_and_legacy_data_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mixed.sqlite';store=Store(path)
            with store.atomic():
                event=Events(store).create(EventData(content='Встреча',start_date='2026-10-20',start_time='12:00'),NOW,1)
                Events(store).add_rule(event['id'],ReminderRuleData(kind='calendar',offset=1),NOW,scope='future',from_position=0)
                store.db.execute("INSERT INTO reminders(content,due_at,created_at,source_message_id) VALUES('Отдельное',?,1,2)",(NOW+300,))
                store.set_meta('events_schema_version',None)
            before=[dict(r) for r in store.db.execute('SELECT * FROM reminders')]
            store.db.close()
            reopened=Store(path)
            self.assertEqual(before,[dict(r) for r in reopened.db.execute('SELECT * FROM reminders')])
            self.assertEqual(reopened.db.execute('SELECT count(*) FROM events').fetchone()[0],1)
            self.assertEqual(reopened.db.execute('PRAGMA foreign_key_check').fetchall(),[])
            reopened.db.close()

class FullMigrationRollbackTests(unittest.TestCase):
    def test_failure_in_store_initialization_does_not_leave_prefix_schema_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.sqlite';old=MigrationTests().legacy(path)
            # This predecessor had no calendar table or reminder source columns.
            old.db.execute('ALTER TABLE reminders DROP COLUMN schedule_id')
            old.db.execute('ALTER TABLE reminders DROP COLUMN occurrence_at')
            old.db.execute('ALTER TABLE reminders DROP COLUMN is_snooze')
            old.db.execute('DROP TABLE schedules')
            old.db.commit()
            before=[tuple(r) for r in old.db.execute("SELECT name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name")]
            old.db.close()
            original=Store.set_meta
            def fail(store,key,value):
                if key=='events_schema_version':
                    raise RuntimeError('simulated disk failure')
                original(store,key,value)
            with patch.object(Store,'set_meta',fail):
                with self.assertRaises(RuntimeError):
                    Store(path)
            check=sqlite3.connect(path)
            self.assertEqual(check.execute("SELECT name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name").fetchall(),before)
            check.close()

class GroupConstraintTests(unittest.TestCase):
    setUp=EventTests.setUp
    create=EventTests.create
    occs=EventTests.occs
    rule=EventTests.rule
    tick=EventTests.tick

    def test_cross_event_member_and_invalid_state_are_rejected(self):
        first=self.create();second=self.create();self.rule(first);self.rule(second)
        self.tick('2026-10-19T09:00')
        group=self.db.execute('SELECT * FROM notification_groups ORDER BY id LIMIT 1').fetchone()
        other=self.db.execute('SELECT id FROM reminders WHERE event_occurrence_id!=?',(group['occurrence_id'],)).fetchone()[0]
        # Remove its existing membership so the intended consistency trigger is exercised.
        self.db.execute('DELETE FROM notification_members WHERE reminder_id=?',(other,))
        with self.assertRaisesRegex(sqlite3.IntegrityError,'group member'):
            self.db.execute('INSERT INTO notification_members VALUES(?,?)',(group['id'],other))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE reminders SET status='done'")
