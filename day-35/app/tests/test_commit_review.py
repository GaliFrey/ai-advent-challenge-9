"""Regression scenarios found reviewing commit a694009 against product requirements."""
import json
import unittest
from datetime import date
from unittest.mock import patch

from personal_assistent.events import EventData, EventError, Events
from personal_assistent.profile import load_profile
from personal_assistent.recurrence import Rule
from personal_assistent.storage import dump
from personal_assistent.tools import ToolRunner
from personal_assistent.receipts import confirmed_text
import test_events as fixtures
from test_events import NOW, ts


class ReviewDomainTests(unittest.TestCase):
    setUp=fixtures.EventTests.setUp
    create=fixtures.EventTests.create
    rule=fixtures.EventTests.rule
    change=fixtures.EventTests.change
    occs=fixtures.EventTests.occs
    tick=fixtures.EventTests.tick

    def test_future_rule_change_preserves_implicit_until(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','until':'2026-10-23'})
        self.change(event,scope='future',from_position=1,data={'content':'Встреча','start_date':'2026-10-21','start_time':'15:00','recurrence':{'frequency':'daily','start_date':'2026-10-21'}})
        self.assertEqual(len(self.occs(event)),4)
        self.assertEqual(json.loads(self.e.view(event['id'])['segments'][-1]['rule'])['until'],'2026-10-23')

    def test_future_move_does_not_rewrite_past_history(self):
        event=self.create(start_date='2026-10-15',recurrence={'frequency':'daily','start_date':'2026-10-15','count':6})
        old=self.occs(event)
        self.change(event,scope='future',from_position=0,data={'content':'Новое','start_date':'2026-10-19','start_time':'15:00'})
        actual=self.occs(event)
        self.assertEqual(actual[:2],old[:2])
        self.assertEqual(len(actual),6)

    def test_birthday_duplicate_name_case_and_spaces_are_normalized(self):
        event=self.create(kind='birthday',person='Оля',start_time=None)
        duplicate=self.create(kind='birthday',person=' оЛя ',start_time=None)
        self.assertEqual(event['id'],duplicate['id'])

    def test_exact_interval_validated_for_actual_instance_not_old_segment(self):
        event=self.create(start_time=None,all_day=True)
        occ=self.occs(event)[0]
        self.change(event,data={'content':'Встреча','start_date':'2026-10-20','start_time':'12:00'})
        result=self.rule(event,scope='one',occurrence_id=occ['id'],kind='interval',offset=3600)
        self.assertIn('rule_id',result)

    def test_all_day_conversion_does_not_silently_rebase_interval_to_midnight(self):
        event=self.create();self.rule(event,kind='interval',offset=3600)
        with self.assertRaises(EventError) as error:
            self.change(event,data={'content':'Встреча','start_date':'2026-10-20','all_day':True})
        self.assertEqual(error.exception.code,'all_day_reference_required')
        self.assertEqual(self.occs(event)[0]['start_time'],'12:00:00')

    def test_invalid_fixed_rule_is_cancelled_with_explanation(self):
        event=self.create();occ=self.occs(event)[0]
        result=self.rule(event,scope='one',occurrence_id=occ['id'],kind='fixed',fixed_due_at='2026-10-21T09:00+03:00')
        self.assertTrue(result['warnings'])
        self.assertEqual(self.db.execute('SELECT reason FROM reminders').fetchone()[0],'after_event')

    def test_default_rule_can_become_valid_after_second_profile_change(self):
        event=self.create(start_time='08:00');self.rule(event,offset=0)
        self.assertEqual(self.db.execute('SELECT status FROM reminders').fetchone()[0],'cancelled')
        self.store.set_meta('profile',{'notification_time':'07:00'})
        with self.store.atomic():
            self.e.profile_changed(NOW)
        self.assertEqual(tuple(self.db.execute('SELECT status,due_at FROM reminders').fetchone()),('pending',ts('2026-10-20T07:00')))

    def test_queued_old_month_digest_is_not_sent_after_restart(self):
        self.create(kind='birthday',person='Оля',start_time=None)
        self.store.set_meta('profile',{'birthday_digest':True})
        self.e.digest(NOW)
        self.tick('2026-11-02T10:00')
        self.assertIsNone(self.store.next_outbox(ts('2026-11-02T10:00')))

    def test_deleted_birthday_is_removed_from_unstarted_digest(self):
        event=self.create(kind='birthday',person='Оля',start_time=None)
        self.store.set_meta('profile',{'birthday_digest':True});self.e.digest(NOW)
        self.change(event,operation='delete',scope='future')
        self.assertIsNone(self.store.next_outbox(NOW))

    def test_group_button_snoozes_remaining_rule_after_leader_rule_disabled(self):
        event=self.create();first=self.rule(event);self.rule(event)
        self.tick('2026-10-19T09:00')
        queued=self.store.next_outbox(ts('2026-10-19T09:00'));self.store.sent(queued['key'],1,ts('2026-10-19T09:00'))
        self.rule(event,scope='one',occurrence_id=self.occs(event)[0]['id'],kind='interval',offset=3600,replace_rule_id=first['rule_id'])
        self.assertEqual(self.store.snooze(queued['reminder_id'],ts('2026-10-19T09:01')),'Отложено на 10 минут.')
        self.assertEqual(self.db.execute('SELECT count(*) FROM reminders WHERE is_snooze=1').fetchone()[0],1)

    def test_calendar_projects_notification_rules_for_distant_occurrence(self):
        event=self.create(recurrence={'frequency':'yearly','start_date':'2026-10-20','dates':[{'month':10,'day':20}]})
        self.rule(event)
        result=self.e.calendar(date(2028,10,1),date(2028,10,31),NOW)
        self.assertEqual(len(result),1)
        self.assertEqual(len(result[0]['notifications']),1)
        self.assertEqual(result[0]['notifications'][0]['due_at'],ts('2028-10-19T09:00'))


class ReviewToolTests(unittest.TestCase):
    setUp=fixtures.EventToolTests.setUp
    tool=fixtures.EventToolTests.tool
    execute=fixtures.EventToolTests.execute

    def event(self):
        return self.execute('create_event',data={'content':'Встреча','start_date':'2026-10-20','start_time':'12:00'})['result']

    def test_cancel_and_disable_through_requests_write_receipts_and_replay(self):
        for operation in ('cancel','disable','delete'):
            with self.subTest(operation=operation):
                event=self.event()
                result=self.execute('change_event',id=event['id'],operation=operation,scope='future',from_position=0)
                self.assertEqual(result['status'],'done')
                self.assertIn({'cancel':'Событие отменено','disable':'Уведомления отключены','delete':'Событие удалено'}[operation],confirmed_text(self.store,1))
                replay=self.tool('execute_request',request_id=result['request_id'])
                self.assertEqual(replay,result)

    def test_receipt_cannot_confirm_faked_event_operation(self):
        event=self.event()
        ready=self.tool('prepare_request',action='change_event',parameters={'id':event['id'],'operation':'cancel','scope':'future','from_position':0})
        fake={'event':Events(self.store).view(event['id']),'operation':'cancel','warnings':[]}
        with patch.object(self.runner.requests,'apply',return_value=fake):
            with self.assertRaises(ValueError):
                self.tool('execute_request',request_id=ready['request_id'])
        self.assertEqual(self.store.db.execute('SELECT status FROM event_occurrences').fetchone()[0],'active')

    def test_standalone_calendar_preserves_pin_and_actual_move(self):
        reminder=self.execute('create_reminder',content='Вода',due_date='2026-10-20',due_time='09:00')['result']
        self.store.set_meta('profile',{'timezone':'Asia/Tokyo'})
        self.execute('update_reminder',id=reminder['id'],content='Вода позже',due_at='2026-10-21T10:00+03:00')
        item=self.tool('calendar',start='2026-10-20',end='2026-10-21')['items'][0]
        self.assertEqual(item['due_at'],ts('2026-10-21T10:00'))
        self.assertEqual(item['timezone'],'Europe/Kirov')

    def test_calendar_suppresses_original_slot_of_moved_or_skipped_standalone(self):
        schedule=self.execute('create_schedule',content='Вода',recurrence={'frequency':'daily','start_date':'2026-10-20','count':3},due_time='09:00')['result']
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        reminder=self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.execute('update_reminder',id=reminder,content='Вода позже',due_at='2026-10-22T10:00+03:00')
        page=self.tool('calendar',start='2026-10-20',end='2026-10-22')
        self.assertNotIn(ts('2026-10-20T09:00'),[r['due_at'] for r in page['items']])
        self.assertIn(ts('2026-10-22T10:00'),[r['due_at'] for r in page['items']])
        self.assertEqual(schedule['id'],page['items'][0]['schedule_id'])

    def test_skip_final_standalone_position_is_successful(self):
        schedule=self.execute('create_schedule',content='Вода',recurrence={'frequency':'daily','start_date':'2026-10-20','count':1},due_time='09:00')['result']
        skipped=self.execute('skip_schedule',id=schedule['id'])
        self.assertEqual(skipped['status'],'done')
        self.store.prepare_reminders(ts('2026-10-20T10:00'),lambda r:r['content'])
        self.assertIsNone(self.store.next_outbox(ts('2026-10-20T10:00')))

    def test_event_page_always_makes_progress_with_many_notifications(self):
        event=self.event()
        for _ in range(75):
            self.execute('set_event_rule',id=event['id'],scope='future',from_position=0,rule={'kind':'calendar','offset':1})
        page=self.tool('calendar',start='2026-10-20',end='2026-10-20')
        self.assertTrue(page['items'])
        self.assertTrue(page['items'][0].get('notifications_incomplete'))

    def test_default_time_create_can_use_dst_gap_resolution(self):
        self.store.set_meta('profile',{'timezone':'Europe/Berlin','notification_time':'02:30'})
        ready=self.tool('prepare_request',action='create_reminder',parameters={'content':'Вода','due_date':'2027-03-28'})
        self.assertEqual(ready['status'],'ready',ready)
        result=self.tool('execute_request',request_id=ready['request_id'])
        self.assertEqual(result['status'],'done')
        self.assertEqual(self.store.db.execute('SELECT due_at FROM reminders').fetchone()[0],ts('2027-03-28T03:30+02:00'))

class ReviewRecoveryTests(unittest.TestCase):
    setUp=fixtures.EventTests.setUp
    create=fixtures.EventTests.create
    rule=fixtures.EventTests.rule
    change=fixtures.EventTests.change
    occs=fixtures.EventTests.occs
    tick=fixtures.EventTests.tick

    def test_existing_id_beyond_new_lookahead_is_not_cancelled(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':3})
        before=self.occs(event)
        self.change(event,scope='future',from_position=1,data={'content':'Встреча','start_date':'2026-10-21','start_time':'15:00','recurrence':{'frequency':'yearly','start_date':'2026-10-21','interval':10,'dates':[{'month':10,'day':21}]}})
        after=self.occs(event)
        self.assertEqual(after[2]['id'],before[2]['id'])
        self.assertEqual(after[2]['start_date'],'2036-10-21')
        self.assertEqual(after[2]['status'],'active')

    def test_profile_rescheduled_queued_slot_is_superseded_after_downtime(self):
        runner=ToolRunner(self.store)
        job={'id':1,'payload':dump({'message':{'text':'Вода','message_id':1,'date':NOW}})}
        def tool(name,args,identity):
            return runner.run({'id':identity,'function':{'name':name,'arguments':dump(args)}},job,NOW)
        ready=tool('prepare_request',{'action':'create_schedule','parameters':{'content':'Вода','recurrence':{'frequency':'daily'}}},'p')
        tool('execute_request',{'request_id':ready['request_id']},'x')
        self.tick('2026-10-18T09:00')
        with self.store.atomic():
            self.store.set_meta('profile',{'notification_time':'11:00'})
            self.e.profile_changed(ts('2026-10-18T09:01'))
        self.tick('2026-10-20T12:00')
        self.assertEqual(self.db.execute("SELECT count(*) FROM reminders WHERE status='queued'").fetchone()[0],1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox WHERE sent_at IS NULL').fetchone()[0],1)


class ReviewStandaloneLimitsTests(unittest.TestCase):
    setUp=fixtures.EventToolTests.setUp
    tool=fixtures.EventToolTests.tool
    execute=fixtures.EventToolTests.execute

    def test_frequency_change_does_not_reset_delivered_position_count(self):
        schedule=self.execute('create_schedule',content='Вода',due_time='09:00',recurrence={'frequency':'daily','start_date':'2026-10-20','count':3})['result']
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        queued=self.store.next_outbox(ts('2026-10-20T09:00'));self.store.sent(queued['key'],1,ts('2026-10-20T09:00'))
        ready=self.runner.requests.prepare('update_schedule',{'id':schedule['id'],'recurrence':{'frequency':'weekly','weekdays':[2]}},self.job,ts('2026-10-20T10:00'))
        changed=self.runner.requests.execute(ready['request_id'],{**self.job,'tool_call_id':'change'},ts('2026-10-20T10:00'))
        rule=Rule.model_validate(changed['result']['recurrence'])
        self.assertEqual(rule.count,2)
        self.assertEqual(rule.start_date,date(2026,10,21))

    def test_update_series_preserves_manual_deadline_and_snooze(self):
        schedule=self.execute('create_schedule',content='Вода',due_time='09:00',recurrence={'frequency':'daily','start_date':'2026-10-20'})['result']
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        reminder=self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.execute('update_reminder',id=reminder,content='Позже',due_at='2026-10-25T10:00+03:00')
        self.execute('update_schedule',id=schedule['id'],due_time='10:00')
        row=self.store.db.execute('SELECT * FROM reminders WHERE id=?',(reminder,)).fetchone()
        self.assertEqual((row['status'],row['due_at']),('pending',ts('2026-10-25T10:00')))

    def test_final_queued_standalone_can_be_skipped_after_series_completed(self):
        schedule=self.execute('create_schedule',content='Вода',due_time='09:00',recurrence={'frequency':'daily','start_date':'2026-10-20','count':1})['result']
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        self.assertEqual(self.store.db.execute('SELECT status FROM schedules').fetchone()[0],'completed')
        skipped=self.execute('skip_schedule',id=schedule['id'])
        self.assertEqual(skipped['status'],'done')
        self.assertIsNone(self.store.next_outbox(ts('2026-10-20T09:00')))

class ReviewReadAndMigrationTests(unittest.TestCase):
    setUp=fixtures.EventToolTests.setUp
    tool=fixtures.EventToolTests.tool
    execute=fixtures.EventToolTests.execute

    def test_all_notification_pages_are_readable_without_zero_progress(self):
        event=self.execute('create_event',data={'content':'Встреча','start_date':'2026-10-20','start_time':'12:00'})['result']
        for _ in range(75):
            self.execute('set_event_rule',id=event['id'],scope='future',from_position=0,rule={'kind':'calendar','offset':1})
        page=self.tool('calendar',start='2026-10-20',end='2026-10-20')
        self.assertTrue(page['incomplete'])
        occ=page['items'][0]
        offset=0;identities=[]
        while offset is not None:
            read=self.tool('get_event_occurrence',id=occ['id'],offset=offset,limit=50)
            self.assertNotIn('error',read,read)
            self.assertTrue(read['notifications'])
            identities.extend(r['id'] for r in read['notifications'])
            offset=read['next_notification_offset']
        self.assertEqual(len(set(identities)),75)
        first=self.tool('get_event',id=event['id'],limit=50)
        self.assertNotIn('error',first,first)
        self.assertTrue(first['incomplete'])
        self.assertIsNotNone(first['next_rules_offset'])

    def test_long_historical_segment_text_remains_readable(self):
        content='Исходный текст '*150
        event=self.execute('create_event',data={'content':content,'start_date':'2026-10-20','start_time':'12:00'})['result']
        self.execute('change_event',id=event['id'],operation='move',scope='future',from_position=0,data={'content':'Новое','start_date':'2026-10-21','start_time':'12:00'})
        read=self.tool('get_event',id=event['id'],content_offset=300,content_limit=300)
        historical=[r for r in read['segments'] if r['content'].startswith('ый текст') or r['text_lengths']['content']>300]
        self.assertEqual(len(historical),1)
        self.assertEqual(historical[0]['content'],content[300:600])

    def test_completed_source_default_pending_notification_recomputes(self):
        self.execute('create_schedule',content='Вода',recurrence={'frequency':'daily','start_date':'2026-10-20','count':1})
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        with self.store.atomic():
            self.store.set_meta('profile',{'notification_time':'11:00'})
            Events(self.store).profile_changed(ts('2026-10-20T09:01'))
        row=self.store.db.execute('SELECT * FROM reminders').fetchone()
        self.assertEqual((row['status'],row['due_at']),('pending',ts('2026-10-20T11:00')))

    def test_v1_upgrade_preserves_manual_deadline_and_saved_target(self):
        import tempfile
        from pathlib import Path
        from personal_assistent.storage import Store
        from personal_assistent.migrate import requires_migration,migrate_database
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v1.sqlite';backup=Path(directory)/'backup.sqlite'
            old=Store(path)
            old.db.execute('ALTER TABLE reminders DROP COLUMN is_manual')
            rule=Rule(frequency='daily',start_date='2026-10-17')
            old.db.execute("INSERT INTO schedules(id,content,rule,due_time,timezone,next_due_at,source_message_id,created_at,updated_at) VALUES(7,'Вода',?,'09:00','Europe/Kirov',?,1,1,1)",(rule.model_dump_json(),ts('2026-10-21T09:00')))
            old.db.execute("INSERT INTO reminders(id,content,due_at,planned_due_at,created_at,source_message_id,schedule_id,occurrence_at) VALUES(11,'Позже',?,?,1,1,7,?)",(ts('2026-10-25T10:00'),ts('2026-10-25T10:00'),ts('2026-10-20T09:00')))
            old.db.execute("INSERT INTO schedules(id,content,rule,due_time,timezone,next_due_at,source_message_id,created_at,updated_at,time_mode) VALUES(8,'Вода',?,'09:00','Europe/Kirov',?,1,1,1,'default')",(rule.model_dump_json(),ts('2026-10-21T09:00')))
            old.db.execute("INSERT INTO reminders(id,content,due_at,planned_due_at,created_at,source_message_id,schedule_id,occurrence_at) VALUES(12,'Позже',?,?,1,1,8,?)",(ts('2026-10-26T10:00'),ts('2026-10-20T11:00'),ts('2026-10-20T09:00')))
            target=dict(old.db.execute('SELECT * FROM reminders WHERE id=11').fetchone())
            old.set_meta('events_schema_version',1);old.db.commit();old.db.close()
            self.assertTrue(requires_migration(path))
            migrate_database(path,backup)
            upgraded=Store(path)
            actual=dict(upgraded.db.execute('SELECT * FROM reminders WHERE id=11').fetchone())
            self.assertEqual({key:actual[key] for key in target},target)
            self.assertEqual(actual['is_manual'],1)
            self.assertEqual(upgraded.db.execute('SELECT is_manual FROM reminders WHERE id=12').fetchone()[0],1)
            upgraded.prepare_reminders(ts('2026-10-22T09:00'),lambda r:r['content'])
            self.assertEqual(upgraded.db.execute('SELECT status FROM reminders WHERE id=11').fetchone()[0],'pending')
            self.assertEqual(upgraded.db.execute('SELECT status FROM reminders WHERE id=12').fetchone()[0],'pending')
            self.assertFalse(requires_migration(path))
            upgraded.db.close()

class ReviewGroupEdgeTests(unittest.TestCase):
    setUp=fixtures.EventTests.setUp
    create=fixtures.EventTests.create
    rule=fixtures.EventTests.rule
    change=fixtures.EventTests.change
    occs=fixtures.EventTests.occs
    tick=fixtures.EventTests.tick

    def test_equal_deadline_added_after_queue_reuses_existing_group(self):
        event=self.create();self.rule(event,offset=3,due_time='10:00')
        self.store.prepare_reminders(NOW,lambda r:r['content'])
        self.rule(event,offset=3,due_time='10:00')
        self.store.prepare_reminders(NOW,lambda r:r['content'])
        self.assertEqual(self.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM notification_members').fetchone()[0],2)

    def test_equal_deadline_added_after_delivery_does_not_send_twice(self):
        event=self.create();self.rule(event,offset=3,due_time='10:00')
        self.store.prepare_reminders(NOW,lambda r:r['content'])
        queued=self.store.next_outbox(NOW);self.store.sent(queued['key'],1,NOW)
        self.rule(event,offset=3,due_time='10:00')
        self.store.prepare_reminders(NOW,lambda r:r['content'])
        self.assertIsNone(self.store.next_outbox(NOW))
        self.assertEqual(self.db.execute('SELECT count(*) FROM notification_members').fetchone()[0],2)
        self.assertEqual(self.db.execute("SELECT count(*) FROM reminders WHERE status='sent'").fetchone()[0],2)

    def test_started_digest_retained_and_deletion_explains_uncertainty(self):
        event=self.create(kind='birthday',person='Оля',start_time=None)
        self.store.set_meta('profile',{'birthday_digest':True});self.e.digest(NOW)
        queued=self.store.next_outbox(NOW);self.store.start_delivery(queued['key'])
        result=self.change(event,operation='delete',scope='future')
        self.assertTrue(result['warnings'])
        self.assertIsNotNone(self.store.next_outbox(NOW))

class ReviewDigestClockTests(unittest.TestCase):
    setUp=fixtures.EventTests.setUp
    create=fixtures.EventTests.create

    def test_future_month_queue_from_previous_zone_does_not_send_early(self):
        self.create(kind='birthday',person='Оля',start_date='2026-11-20',start_time=None)
        now=ts('2026-11-01T00:30')
        self.store.set_meta('profile',{'birthday_digest':True,'notification_time':'00:00'})
        self.e.digest(now)
        self.assertIsNotNone(self.store.next_outbox(now))
        self.store.set_meta('profile',{'birthday_digest':True,'notification_time':'00:00','timezone':'America/Los_Angeles'})
        self.e.recover_digests(now)
        self.assertIsNone(self.store.next_outbox(now))
        self.e.digest(ts('2026-11-01T07:00Z'))
        self.assertIsNotNone(self.store.next_outbox(ts('2026-11-01T07:00Z')))

    def test_unstarted_current_digest_follows_new_default_clock(self):
        self.create(kind='birthday',person='Оля',start_time=None)
        now=ts('2026-10-01T09:00')
        self.store.set_meta('profile',{'birthday_digest':True});self.e.digest(now)
        self.store.set_meta('profile',{'birthday_digest':True,'notification_time':'11:00'})
        self.e.reschedule_digest(now)
        self.assertIsNone(self.store.next_outbox(now))
        self.assertIsNotNone(self.store.next_outbox(ts('2026-10-01T11:00')))

class ReviewCancellationTests(unittest.TestCase):
    setUp=fixtures.EventToolTests.setUp
    tool=fixtures.EventToolTests.tool
    execute=fixtures.EventToolTests.execute

    def test_user_cancelled_slot_does_not_reappear_after_clock_change(self):
        self.execute('create_schedule',content='Вода',recurrence={'frequency':'daily','start_date':'2026-10-20'})
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        identity=self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.execute('cancel_reminder',id=identity)
        self.store.set_meta('profile',{'notification_time':'11:00'})
        with self.store.atomic():
            Events(self.store).profile_changed(ts('2026-10-20T09:01'))
        self.store.prepare_reminders(ts('2026-10-20T11:00'),lambda r:r['content'])
        self.assertIsNone(self.store.next_outbox(ts('2026-10-20T11:00')))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0],1)

    def test_updated_calendar_clock_projects_new_slot_before_delivery(self):
        schedule=self.execute('create_schedule',content='Вода',due_time='09:00',recurrence={'frequency':'daily','start_date':'2026-10-20'})['result']
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        self.execute('update_schedule',id=schedule['id'],due_time='10:00')
        page=self.tool('calendar',start='2026-10-20',end='2026-10-20')
        self.assertIn(ts('2026-10-20T10:00'),[r['due_at'] for r in page['items'] if r['status']=='planned'])

    def test_skip_chooses_calendar_slot_before_later_manual_deadline(self):
        schedule=self.execute('create_schedule',content='Вода',due_time='09:00',recurrence={'frequency':'daily','start_date':'2026-10-20'})['result']
        self.store.prepare_reminders(ts('2026-10-20T09:00'),lambda r:r['content'])
        identity=self.store.db.execute('SELECT id FROM reminders').fetchone()[0]
        self.execute('update_reminder',id=identity,content='Позже',due_at='2026-10-25T10:00+03:00')
        skipped=self.execute('skip_schedule',id=schedule['id'])
        self.assertEqual(skipped['result']['skipped_due_at'],'2026-10-21T09:00:00+03:00')
        self.assertEqual(self.store.db.execute('SELECT status FROM reminders WHERE id=?',(identity,)).fetchone()[0],'pending')

class ReviewActualOccurrenceTests(unittest.TestCase):
    setUp=fixtures.EventToolTests.setUp
    tool=fixtures.EventToolTests.tool
    execute=fixtures.EventToolTests.execute

    def test_read_event_exposes_manual_move_beyond_one_year_calendar_window(self):
        event=self.execute('create_event',data={'content':'Встреча','start_date':'2026-10-20','start_time':'12:00'})['result']
        occ=self.tool('calendar',start='2026-10-20',end='2026-10-20')['items'][0]
        self.execute('change_event',id=event['id'],operation='move',scope='one',occurrence_id=occ['id'],data={'content':'Встреча','start_date':'2030-01-15','start_time':'14:00'})
        actual=self.tool('get_event',id=event['id'])['next_occurrence']
        self.assertEqual(actual['id'],occ['id'])
        self.assertEqual(actual['start_date'],'2030-01-15')
        self.assertEqual(actual['start_time'],'14:00:00')


class ReviewCountEchoTests(unittest.TestCase):
    setUp=fixtures.EventTests.setUp
    create=fixtures.EventTests.create
    change=fixtures.EventTests.change
    occs=fixtures.EventTests.occs

    def test_echoing_full_rule_count_does_not_add_an_extra_position(self):
        event=self.create(recurrence={'frequency':'daily','start_date':'2026-10-20','count':4})
        self.change(event,scope='future',from_position=1,data={'content':'Встреча','start_date':'2026-10-21','start_time':'15:00','recurrence':{'frequency':'daily','start_date':'2026-10-21','count':4}})
        self.assertEqual(len(self.occs(event)),4)
        self.assertEqual(json.loads(self.e.view(event['id'])['segments'][-1]['rule'])['count'],3)


class ReviewBirthdayLimitTests(unittest.TestCase):
    setUp=fixtures.EventTests.setUp
    create=fixtures.EventTests.create
    occs=fixtures.EventTests.occs

    def test_birthday_limit_is_not_ignored_and_birth_year_is_only_metadata(self):
        event=self.create(kind='birthday',person='Оля',start_time=None,birth_year=1990,recurrence={'frequency':'yearly','start_date':'2026-10-20','dates':[{'month':10,'day':20}],'count':3})
        self.e.materialize(event['id'],NOW,date(2035,1,1))
        self.assertEqual([r['start_date'] for r in self.occs(event)],['2026-10-20','2027-10-20','2028-10-20'])
