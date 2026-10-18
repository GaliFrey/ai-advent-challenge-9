import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import httpx
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner, local_time
from personal_assistent.agent import Agent, DeepSeek
from personal_assistent.runtime import Runtime
from personal_assistent.telegram import Telegram, TelegramError

NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc).timestamp()
OWNER = 12345


def update(text='запомни: Оля обещала не пить газировку', number=1):
    return {'update_id': number, 'message': {'message_id': number, 'date': int(NOW),
            'from': {'id': OWNER}, 'chat': {'id': OWNER, 'type': 'private'}, 'text': text}}


def call(name, args, identity='call-1'):
    return {'id': identity, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}


def answer(*calls, text=None):
    result = {'role': 'assistant', 'content': text}
    if calls:
        result['tool_calls'] = list(calls)
    return result


class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.retrieval_fixtures = list(responses)
        self.inputs = []

    async def retrieve(self, messages):
        from retrieval_stub import retrieval_reply
        if not hasattr(self, 'retrieval_inputs'):
            self.retrieval_inputs = []
        self.retrieval_inputs.append(json.loads(json.dumps(messages)))
        return retrieval_reply(messages, self)

    async def complete(self, messages):
        self.inputs.append(json.loads(json.dumps(messages)))
        if not self.responses:
            raise AssertionError('Unexpected model call')
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        if not result.get('tool_calls') and result.get('content'):
            result = {**result, 'content': json.dumps({'kind': 'answer', 'text': result['content']}, ensure_ascii=False)}
        return result


class FakeTelegram:
    def __init__(self):
        self.sent = []
        self.incoming = []
        self.offsets = []
        self.error = None

    async def updates(self, offset):
        self.offsets.append(offset)
        result, self.incoming = self.incoming, []
        return result

    async def send(self, owner, text, reminder_id=None):
        if self.error:
            raise self.error
        self.sent.append((owner, text, reminder_id))
        return {'message_id': len(self.sent)}

    async def acknowledge(self, identity):
        pass


class Case(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.runner = ToolRunner(self.store)
        self.store.enqueue(update(), NOW)
        self.job = self.store.next_job()
        self.count = 0

    def tearDown(self):
        self.store.db.close()

    def tool(self, name, **args):
        self.count += 1
        return self.runner.run(call(name, args, str(self.count)), self.job, NOW)

    def reminder(self, seconds=300, content='Проверить духовку'):
        return self.tool('create_reminder', content=content, due_at=local_time(NOW + seconds))

    def test_note_distinguishes_event_and_storage_date(self):
        note = self.tool('create_note', content='Оля обещала не пить газировку', event_date='2026-10-09')
        self.assertEqual(note['event_date'], '2026-10-09')
        self.assertTrue(note['created_at'].startswith('2026-10-16T15:00'))
        self.assertIn('Оля', note['original'])

    def test_unknown_event_date_is_not_invented(self):
        note = self.tool('create_note', content='Факт без даты')
        self.assertIsNone(note['event_date'])

    def test_conflicting_notes_remain_separate(self):
        self.tool('create_note', content='Оля обещала не пить газировку')
        self.tool('create_note', content='Оля снова пьёт газировку')
        self.assertEqual(self.tool('list_notes')['total'], 2)

    def test_search_then_edit_then_delete(self):
        note = self.tool('create_note', content='Оля обещала не пить газировку')
        found = self.tool('search_notes', query='Оля газировка')
        self.assertEqual(found['items'][0]['id'], note['id'])
        self.tool('update_note', id=note['id'], content='Маша обещала не пить кофе', event_date=None)
        self.assertEqual(self.tool('search_notes', query='Оля')['items'], [])
        self.assertEqual(len(self.tool('search_notes', query='Маша')['items']), 1)
        self.tool('delete_note', id=note['id'])
        self.assertEqual(self.tool('list_notes')['total'], 0)
        self.assertEqual(self.tool('search_notes', query='Маша')['items'], [])

    def test_empty_search_is_not_a_hallucinated_fact(self):
        self.assertEqual(self.tool('search_notes', query='Оля')['items'], [])

    def test_natural_intent_is_not_rejected_by_keyword_gate(self):
        result = self.runner.run(call('create_note', {'content': 'Факт'}), self.job, NOW)
        self.assertIn('id', result)

    def test_tool_errors_identify_the_actual_reason(self):
        cases = [
            ('unknown', {}, 'unknown_tool'),
            ('create_note', {'content': 'Факт', 'unexpected': True}, 'invalid_arguments'),
            ('create_reminder', {'content': 'Чай', 'due_at': '2026-10-17T10:00:00'}, 'invalid_arguments'),
            ('update_reminder', {'id':1,'content': 'Чай', 'due_at': local_time(NOW - 60)}, 'past_due'),
            ('get_note', {'id': 100}, 'note_not_found'),
            ('cancel_reminder', {'id': 100}, 'reminder_not_active'),
            ('create_note', {'content': 'Факт', 'continue_request': True}, 'no_pending_request'),
        ]
        for name, args, code in cases:
            with self.subTest(name=name, code=code):
                result = self.tool(name, **args)
                self.assertEqual(result['error']['code'], code)
                self.assertTrue(result['error']['message'])
        invalid = self.tool('create_reminder', content='Чай', due_at='not-a-date')
        self.assertEqual(invalid['error']['details'][0]['field'], 'due_at')
        self.assertNotIn('now', invalid['error'].get('details', {}))
        past = self.tool('update_reminder', id=1, content='Чай', due_at=local_time(NOW - 1))
        self.assertEqual(past['error']['details']['now'], local_time(NOW))
        self.assertEqual(self.tool('list_notes')['total'], 0)
        self.assertEqual(self.tool('list_reminders')['total'], 0)

    def test_malformed_json_has_distinct_error_and_is_cached(self):
        command = call('create_note', {})
        command['function']['arguments'] = '{broken'
        first = self.runner.run(command, self.job, NOW)
        self.assertEqual(first['error']['code'], 'invalid_json')
        self.assertEqual(first, self.runner.run(command, self.job, NOW))

    def test_replayed_tool_call_does_not_duplicate(self):
        command = call('create_reminder', {'content': 'Чай', 'due_at': local_time(NOW + 300)})
        first = self.runner.run(command, self.job, NOW)
        second = self.runner.run(command, self.job, NOW)
        self.assertEqual(first, second)
        self.assertEqual(self.tool('list_reminders')['total'], 1)

    def test_past_creation_is_due_immediately_but_naive_time_is_rejected(self):
        self.assertIn('error', self.tool('create_reminder', content='Чай', due_at='2026-10-17T10:00:00'))
        result=self.tool('create_reminder',content='Чай',due_at=local_time(NOW-1))
        self.assertEqual(result['due_at'],local_time(NOW-1))
        self.store.prepare_reminders(NOW,lambda r:r['content'])
        self.assertIsNotNone(self.store.next_outbox(NOW))

    def test_schedule_is_not_early_and_does_not_duplicate(self):
        self.reminder()
        self.store.prepare_reminders(NOW + 299, lambda _: 'Духовка')
        self.assertIsNone(self.store.next_outbox(NOW + 299))
        self.store.prepare_reminders(NOW + 300, lambda _: 'Духовка')
        self.store.prepare_reminders(NOW + 301, lambda _: 'Духовка')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM outbox').fetchone()[0], 1)

    def test_cancel_removes_queued_notification(self):
        r = self.reminder()
        self.store.prepare_reminders(NOW + 300, lambda _: 'Духовка')
        self.tool('cancel_reminder', id=r['id'])
        self.assertIsNone(self.store.next_outbox(NOW + 300))

    def test_reschedule_removes_old_notification(self):
        r = self.reminder()
        self.store.prepare_reminders(NOW + 300, lambda _: 'Старый срок')
        self.tool('update_reminder', id=r['id'], content='Новый текст', due_at=local_time(NOW + 900))
        self.assertIsNone(self.store.next_outbox(NOW + 300))
        self.assertEqual(self.store.due_reminders(NOW + 900)[0]['content'], 'Новый текст')

    def test_snooze_is_once_and_from_click_time(self):
        r = self.reminder()
        self.store.prepare_reminders(NOW + 300, lambda _: 'Духовка')
        row = self.store.next_outbox(NOW + 300)
        self.store.sent(row['key'], 1, NOW + 300)
        self.store.snooze(r['id'], NOW + 400)
        self.store.snooze(r['id'], NOW + 500)
        rows = self.tool('list_reminders')['items']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['due_at'], local_time(NOW + 1000))

    def test_telegram_retry_after_is_respected(self):
        with self.store.db:
            self.store.queue_message('test', 'hello', now=NOW)
        row = self.store.next_outbox(NOW)
        self.store.retry(row['key'], NOW, 120)
        self.assertIsNone(self.store.next_outbox(NOW + 119))
        self.assertIsNotNone(self.store.next_outbox(NOW + 120))

    def test_unicode_chunks_fit_telegram_utf16_limit(self):
        text = '😀' * 5000
        with self.store.db:
            self.store.queue_message('long', text, now=NOW)
        rows = self.store.db.execute('SELECT text FROM outbox ORDER BY key').fetchall()
        self.assertEqual(''.join(r[0] for r in rows), text)
        self.assertTrue(all(len(r[0].encode('utf-16-le')) // 2 <= 4096 for r in rows))

    def test_failed_first_chunk_blocks_later_chunks_only_in_same_message(self):
        with self.store.db:
            self.store.queue_message('long', 'a' * 2000, now=NOW)
        first = self.store.next_outbox(NOW)
        self.store.retry(first['key'], NOW, 60)
        self.assertIsNone(self.store.next_outbox(NOW + 1))
        with self.store.db:
            self.store.queue_message('unrelated', 'other', now=NOW + 1)
        self.assertEqual(self.store.next_outbox(NOW + 1)['text'], 'other')

    def test_rescheduling_partial_delivery_is_rejected_and_original_message_completes(self):
        reminder = self.reminder(content='a' * 2000)
        self.store.prepare_reminders(NOW + 300, lambda _: 'a' * 2000)
        first = self.store.next_outbox(NOW + 300)
        self.store.sent(first['key'], 1, NOW + 300)
        result = self.tool('update_reminder', id=reminder['id'], content='b' * 2000,
                           due_at=local_time(NOW + 900))
        self.assertEqual(result['error']['code'], 'delivery_in_progress')
        parts = [first['text']]
        while row := self.store.next_outbox(NOW + 900):
            parts.append(row['text'])
            self.store.sent(row['key'], len(parts) + 1, NOW + 900)
        self.assertEqual(''.join(parts), 'a' * 2000)
        self.assertEqual(self.store.db.execute('SELECT status FROM reminders WHERE id=?',
                         (reminder['id'],)).fetchone()[0], 'sent')

    def test_snooze_rolls_back_with_callback_transaction(self):
        reminder = self.reminder()
        self.store.prepare_reminders(NOW + 300, lambda _: 'Чай')
        row = self.store.next_outbox(NOW + 300)
        self.store.sent(row['key'], 1, NOW + 300)
        with self.assertRaises(RuntimeError):
            with self.store.db:
                # Start the callback transaction before calling the nested operation.
                self.store.set_meta('callback-test', True)
                self.store.snooze(reminder['id'], NOW + 400)
                raise RuntimeError('simulate receipt failure')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 1)
        self.assertIsNone(self.store.get_meta('callback-test'))

    def test_repeated_update_has_one_job(self):
        self.store.enqueue(update(), NOW)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM jobs').fetchone()[0], 1)
        self.assertEqual(self.store.get_meta('offset'), 2)

    def test_invalid_arguments_do_not_mutate(self):
        self.assertIn('error', self.tool('create_note', content='Факт', unexpected=True))
        self.assertEqual(self.tool('list_notes')['total'], 0)


class AsyncCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(':memory:')
        self.telegram = FakeTelegram()

    async def asyncTearDown(self):
        self.store.db.close()

    def runtime(self, model=None, now=NOW):
        return Runtime(self.store, self.telegram, Agent(self.store, model or Model(), lambda: now), OWNER, lambda: now)

    async def test_agent_persists_note_and_reply(self):
        self.store.enqueue(update(), NOW)
        model = Model(answer(call('create_note', {'content': 'Оля обещала не пить газировку', 'event_date': '2026-10-16'})), answer(text='Сохранена памятка №1.'))
        await self.runtime(model).process_once()
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
        self.assertEqual(self.store.history()[-1]['content'], 'Памятка сохранена: "Оля обещала не пить газировку". Дата события: 16.10.2026.')
        await self.runtime().deliver_once()
        self.assertEqual(self.telegram.sent[0][1], 'Памятка сохранена: "Оля обещала не пить газировку". Дата события: 16.10.2026.')

    async def test_natural_reminder_formulations_reach_creation_tool(self):
        phrases = [
            'мне нужно напомнить через 3 минуты почистить зубы',
            'поставь напоминание через 3 минуты почистить зубы',
            'через 3 минуты скажи, что пора почистить зубы',
        ]
        for number, phrase in enumerate(phrases, 1):
            with self.subTest(phrase=phrase):
                self.store.enqueue(update(phrase, number), NOW)
                model = Model(answer(call('create_reminder', {'content': 'Почистить зубы',
                              'due_at': local_time(NOW + 180)})), answer(text='Напоминание создано.'))
                await self.runtime(model).process_once()
                row = self.store.db.execute('SELECT * FROM reminders ORDER BY id DESC LIMIT 1').fetchone()
                self.assertEqual(row['due_at'], NOW + 180)
                self.assertEqual(row['source_message_id'], number)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 3)

    async def test_natural_note_formulations_reach_creation_tool(self):
        for number, phrase in enumerate(('Сохрани факт: Оля обещала не пить газировку',
                                        'Хочу запомнить: Оля обещала не пить газировку'), 1):
            self.store.enqueue(update(phrase, number), NOW)
            model = Model(answer(call('create_note', {'content': 'Оля обещала не пить газировку'})),
                          answer(text='Сохранено.'))
            await self.runtime(model).process_once()
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 2)

    async def test_new_request_after_clarification_uses_current_message(self):
        self.store.enqueue(update('Напомни завтра про молоко'), NOW)
        model = Model(answer(call('ask_clarification', {'question': 'Во сколько?', 'kind': 'reminder'})))
        await self.runtime(model).process_once()
        second = update('Лучше поставь напоминание через 3 минуты про зубы', 2)
        second['message']['date'] = int(NOW + 3600)
        self.store.enqueue(second, NOW + 3600)
        model = Model(answer(call('create_reminder', {'content': 'Почистить зубы',
                      'due_at': local_time(NOW + 3780), 'continue_request': False})),
                      answer(text='Создано.'))
        await self.runtime(model, NOW + 3600).process_once()
        row = self.store.db.execute('SELECT * FROM reminders').fetchone()
        self.assertEqual(row['source_message_id'], 2)
        self.assertEqual(row['due_at'], NOW + 3780)
        self.assertIn('Время текущего сообщения: ' + local_time(NOW + 3600), model.inputs[0][0]['content'])

    async def test_exact_validation_error_is_given_back_to_model(self):
        self.store.enqueue(update('Нужно напомнить про зубы через 3 минуты'), NOW)
        model = Model(answer(call('create_reminder', {'content': 'Зубы', 'due_at': 'bad-date'})),
                      answer(call('create_reminder', {'content': 'Зубы', 'due_at': local_time(NOW + 180)}, 'fixed')),
                      answer(text='Создано.'))
        await self.runtime(model).process_once()
        error = json.loads(model.inputs[1][-1]['content'])['error']
        self.assertEqual(error['code'], 'invalid_arguments')
        self.assertEqual(error['details'][0]['field'], 'due_at')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 1)

    async def test_clarification_survives_reopening_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bot.db'
            self.store.db.close()
            self.store = Store(path)
            self.store.enqueue(update('напомни завтра купить молоко'), NOW)
            first = Model(answer(call('ask_clarification', {'question': 'Во сколько?', 'kind': 'reminder'})))
            await self.runtime(first).process_once()
            self.store.db.close()
            self.store = Store(path)
            self.store.enqueue(update('В девять утра', 2), NOW + 60)
            second = Model(answer(call('create_reminder', {'content': 'Купить молоко', 'due_at': '2026-10-17T09:00:00+03:00', 'continue_request': True})), answer(text='Напомню 17 октября в 09:00 (Киров).'))
            await self.runtime(second, NOW + 60).process_once()
            row = self.store.db.execute('SELECT * FROM reminders').fetchone()
            self.assertEqual(row['source_message_id'], 1)
            self.assertEqual(local_time(row['due_at']), '2026-10-17T09:00:00+03:00')
            self.assertIsNone(self.store.get_meta('pending'))

    async def test_crash_after_tool_commit_resumes_without_duplicate(self):
        self.store.enqueue(update(), NOW)
        job = self.store.next_job()
        model = Model(answer(call('create_note', {'content': 'Факт'})), RuntimeError('network'))
        with self.assertRaises(RuntimeError):
            await Agent(self.store, model, lambda: NOW).process(job)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
        await Agent(self.store, Model(answer(text='Сохранено.')), lambda: NOW).process(self.store.next_job())
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_paraphrased_question_uses_notes_and_can_scan(self):
        self.store.enqueue(update(), NOW)
        ToolRunner(self.store).run(call('create_note', {'content': 'Оля дала обещание отказаться от сладких напитков', 'event_date': '2026-10-09'}), self.store.next_job(), NOW)
        with self.store.db:
            self.store.db.execute("UPDATE jobs SET status='done'")
        self.store.enqueue(update('Когда Оля сказала, что больше не будет пить газировку?', 2), NOW)
        model = Model(answer(call('search_notes', {'query': 'газировку'}, 'search')),
                      answer(call('list_notes', {}, 'list')),
                      answer(text='9 октября 2026 года, памятка №1.'))
        await self.runtime(model).process_once()
        self.assertIn('2026-10-09', model.inputs[-1][-1]['content'])
        self.assertIn('9 октября', self.store.history()[-1]['content'])

    async def test_other_user_is_ignored_without_model_call(self):
        data = update()
        data['message']['from']['id'] = 9876
        self.store.enqueue(data, NOW)
        await self.runtime().process_once()
        self.assertEqual(self.store.next_job(), None)
        self.assertEqual(self.store.history(), [])
        self.assertIsNone(self.store.next_outbox(NOW + 5))

    async def test_group_is_ignored(self):
        data = update()
        data['message']['chat']['type'] = 'group'
        self.store.enqueue(data, NOW)
        await self.runtime().process_once()
        self.assertEqual(self.store.history(), [])

    async def test_reminder_delivery_does_not_call_model(self):
        self.store.enqueue(update('напомни через пять минут проверить духовку'), NOW)
        ToolRunner(self.store).run(call('create_reminder', {'content': 'Духовка', 'due_at': local_time(NOW + 300)}), self.store.next_job(), NOW)
        rt = self.runtime(now=NOW + 3600)
        rt.schedule_once()
        await rt.deliver_once()
        self.assertIn('после срока', self.telegram.sent[0][1])
        self.assertEqual(self.telegram.sent[0][2], 1)

    async def test_delivery_failure_keeps_reminder_for_retry(self):
        with self.store.db:
            self.store.queue_message('test', 'hello', now=NOW)
        self.telegram.error = TelegramError(429, 60)
        await self.runtime().deliver_once()
        self.assertIsNone(self.store.next_outbox(NOW + 59))
        self.telegram.error = None
        await self.runtime(now=NOW + 60).deliver_once()
        self.assertEqual(len(self.telegram.sent), 1)

    async def test_poll_durably_advances_offset_and_deduplicates(self):
        self.telegram.incoming = [update(), update()]
        await self.runtime().poll_once()
        self.assertEqual(self.store.get_meta('offset'), 2)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM jobs').fetchone()[0], 1)

    async def test_model_failure_does_not_claim_success(self):
        self.store.enqueue(update(), NOW)
        await self.runtime(Model(RuntimeError('secret-value'))).process_once()
        self.assertIn('Не удалось', self.store.history()[-1]['content'])
        self.assertNotIn('secret-value', self.store.history()[-1]['content'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)

    async def test_snooze_receipt_is_available_at_click_time(self):
        self.store.enqueue(update(), NOW)
        reminder = ToolRunner(self.store).run(call('create_reminder', {
            'content': 'Чай', 'due_at': local_time(NOW + 300)}),
            self.store.next_job(), NOW)
        with self.store.db:
            self.store.db.execute("UPDATE jobs SET status='done'")
        self.store.prepare_reminders(NOW + 300, lambda _: 'Чай')
        row = self.store.next_outbox(NOW + 300)
        self.store.sent(row['key'], 4, NOW + 300)
        callback = {'update_id': 2, 'callback_query': {'id': 'cb',
            'from': {'id': OWNER}, 'data': f"snooze:{reminder['id']}",
            'message': {'chat': {'id': OWNER, 'type': 'private'}}}}
        self.store.enqueue(callback, NOW + 400)
        await self.runtime(now=NOW + 400).process_once()
        row = self.store.next_outbox(NOW + 400)
        self.assertIsNotNone(row)
        self.assertEqual(row['text'], 'Отложено на 10 минут.')

    async def test_cancel_during_send_is_rejected_but_can_retry_after_explicit_delivery_failure(self):
        self.store.enqueue(update(), NOW)
        runner = ToolRunner(self.store)
        job = self.store.next_job()
        reminder = runner.run(call('create_reminder', {'content': 'Чай',
            'due_at': local_time(NOW + 300)}), job, NOW)
        rt = self.runtime(now=NOW + 300)
        rt.schedule_once()
        async def cancel_during_request(*args):
            result = runner.run(call('cancel_reminder', {'id': reminder['id']}, 'cancel'), job, NOW + 300)
            self.assertEqual(result['error']['code'], 'delivery_in_progress')
            raise TelegramError(500)
        self.telegram.send = cancel_during_request
        await rt.deliver_once()
        self.assertIsNotNone(self.store.next_outbox(NOW + 400))
        result = runner.run(call('cancel_reminder', {'id': reminder['id']}, 'cancel-again'), job, NOW + 301)
        self.assertIn('cancelled_reminder_id', result)
        self.assertIsNone(self.store.next_outbox(NOW + 400))
        self.assertEqual(self.store.db.execute('SELECT status FROM reminders WHERE id=?',
            (reminder['id'],)).fetchone()[0], 'cancelled')

    async def test_deepseek_http_contract(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'deepseek-flash')
            self.assertEqual(body['thinking'], {'type': 'disabled'})
            self.assertIn('create_note', [t['function']['name'] for t in body['tools']])
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': answer(text='Привет')} ]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await DeepSeek(client, 'fake').complete([{'role': 'user', 'content': 'Привет'}])
            self.assertEqual(result['content'], 'Привет')

    async def test_telegram_http_plain_text_and_button(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertNotIn('parse_mode', body)
            self.assertEqual(body['reply_markup']['inline_keyboard'][0][0]['callback_data'], 'snooze:12')
            return httpx.Response(200, json={'ok': True, 'result': {'message_id': 9}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await Telegram(client, 'fake').send(OWNER, '<text>', 12)
            self.assertEqual(result['message_id'], 9)

    async def test_callback_replay_snoozes_once(self):
        self.store.enqueue(update(), NOW)
        r = ToolRunner(self.store).run(call('create_reminder', {'content': 'Чай', 'due_at': local_time(NOW + 300)}), self.store.next_job(), NOW)
        with self.store.db:
            self.store.db.execute("UPDATE jobs SET status='done'")
        self.store.prepare_reminders(NOW + 300, lambda _: 'Чай')
        row = self.store.next_outbox(NOW + 300)
        self.store.sent(row['key'], 4, NOW + 300)
        cb = {'update_id': 2, 'callback_query': {'id': 'cb', 'from': {'id': OWNER}, 'data': f"snooze:{r['id']}",
              'message': {'chat': {'id': OWNER, 'type': 'private'}}}}
        self.store.enqueue(cb, NOW + 400)
        await self.runtime(now=NOW + 400).process_once()
        cb['update_id'] = 3
        self.store.enqueue(cb, NOW + 500)
        await self.runtime(now=NOW + 500).process_once()
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM reminders WHERE status='pending'").fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
