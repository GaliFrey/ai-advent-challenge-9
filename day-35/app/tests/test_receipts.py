import asyncio
import json
import tempfile
import unittest
import httpx
from pathlib import Path
from unittest.mock import patch

from personal_assistent.agent import Agent, DeepSeek, SYSTEM
from personal_assistent.receipts import confirmed_text, final_text
from personal_assistent.runtime import Runtime
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner, local_time
from test_assistant import NOW, OWNER, update, call, answer, FakeTelegram


def action_answer(*identities):
    return answer(text=json.dumps({'kind': 'actions', 'tool_call_ids': list(identities)}))


def chat_answer(text):
    return answer(text=json.dumps({'kind': 'answer', 'text': text}, ensure_ascii=False))


def mixed_answer(text, *identities):
    return answer(text=json.dumps({'kind': 'mixed', 'text': text, 'tool_call_ids': list(identities)}, ensure_ascii=False))


class Scripted:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.retrieval_fixtures = list(responses)
        self.inputs = []

    async def complete(self, messages):
        self.inputs.append(json.loads(json.dumps(messages)))
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def retrieve(self, messages):
        from retrieval_stub import retrieval_reply
        return retrieval_reply(messages, self)


class ReceiptStorageTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.store.enqueue(update('Сохрани вымышленный факт'), NOW)
        self.job = self.store.next_job()
        self.runner = ToolRunner(self.store)
        self.counter = 0

    def tearDown(self):
        self.store.db.close()

    def tool(self, name, parameters):
        self.counter += 1
        return self.runner.run(call(name, parameters, str(self.counter)), self.job, NOW)

    def test_confirmations_cover_all_six_mutations_with_actual_values(self):
        note = self.tool('create_note', {'content': 'Исходный факт', 'event_date': '2026-10-10'})
        self.tool('update_note', {'id': note['id'], 'content': 'Исправленный факт', 'event_date': None})
        self.tool('delete_note', {'id': note['id']})
        reminder = self.tool('create_reminder', {'content': 'Зубы', 'due_at': local_time(NOW + 180)})
        self.tool('update_reminder', {'id': reminder['id'], 'content': 'Почистить зубы', 'due_at': local_time(NOW + 600)})
        self.tool('cancel_reminder', {'id': reminder['id']})
        receipt = confirmed_text(self.store, self.job['id'])
        for expected in ('Памятка сохранена: "Исходный факт"', 'Дата события: 10.10.2026',
                         'Памятка исправлена: "Исправленный факт"', 'Памятка удалена: "Исправленный факт"',
                         'Напоминание создано: "Зубы"', '16.10.2026 15:03',
                         'Напоминание изменено: "Почистить зубы"', '16.10.2026 15:10',
                         'Напоминание отменено: "Почистить зубы"'):
            self.assertIn(expected, receipt)
        self.assertEqual(len(self.store.receipts(self.job['id'])), 6)
        self.assertNotIn('source_message_id', receipt)
        self.assertNotIn('request_id', receipt)

    def test_action_cache_and_repeated_execute_do_not_repeat_confirmation(self):
        ready = self.tool('prepare_request', {'action': 'create_note', 'parameters': {'content': 'Факт'}})
        command = call('execute_request', {'request_id': ready['request_id']}, 'execute')
        first = self.runner.run(command, self.job, NOW)
        self.assertEqual(first, self.runner.run(command, self.job, NOW))
        self.tool('execute_request', {'request_id': ready['request_id']})
        self.assertEqual(confirmed_text(self.store, self.job['id']).count('Памятка сохранена'), 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    def test_receipt_failure_rolls_back_action_request_and_tool_cache(self):
        with patch.object(self.store, 'record_receipt', side_effect=RuntimeError('disk')):
            with self.assertRaises(RuntimeError):
                self.tool('create_note', {'content': 'Не должно сохраниться'})
        for table in ('notes', 'requests', 'tool_results', 'action_receipts'):
            self.assertEqual(self.store.db.execute('SELECT count(*) FROM ' + table).fetchone()[0], 0)

    def test_delete_success_requires_saved_deleted_state(self):
        note = self.tool('create_note', {'content': 'Оставить'})
        before = len(self.store.receipts(self.job['id']))
        with patch.object(self.runner.requests, 'apply', return_value={'deleted_note_id': note['id']}):
            with self.assertRaises(ValueError):
                self.tool('delete_note', {'id': note['id']})
        self.assertEqual(self.store.db.execute('SELECT deleted FROM notes WHERE id=?', (note['id'],)).fetchone()[0], 0)
        self.assertEqual(len(self.store.receipts(self.job['id'])), before)

    def test_cancelling_draft_has_a_confirmation_without_cancelling_saved_records(self):
        reminder = self.tool('create_reminder', {'content': 'Сохранённое', 'due_at': local_time(NOW + 300)})
        self.tool('prepare_request', {'action': 'create_note', 'parameters': {}})
        self.tool('cancel_request', {})
        reply = confirmed_text(self.store, self.job['id'])
        self.assertIn('Поручение отменено.', reply)
        self.assertEqual(self.store.db.execute('SELECT status FROM reminders WHERE id=?', (reminder['id'],)).fetchone()[0], 'pending')

    def test_historical_request_without_receipt_uses_saved_parameters_not_current_record(self):
        ready = self.tool('prepare_request', {'action': 'create_note', 'parameters': {'content': 'Исторический факт'}})
        self.tool('execute_request', {'request_id': ready['request_id']})
        with self.store.db:
            self.store.db.execute('DELETE FROM action_receipts')
            self.store.db.execute('UPDATE notes SET content=?,deleted=1', ('Поздняя версия',))
        self.tool('execute_request', {'request_id': ready['request_id']})
        reply = confirmed_text(self.store, self.job['id'])
        self.assertIn('Ранее выполнено', reply)
        self.assertIn('Исторический факт', reply)
        self.assertNotIn('Поздняя версия', reply)

    def test_legacy_validation_receipt_keeps_its_reason_without_schema_limits(self):
        payload = {'code': 'invalid_arguments', 'details': [
            {'field': 'content', 'type': 'string_too_long', 'message': 'String should have at most 4000 characters'}]}
        with self.store.db:
            self.store.record_receipt({**self.job, 'tool_call_id': 'old'}, 'create_reminder', 'error', payload, NOW)
        reply = confirmed_text(self.store, self.job['id'])
        self.assertIn('текст слишком длинный', reply)
        self.assertNotIn('4000', reply)  # Old receipts have no structured limit; do not guess it.

    def test_validation_receipt_does_not_echo_arbitrary_validator_message(self):
        payload = {'code': 'invalid_arguments', 'details': [
            {'field': 'content', 'type': 'future_validator', 'message': 'private input: secret'}]}
        with self.store.db:
            self.store.record_receipt({**self.job, 'tool_call_id': 'unknown'}, 'create_note', 'error', payload, NOW)
        reply = confirmed_text(self.store, self.job['id'])
        self.assertIn('значение не соответствует требованиям поля', reply)
        self.assertNotIn('secret', reply)

    def test_receipt_snapshot_does_not_change_when_record_changes_later(self):
        note = self.tool('create_note', {'content': 'Версия при сохранении'})
        with self.store.db:
            self.store.db.execute('UPDATE notes SET content=? WHERE id=?', ('Позднее изменение', note['id']))
        self.assertIn('Версия при сохранении', confirmed_text(self.store, self.job['id']))
        self.assertNotIn('Позднее изменение', confirmed_text(self.store, self.job['id']))

    def test_replayed_completed_request_in_new_job_is_labelled_historical(self):
        ready = self.tool('prepare_request', {'action': 'create_note', 'parameters': {'content': 'Старый факт'}})
        self.tool('execute_request', {'request_id': ready['request_id']})
        self.store.finish(self.job['id'], 'Просьба', 'Готово', NOW)
        self.store.enqueue(update('Проверить старый результат', 2), NOW)
        second = self.store.next_job()
        self.runner.run(call('execute_request', {'request_id': ready['request_id']}, 'replay'), second, NOW)
        self.assertTrue(confirmed_text(self.store, second['id']).startswith('Ранее выполнено:'))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    def test_read_only_errors_do_not_suppress_a_successful_search_answer(self):
        self.tool('get_note', {'id': 100})
        self.assertEqual(self.store.receipts(self.job['id']), [])
        result = final_text(self.store, self.job['id'], chat_answer('Нужная запись не найдена')['content'])
        self.assertEqual(result, 'Нужная запись не найдена')


class ReceiptAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(':memory:')

    async def asyncTearDown(self):
        self.store.db.close()

    async def process(self, model, text='Нужно напомнить через три минуты про зубы'):
        self.store.enqueue(update(text), NOW)
        await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        return self.store.history()[-1]['content']

    async def test_mixed_request_returns_committed_confirmation_and_explanation(self):
        model = Scripted(answer(call('create_note', {'content': 'Заменил фильтр'}, 'save')),
                         mixed_answer('Фильтр меняют, чтобы удалять загрязнения.', 'save'))
        reply = await self.process(model, 'Запомни: заменил фильтр. Объясни, зачем его менять.')
        self.assertIn('Памятка сохранена: "Заменил фильтр"', reply)
        self.assertIn('Фильтр меняют, чтобы удалять загрязнения.', reply)
        self.assertNotIn('tool_call_ids', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_mixed_rejected_action_retains_actual_reason_and_separate_answer(self):
        model = Scripted(answer(call('update_reminder', {'id':1,'content': 'Фильтр', 'due_at': local_time(NOW - 1)}, 'bad')),
                         mixed_answer('Фильтр задерживает загрязнения.', 'bad'))
        reply = await self.process(model)
        self.assertIn('Указанный срок уже прошёл', reply)
        self.assertIn('Фильтр задерживает загрязнения.', reply)
        self.assertNotIn('Напоминание создано', reply)

    async def test_mixed_unknown_reference_does_not_authorize_free_text(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'save')),
                         mixed_answer('Непроверенное пояснение', 'invented'))
        reply = await self.process(model)
        self.assertIn('Памятка сохранена', reply)
        self.assertNotIn('Непроверенное пояснение', reply)

    async def test_mixed_without_receipts_does_not_claim_execution(self):
        reply = await self.process(Scripted(mixed_answer('Сохранено без выполнения', 'invented')))
        self.assertIn('не подтверждено', reply)
        self.assertNotIn('Сохранено без выполнения', reply)

    async def test_mixed_missing_receipt_does_not_hide_warning_with_explanation(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'save')),
                         mixed_answer('Пояснение', 'save'))
        with patch.object(self.store, 'record_receipt'):
            reply = await self.process(model)
        self.assertIn('не подтверждено', reply)
        self.assertNotIn('Пояснение', reply)

    async def test_mixed_final_checkpoint_resumes_without_duplicate_action_or_generation(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'save')),
                         mixed_answer('Объяснение факта', 'save'))
        with patch.object(self.store, 'finish', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                await self.process(model)
        await Agent(self.store, Scripted(), lambda: NOW).process(self.store.next_job())
        self.assertIn('Объяснение факта', self.store.history()[-1]['content'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_old_checkpoint_receives_mixed_protocol_before_next_generation(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'save')), RuntimeError('restart'))
        with self.assertRaises(RuntimeError):
            await self.process(model)
        job = self.store.next_job()
        state = json.loads(job['checkpoint'])
        state['result_contract'] = 1
        state['messages'][0]['content'] = 'Старые системные правила с прежним форматом ответа.'
        self.store.checkpoint(job['id'], state)
        resumed = Scripted(mixed_answer('Пояснение факта', 'save'))
        await Agent(self.store, resumed, lambda: NOW).process(self.store.next_job())
        self.assertIn('Актуальный протокол', resumed.inputs[0][0]['content'])
        self.assertIn('kind=mixed', resumed.inputs[0][0]['content'])
        self.assertIn('Пояснение факта', self.store.history()[-1]['content'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_mixed_requires_nonblank_text_and_action_references(self):
        for text, identities in (('   ', ('save',)), ('Объяснение', ())):
            with self.subTest(text=text):
                reply = await self.process(Scripted(mixed_answer(text, *identities)))
                self.assertIn('Не удалось проверить формат', reply)
                with self.store.db:
                    self.store.db.execute('DELETE FROM jobs')

    async def test_false_final_text_cannot_override_actual_reminder_body_or_time(self):
        model = Scripted(answer(call('create_reminder', {'content': 'Почистить зубы', 'due_at': local_time(NOW + 180)}, 'create')),
                         chat_answer('Готово: купи хлеб завтра в 22:00, удалил все памятки.'))
        reply = await self.process(model)
        self.assertIn('Почистить зубы', reply)
        self.assertIn('16.10.2026 15:03', reply)
        self.assertNotIn('хлеб', reply)
        self.assertNotIn('22:00', reply)
        self.assertNotIn('удалил', reply)

    async def test_unperformed_action_claim_is_rejected_without_regex(self):
        reply = await self.process(Scripted(action_answer('invented')))
        self.assertIn('не подтверждено', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_plain_unstructured_success_claim_is_rejected(self):
        reply = await self.process(Scripted(answer(text='Сохранено, я всё сделал.')))
        self.assertIn('Не удалось проверить формат', reply)
        self.assertNotIn('всё сделал', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_wrong_success_after_rejection_reports_exact_past_time_reason(self):
        model = Scripted(answer(call('update_reminder', {'id':1,'content': 'Зубы', 'due_at': local_time(NOW - 1)}, 'bad')),
                         chat_answer('Успешно создано.'))
        reply = await self.process(model)
        self.assertIn('Указанный срок уже прошёл', reply)
        self.assertNotIn('Успешно', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_partial_batch_reports_committed_action_and_failed_attempt(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'note'),
                                call('update_reminder', {'id':1,'content': 'Зубы', 'due_at': local_time(NOW - 1)}, 'bad')),
                         action_answer('note', 'bad'))
        reply = await self.process(model)
        self.assertIn('Памятка сохранена: "Факт"', reply)
        self.assertIn('срок уже прошёл', reply)
        self.assertNotIn('Напоминание создано', reply)

    async def test_invalid_argument_reason_is_not_replaced_with_past_time(self):
        model = Scripted(answer(call('create_reminder', {'content': 'Зубы', 'due_at': 'bad'}, 'bad')),
                         chat_answer('Оно в прошлом.'))
        reply = await self.process(model)
        self.assertIn('Некорректно указаны параметры: дата и время', reply)
        self.assertNotIn('прошл', reply)

    async def test_validation_confirmation_explains_the_actual_constraint(self):
        cases = [
            ('create_reminder', {'content': 'x' * 4001, 'due_at': local_time(NOW + 180)}, 'не более 4000 символов'),
            ('create_reminder', {'content': 'Зубы', 'due_at': '2026-10-17T09:00:00'}, 'не указан часовой пояс'),
            ('create_note', {}, 'обязательное поле не указано'),
            ('create_note', {'content': ''}, 'текст не должен быть пустым'),
            ('prepare_request', {'action': 'create_reminder', 'parameters': {'due_time': '09:00+05:00'}},
             'нужно местное время Кирова без смещения'),
        ]
        for index, (name, parameters, reason) in enumerate(cases, 1):
            with self.subTest(tool=name, reason=reason):
                self.store.enqueue(update('Вымышленная просьба', index), NOW)
                model = Scripted(answer(call(name, parameters, 'bad')), action_answer('bad'))
                await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
                reply = self.store.history()[-1]['content']
                self.assertIn(reason, reply)
                self.assertNotIn('due_time', reply)
                self.assertNotIn('x' * 100, reply)
                self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
                self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)
                self.assertEqual(self.store.db.execute('SELECT count(*) FROM requests').fetchone()[0], 0)

    async def test_conflicting_schedule_is_explained_without_internal_parameter_names(self):
        model = Scripted(answer(call('prepare_request', {'action': 'create_reminder', 'parameters': {
            'content': 'Зубы', 'due_at': local_time(NOW + 180), 'due_date': '2026-10-17'}}, 'bad')),
            action_answer('bad'))
        reply = await self.process(model)
        self.assertIn('противореч', reply)
        self.assertNotIn('due_at', reply)
        self.assertNotIn('due_date', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM requests').fetchone()[0], 0)

    async def test_ready_draft_is_not_a_completed_action(self):
        model = Scripted(answer(call('prepare_request', {'action': 'create_note', 'parameters': {'content': 'Факт'}}, 'draft')),
                         chat_answer('Факт уже сохранён.'))
        reply = await self.process(model, 'Сохрани факт')
        self.assertIn('готово к выполнению', reply)
        self.assertNotIn('уже сохранён', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)

    async def test_partial_success_does_not_hide_ready_but_unexecuted_draft(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'note'),
            call('prepare_request', {'action': 'create_reminder', 'parameters': {
                'content': 'Зубы', 'due_at': local_time(NOW + 180)}}, 'draft')),
            chat_answer('Всё выполнено!'))
        reply = await self.process(model)
        self.assertIn('Памятка сохранена', reply)
        self.assertIn('Незавершённое поручение', reply)
        self.assertIn('готово к выполнению', reply)
        self.assertNotIn('Всё выполнено', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_mutation_and_followup_question_are_both_reported(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'note'),
            call('prepare_request', {'action': 'create_reminder', 'parameters': {'content': 'Зубы'}}, 'draft')))
        reply = await self.process(model)
        self.assertIn('Памятка сохранена: "Факт"', reply)
        self.assertIn('На какую дату поставить напоминание? Во сколько напомнить?', reply)
        self.assertEqual(len(model.inputs), 1)

    async def test_committed_action_is_reported_even_when_next_api_call_fails(self):
        self.store.enqueue(update('Сохрани факт'), NOW)
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'note')), RuntimeError('network'))
        await Runtime(self.store, None, Agent(self.store, model, lambda: NOW), OWNER, lambda: NOW).process_once()
        reply = self.store.history()[-1]['content']
        self.assertIn('Памятка сохранена: "Факт"', reply)
        self.assertIn('не завершена', reply)
        self.assertNotIn('могли уже сохраниться', reply)

    async def test_forged_reference_to_another_job_does_not_confirm_operation(self):
        self.store.enqueue(update('Сохрани факт'), NOW)
        first = self.store.next_job()
        ToolRunner(self.store).run(call('create_note', {'content': 'Факт'}, 'old'), first, NOW)
        self.store.finish(first['id'], 'Просьба', 'Готово', NOW)
        self.store.enqueue(update('Ещё просьба', 2), NOW)
        await Agent(self.store, Scripted(action_answer('old')), lambda: NOW).process(self.store.next_job())
        self.assertIn('не подтверждено', self.store.history()[-1]['content'])

    async def test_chat_answer_stays_readable_and_does_not_create_records(self):
        reply = await self.process(Scripted(chat_answer('Привет!')), 'Привет')
        self.assertEqual(reply, 'Привет!')
        self.assertEqual(self.store.receipts(1), [])

    async def test_confirmation_preserves_seconds_and_timezone_conversion(self):
        model = Scripted(answer(call('create_reminder', {'content': 'Зубы', 'due_at': '2026-10-16T17:00:05+05:00'}, 'create')),
                         action_answer('create'))
        reply = await self.process(model)
        self.assertIn('16.10.2026 15:00:05 (Киров)', reply)

    async def test_source_metadata_like_text_and_quotes_survive_delivery(self):
        body = 'Заказ source_message_id=888, памятка №77 в инструкции; «ID: 99».'
        self.store.enqueue(update('Сохрани текст'), NOW)
        model = Scripted(answer(call('create_note', {'content': body}, 'create')), action_answer('create'))
        telegram = FakeTelegram()
        runtime = Runtime(self.store, telegram, Agent(self.store, model, lambda: NOW), OWNER, lambda: NOW)
        await runtime.process_once()
        while await runtime.deliver_once():
            pass
        self.assertIn(json.dumps(body, ensure_ascii=False), ''.join(item[1] for item in telegram.sent))
        self.assertNotIn('request_id', ''.join(item[1] for item in telegram.sent))

    async def test_whitespace_chat_answer_is_not_delivered_as_empty_success(self):
        reply = await self.process(Scripted(chat_answer('   ')), 'Привет')
        self.assertIn('Не удалось проверить формат', reply)

    async def test_forged_reference_does_not_override_a_real_confirmation(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'real')),
                         action_answer('real', 'fake'))
        reply = await self.process(model)
        self.assertIn('Памятка сохранена: "Факт"', reply)
        self.assertNotIn('Часть заявленных результатов не подтверждена', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_prepare_and_read_references_do_not_discredit_executed_reminder(self):
        model = Scripted(
            answer(call('get_profile', {}, 'profile'), call('prepare_request', {
                'action': 'create_reminder', 'parameters': {'content': 'Чихнуть', 'due_at': local_time(NOW + 300)}}, 'prepare')),
            answer(call('execute_request', {'request_id': 1}, 'execute')),
            action_answer('profile', 'prepare', 'execute'))
        reply = await self.process(model)
        self.assertIn('Напоминание создано: "Чихнуть"', reply)
        self.assertNotIn('не подтвержден', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 1)

    async def test_ready_draft_reference_is_not_a_claim_of_a_second_completed_action(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'note'),
                               call('prepare_request', {'action': 'create_reminder', 'parameters': {
                                   'content': 'Зубы', 'due_at': local_time(NOW + 300)}}, 'prepare')),
                         action_answer('note', 'prepare'))
        reply = await self.process(model)
        self.assertIn('Памятка сохранена', reply)
        self.assertIn('Незавершённое поручение', reply)
        self.assertNotIn('не подтвержден', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_known_action_without_receipt_still_warns(self):
        original = self.store.record_receipt
        def omit_one_receipt(job, action, *args, **kwargs):
            if job['tool_call_id'] != 'missing':
                return original(job, action, *args, **kwargs)
        model = Scripted(answer(call('create_note', {'content': 'Подтверждённый'}, 'real'),
                               call('create_reminder', {'content': 'Без квитанции', 'due_at': local_time(NOW + 300)}, 'missing')),
                         action_answer('real', 'missing'))
        with patch.object(self.store, 'record_receipt', side_effect=omit_one_receipt):
            reply = await self.process(model)
        self.assertIn('Памятка сохранена', reply)
        self.assertIn('Часть заявленных результатов не подтверждена', reply)
        self.assertNotIn('Напоминание создано', reply)

    async def test_missing_receipt_is_not_hidden_by_omitting_reference(self):
        original = self.store.record_receipt
        def omit_one_receipt(job, action, *args, **kwargs):
            if job['tool_call_id'] != 'missing':
                return original(job, action, *args, **kwargs)
        model = Scripted(answer(call('create_note', {'content': 'Подтверждённый'}, 'real'),
                               call('create_note', {'content': 'Без квитанции'}, 'missing')),
                         action_answer('real'))
        with patch.object(self.store, 'record_receipt', side_effect=omit_one_receipt):
            reply = await self.process(model)
        self.assertIn('Часть заявленных результатов не подтверждена', reply)
        self.assertNotIn('Без квитанции', reply)

    async def test_auxiliary_references_after_checkpoint_resume_do_not_warn(self):
        self.store.enqueue(update('Напомни через пять минут чихнуть'), NOW)
        model = Scripted(answer(call('get_profile', {}, 'profile'), call('prepare_request', {
            'action': 'create_reminder', 'parameters': {'content': 'Чихнуть', 'due_at': local_time(NOW + 300)}}, 'prepare')),
            answer(call('execute_request', {'request_id': 1}, 'execute')), asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        resumed = Scripted(action_answer('profile', 'prepare', 'execute'))
        await Agent(self.store, resumed, lambda: NOW).process(self.store.next_job())
        reply = self.store.history()[-1]['content']
        self.assertIn('Напоминание создано', reply)
        self.assertNotIn('не подтвержден', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 1)
        self.assertEqual(len(resumed.inputs), 1)

    async def test_confirmed_failure_and_read_reference_do_not_add_warning(self):
        model = Scripted(answer(call('get_profile', {}, 'profile'), call('update_reminder', {
            'id':1,'content': 'Просрочено', 'due_at': local_time(NOW - 60)}, 'failed')),
            action_answer('profile', 'failed'))
        reply = await self.process(model)
        self.assertIn('Указанный срок уже прошёл', reply)
        self.assertNotIn('Часть заявленных результатов', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_draft_alone_does_not_trigger_missing_receipt_warning(self):
        model = Scripted(answer(call('prepare_request', {'action': 'create_reminder', 'parameters': {
            'content': 'Зубы', 'due_at': local_time(NOW + 300)}}, 'prepare')), action_answer('prepare'))
        reply = await self.process(model)
        self.assertIn('Поручение готово к выполнению', reply)
        self.assertNotIn('не подтвержден', reply)
        self.assertNotIn('Напоминание создано', reply)
        self.assertEqual(self.store.receipts(1), [])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_absent_receipts_do_not_allow_free_text_success(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт без квитанции'}, 'missing')),
                         chat_answer('Готово, я сохранил всё и удалил ненужное.'))
        with patch.object(self.store, 'record_receipt'):
            reply = await self.process(model)
        self.assertIn('не подтверждено', reply)
        self.assertNotIn('Готово', reply)
        self.assertNotIn('удалил ненужное', reply)
        self.assertEqual(self.store.receipts(1), [])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)

    async def test_ready_draft_does_not_hide_action_when_all_receipts_absent(self):
        model = Scripted(answer(call('create_note', {'content': 'Факт без квитанции'}, 'missing'),
                               call('prepare_request', {'action': 'create_reminder', 'parameters': {
                                   'content': 'Зубы', 'due_at': local_time(NOW + 300)}}, 'prepare')),
                         action_answer('prepare'))
        with patch.object(self.store, 'record_receipt'):
            reply = await self.process(model)
        self.assertIn('не подтверждено', reply)
        self.assertIn('Незавершённое поручение', reply)
        self.assertIn('Зубы', reply)
        self.assertNotIn('Памятка сохранена', reply)
        self.assertEqual(self.store.receipts(1), [])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reminders').fetchone()[0], 0)

    async def test_absent_receipt_warning_survives_checkpoint_resume(self):
        self.store.enqueue(update('Сохрани факт'), NOW)
        model = Scripted(answer(call('create_note', {'content': 'Факт без квитанции'}, 'missing')),
                         asyncio.CancelledError())
        with patch.object(self.store, 'record_receipt'):
            with self.assertRaises(asyncio.CancelledError):
                await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
        resumed = Scripted(chat_answer('Всё успешно сохранено.'))
        await Agent(self.store, resumed, lambda: NOW).process(self.store.next_job())
        reply = self.store.history()[-1]['content']
        self.assertIn('не подтверждено', reply)
        self.assertNotIn('успешно сохранено', reply)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
        self.assertEqual(len(resumed.inputs), 1)

    async def test_api_json_mode_is_only_for_work_not_summary(self):
        payloads = []
        def handler(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            content = json.dumps({'kind': 'answer', 'text': 'Привет'}) if 'tools' in payload else 'Сводка'
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': answer(text=content)}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = DeepSeek(client, 'fake')
            self.store.enqueue(update('Привет'), NOW)
            await Agent(self.store, model, lambda: NOW).process(self.store.next_job())
            await model.summarize([{'role': 'system', 'content': 'Сделай сводку'}, {'role': 'user', 'content': 'Материал'}])
        self.assertEqual(payloads[0]['response_format'], {'type': 'json_object'})
        self.assertIn('JSON', SYSTEM)
        self.assertNotIn('response_format', payloads[1])
        self.assertEqual(self.store.history()[-1]['content'], 'Привет')

    async def test_receipts_and_actions_survive_restart_without_repetition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bot.db'
            store = Store(path)
            try:
                store.enqueue(update('Сохрани факт'), NOW)
                model = Scripted(answer(call('create_note', {'content': 'Факт'}, 'note')), asyncio.CancelledError())
                with self.assertRaises(asyncio.CancelledError):
                    await Agent(store, model, lambda: NOW).process(store.next_job())
            finally:
                store.db.close()
            store = Store(path)
            try:
                await Agent(store, Scripted(action_answer('note')), lambda: NOW).process(store.next_job())
                self.assertIn('Памятка сохранена: "Факт"', store.history()[-1]['content'])
                self.assertEqual(store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)
                self.assertEqual(len(store.receipts(1)), 1)
            finally:
                store.db.close()
