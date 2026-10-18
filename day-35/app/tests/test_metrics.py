import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from personal_assistent.agent import DeepSeek
from personal_assistent.config import Config
from personal_assistent.storage import Store
from personal_assistent.runtime import Runtime


def completion(usage=None, finish='stop'):
    result = {'choices': [{'finish_reason': finish, 'message': {'role': 'assistant', 'content': '4'}}]}
    if usage is not None:
        result['usage'] = usage
    return result


class MetricsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(':memory:')

    async def asyncTearDown(self):
        self.store.db.close()

    async def request(self, payload, status=200):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status, json=payload))) as client:
            return await DeepSeek(client, 'fake', on_call=self.store.record_llm_call).complete([{'role': 'user', 'content': '2+2?'}])

    async def test_generation_settings_and_usage_are_recorded(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['temperature'], 0.2)
            self.assertEqual(body['max_tokens'], 1500)
            self.assertEqual(body['model'], 'deepseek-flash')
            return httpx.Response(200, json=completion({'prompt_tokens': 100, 'completion_tokens': 5, 'prompt_cache_hit_tokens': 20}))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await DeepSeek(client, 'fake', temperature=.2, max_tokens=1500,
                on_call=self.store.record_llm_call).complete([{'role': 'user', 'content': '2+2?'}])
        self.assertEqual(result['content'], '4')
        row = self.store.db.execute('SELECT * FROM llm_calls').fetchone()
        self.assertEqual((row['status'], row['prompt_tokens'], row['completion_tokens'], row['cached_tokens']), ('ok', 100, 5, 20))
        self.assertGreaterEqual(row['duration'], 0)
        self.assertIn('Входные токены: 100.', self.store.stats_text())

    async def test_missing_or_invalid_usage_is_unknown_not_zero(self):
        for usage in (None, {'prompt_tokens': True, 'completion_tokens': -1}, {'prompt_tokens': '10'}):
            await self.request(completion(usage))
        rows = self.store.db.execute('SELECT * FROM llm_calls').fetchall()
        self.assertTrue(all(r['prompt_tokens'] is None and r['completion_tokens'] is None for r in rows))
        self.assertIn('Входные токены: нет данных.', self.store.stats_text())
        self.assertIn('Из кеша: нет данных.', self.store.stats_text())

    async def test_cache_details_fallback_and_invalid_cache_count(self):
        await self.request(completion({'prompt_tokens': 100,
            'prompt_tokens_details': {'cached_tokens': 20}}))
        await self.request(completion({'prompt_tokens': 100, 'prompt_cache_hit_tokens': 200}))
        rows = self.store.db.execute('SELECT cached_tokens FROM llm_calls ORDER BY id').fetchall()
        self.assertEqual([r[0] for r in rows], [20, None])

    async def test_metrics_failure_does_not_discard_answer_or_mask_http_error(self):
        def broken_recorder(record):
            raise RuntimeError('private-storage-path')
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion()))) as client:
            with self.assertLogs('personal_assistent.agent', level='WARNING') as logs:
                result = await DeepSeek(client, 'fake', on_call=broken_recorder).complete([])
            self.assertEqual(result['content'], '4')
            self.assertNotIn('private-storage-path', ''.join(logs.output))
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503, json={}))) as client:
            with self.assertLogs('personal_assistent.agent', level='WARNING'):
                with self.assertRaises(httpx.HTTPStatusError):
                    await DeepSeek(client, 'fake', on_call=broken_recorder).complete([])

    async def test_failed_http_call_counts_as_attempt_without_leaking_body(self):
        with self.assertRaises(httpx.HTTPStatusError):
            await self.request({'error': 'private-payload'}, 503)
        row = self.store.db.execute('SELECT * FROM llm_calls').fetchone()
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['error_type'], 'HTTPStatusError')
        self.assertNotIn('private-payload', self.store.stats_text())
        self.assertIn('ошибки 1', self.store.stats_text())

    async def test_truncated_completion_retains_usage_and_failure_reason(self):
        with self.assertRaises(ValueError):
            await self.request(completion({'prompt_tokens': 100, 'completion_tokens': 2000}, finish='length'))
        row = self.store.db.execute('SELECT * FROM llm_calls').fetchone()
        self.assertEqual((row['status'], row['finish_reason'], row['completion_tokens']), ('failed', 'length', 2000))

    async def test_cancelled_request_is_counted_separately(self):
        async def handler(request):
            raise asyncio.CancelledError
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(asyncio.CancelledError):
                await DeepSeek(client, 'fake', on_call=self.store.record_llm_call).complete([])
        self.assertIn('отменено 1', self.store.stats_text())

    async def test_partial_usage_reports_coverage(self):
        await self.request(completion({'prompt_tokens': 100, 'completion_tokens': 5}))
        await self.request(completion())
        self.assertIn('100 (данные 1/2 вызовов)', self.store.stats_text())
        self.assertIn('Вызовов API: 2', self.store.stats_text())

    async def test_empty_answer_is_failed_before_agent_receives_it(self):
        payload = completion({'prompt_tokens': 2, 'completion_tokens': 0})
        payload['choices'][0]['message']['content'] = ''
        with self.assertRaises(ValueError):
            await self.request(payload)
        self.assertIn('ошибки 1', self.store.stats_text())

    async def test_malformed_tool_calls_are_failed_before_checkpoint(self):
        valid_call = {'id': 'call1', 'type': 'function',
                      'function': {'name': 'list_notes', 'arguments': '{}'}}
        cases = [
            ('tool_calls', 'not-a-list'),
            ('tool_calls', []),
            ('tool_calls', [{}]),
            ('tool_calls', [{**valid_call, 'function': {'name': 'list_notes', 'arguments': {}}}]),
            ('tool_calls', [{**valid_call, 'id': ''}]),
            ('tool_calls', [valid_call, valid_call]),
            ('stop', [valid_call]),
        ]
        for finish, calls in cases:
            with self.subTest(finish=finish, calls=calls):
                payload = completion({'prompt_tokens': 5, 'completion_tokens': 1}, finish)
                payload['choices'][0]['message']['tool_calls'] = calls
                with self.assertRaises(ValueError):
                    await self.request(payload)
        rows = self.store.db.execute('SELECT status FROM llm_calls').fetchall()
        self.assertEqual([row[0] for row in rows], ['failed'] * len(cases))

    async def test_valid_tool_call_preserves_arguments_for_domain_validation(self):
        payload = completion({'prompt_tokens': 5, 'completion_tokens': 1}, 'tool_calls')
        payload['choices'][0]['message'].update(content=None, tool_calls=[{
            'id': 'call1', 'type': 'function', 'function': {'name': 'create_note', 'arguments': '{bad-json'}}])
        # Broken JSON arguments belong to the existing precise tool-error path;
        # the API envelope itself is valid and must reach the agent.
        result = await self.request(payload)
        self.assertEqual(result['tool_calls'][0]['function']['arguments'], '{bad-json')
        self.assertEqual(self.store.db.execute('SELECT status FROM llm_calls').fetchone()[0], 'ok')

    async def test_stats_command_uses_no_model_and_does_not_enter_history(self):
        update = {'update_id': 1, 'message': {'message_id': 1, 'date': 100,
            'from': {'id': 42}, 'chat': {'id': 42, 'type': 'private'}, 'text': '/stats'}}
        self.store.enqueue(update, 100)
        # No Telegram or model implementation is needed for this local command.
        await Runtime(self.store, None, None, 42, lambda: 100).process_once()
        self.assertEqual(self.store.history(), [])
        self.assertIsNone(self.store.next_job())
        self.assertIn('Статистики API пока нет', self.store.next_outbox(100)['text'])


class PersistentMetricsTests(unittest.TestCase):
    def test_schema_upgrade_and_metrics_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bot.db'
            store = Store(path)
            with store.db:
                store.set_meta('existing_data', 'kept')
                store.db.execute('DROP TABLE llm_calls')
            store.db.close()
            store = Store(path)
            self.assertEqual(store.get_meta('existing_data'), 'kept')
            store.record_llm_call({'created_at': 100, 'duration': .5, 'status': 'ok',
                'error_type': None, 'finish_reason': 'stop', 'prompt_tokens': 10,
                'completion_tokens': 2, 'cached_tokens': 0})
            store.db.close()
            store = Store(path)
            self.assertIn('Вызовов API: 1', store.stats_text())
            self.assertIn('Входные токены: 10.', store.stats_text())
            store.db.close()

    def test_generation_settings_defaults_and_validation(self):
        base = {'TELEGRAM_BOT_TOKEN': 'fake', 'TELEGRAM_OWNER_ID': '42', 'DEEPSEEK_API_KEY': 'fake'}
        with patch('personal_assistent.config.load_dotenv'), patch.dict(os.environ, base, clear=True):
            config = Config.load()
            self.assertEqual((config.temperature, config.max_tokens), (0., 2000))
        invalid = [('DEEPSEEK_TEMPERATURE', 'nan'), ('DEEPSEEK_TEMPERATURE', 'inf'),
                   ('DEEPSEEK_TEMPERATURE', '3'), ('DEEPSEEK_MAX_TOKENS', '1'),
                   ('DEEPSEEK_MAX_TOKENS', '2.5'), ('DEEPSEEK_MAX_TOKENS', '9000')]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                with patch('personal_assistent.config.load_dotenv'), patch.dict(os.environ, {**base, key: value}, clear=True):
                    with self.assertRaises(ValueError):
                        Config.load()
