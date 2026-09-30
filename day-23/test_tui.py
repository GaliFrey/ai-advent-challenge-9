import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from textual.widgets import DataTable, Input, LoadingIndicator, Static, TabbedContent

from pipeline import Runner
from test_pipeline import CASES, FakeRetriever, FakeReranker
from tui import RagApp


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_is_visible_and_logged_without_starting_runner(self):
        with tempfile.TemporaryDirectory() as directory, patch('tui.DAY', Path(directory)), \
                patch.dict('os.environ', {'DEEPSEEK_API_KEY': ''}):
            app = RagApp(cases=CASES, runner_factory=lambda **kwargs: self.fail('Must not launch'),
                         log_directory=Path(directory) / 'logs')
            async with app.run_test(size=(180, 60)) as pilot:
                self.assertIn('НЕТ КЛЮЧА', str(app.query_one('#status', Static).content))
                await pilot.click('#ask')
                self.assertFalse(app.running)
                self.assertIn('ЗАПУСК ОСТАНОВЛЕН', str(app.query_one('#status', Static).content))
                events = [json.loads(line) for line in app.session_log.path.read_text().splitlines()]
                self.assertEqual(events[-1]['event'], 'stage_failed')
                self.assertEqual(events[-1]['stage'], 'configuration')

    async def test_pending_model_request_has_indicator_stage_and_elapsed_time(self):
        entered, release = Event(), Event()

        def factory(**kwargs):
            def completion(messages, key, model):
                if not entered.is_set():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('Test request was not released')
                return {'answer': '{"query": "Поисковая формулировка"}' if 'query' in messages[0]['content'] else 'Ответ',
                        'usage': {}, 'llm_seconds': 0}
            return Runner(**kwargs, completion=completion, retriever_factory=FakeRetriever,
                          reranker_factory=FakeReranker)

        with tempfile.TemporaryDirectory() as directory, patch('tui.DAY', Path(directory)), \
                patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-secret'}):
            app = RagApp(cases=CASES, runner_factory=factory, log_directory=Path(directory) / 'logs')
            async with app.run_test(size=(180, 60)) as pilot:
                try:
                    await pilot.click('#ask')
                    for _ in range(100):
                        if entered.is_set():
                            break
                        await asyncio.sleep(.02)
                    self.assertTrue(entered.is_set())
                    await pilot.pause(.3)
                    self.assertTrue(app.running)
                    self.assertTrue(app.query_one('#busy', LoadingIndicator).display)
                    status = str(app.query_one('#status', Static).content)
                    self.assertIn('DeepSeek формирует ответ', status)
                    self.assertIn('прошло', status)
                    self.assertNotIn('test-secret', status)
                finally:
                    release.set()
                await app.workers.wait_for_complete()
                self.assertFalse(app.query_one('#busy', LoadingIndicator).display)

    async def test_live_tabs_candidates_results_and_saved_replay(self):
        def factory(**kwargs):
            def completion(messages, key, model):
                return {'answer': '{"query": "Поисковая формулировка"}' if 'query' in messages[0]['content'] else 'Результат [S1]',
                        'usage': {'total_tokens': 10}, 'llm_seconds': .01}
            return Runner(**kwargs, completion=completion, retriever_factory=FakeRetriever, reranker_factory=FakeReranker)

        with tempfile.TemporaryDirectory() as directory, patch('tui.DAY', Path(directory)), \
                patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-secret'}):
            app = RagApp(cases=CASES, runner_factory=factory, log_directory=Path(directory) / 'logs')
            async with app.run_test(size=(180, 60)) as pilot:
                app.query_one('#before', Input).value = '3'
                app.query_one('#after', Input).value = '2'
                await pilot.click('#ask')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(app.running)
                self.assertEqual(app.report['status'], 'complete')
                self.assertEqual(app.query_one('#chunks-rewrite_rerank', DataTable).row_count, 3)
                app.query_one('#modes', TabbedContent).active = 'pane-rewrite_rerank'
                await pilot.pause()
                self.assertIn('Поисковая формулировка', str(app.query_one('#query-rewrite_rerank', Static).content))
                self.assertIn('запрос изменён', str(app.query_one('#query-rewrite_rerank', Static).content))
                self.assertIn('Результат', str(app.query_one('#answer-rewrite_rerank', Static).content))
                self.assertIn('Фрагмент', str(app.query_one('#chunk-text-rewrite_rerank', Static).content))
                app.query_one('#modes', TabbedContent).active = 'pane-summary'
                await pilot.pause()
                for mode in ('baseline', 'rewrite', 'rewrite_filter', 'rewrite_rerank'):
                    self.assertIn('Результат', str(app.query_one(f'#compare-answer-{mode}', Static).content))
                    self.assertIn('day-20/README.md', str(app.query_one(f'#compare-sources-{mode}', Static).content))
                self.assertIn('Поисковая формулировка', str(app.query_one('#comparison-note', Static).content))
                saved = json.loads(app.report_path.read_text())
            with patch('urllib.request.urlopen', side_effect=AssertionError('Replay must be offline')):
                replay = RagApp(report=saved, cases=CASES, log_directory=Path(directory) / 'logs')
                async with replay.run_test(size=(180, 60)):
                    self.assertIn('Результат', str(replay.query_one('#answer-baseline', Static).content))
                    self.assertIn('Результат', str(replay.query_one('#compare-answer-rewrite_rerank', Static).content))
                saved['items'][0]['rewrite']['query'] = CASES[0]['question']
                saved['items'][0]['rewrite'].pop('changed', None)
                for result in saved['items'][0]['results'].values():
                    result['query'] = CASES[0]['question']
                unchanged = RagApp(report=saved, cases=CASES, log_directory=Path(directory) / 'logs')
                async with unchanged.run_test(size=(180, 60)):
                    for mode in ('rewrite', 'rewrite_filter', 'rewrite_rerank'):
                        self.assertIn('запрос не изменён', str(unchanged.query_one(f'#query-{mode}', Static).content))
                    self.assertIn('запрос не изменён', str(unchanged.query_one('#comparison-note', Static).content))
