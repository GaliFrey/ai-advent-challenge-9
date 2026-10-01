import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import DataTable, Input, Static, TabbedContent
from pipeline import Runner
from test_pipeline import CASES, FakeRetriever, FakeReranker, completion
from tui import RagApp


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_evidence_context_llm_assessment_and_offline_replay(self):
        def factory(**kwargs):
            return Runner(**kwargs, completion=completion, retriever_factory=FakeRetriever, reranker_factory=FakeReranker)
        with tempfile.TemporaryDirectory() as directory, patch('tui.DAY', Path(directory)), patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-secret'}):
            app = RagApp(cases=CASES, runner_factory=factory, log_directory=Path(directory) / 'logs')
            async with app.run_test(size=(180, 60)) as pilot:
                await pilot.click('#ask')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(app.report['status'], 'complete')
                self.assertIn('Очищаются', str(app.query_one('#answer', Static).content))
                self.assertIn('Новая задача', str(app.query_one('#quotes', Static).content))
                self.assertIn('chunk-1', str(app.query_one('#quotes', Static).content))
                self.assertEqual(app.report['items'][0]['judge']['assessment']['verdict'], 'pass')
                self.assertIn('Подтверждено цитатой', str(app.query_one('#review', Static).content))
                self.assertFalse(app.query('#pass'))
                self.assertFalse(app.query('#note'))
                app.query_one('#tabs', TabbedContent).active = 'pane-context'
                await pilot.pause()
                self.assertEqual(app.query_one('#chunks', DataTable).row_count, 1)
                app.query_one('#tabs', TabbedContent).active = 'pane-summary'
                await pilot.pause()
                self.assertEqual(app.query_one('#summary', DataTable).row_count, 1)
                await pilot.click('#demo')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(len(app.report['items']), 10)
                self.assertTrue(all(i['judge']['status'] == 'complete' for i in app.report['items']))
                self.assertEqual(app.query_one('#summary', DataTable).row_count, 10)
                saved = json.loads(app.report_path.read_text())
                path = app.report_path
            with patch('urllib.request.urlopen', side_effect=AssertionError('Offline replay')):
                replay = RagApp(report=saved, report_path=path, cases=CASES, log_directory=Path(directory) / 'logs')
                async with replay.run_test(size=(120, 40)):
                    self.assertIn('Подтверждено цитатой', str(replay.query_one('#review', Static).content))
                    self.assertIn('Очищаются', str(replay.query_one('#answer', Static).content))
            saved['items'][0].pop('judge')
            saved['items'][0]['review'] = {'verdict': 'pass', 'note': 'Ручная оценка старого отчёта'}
            legacy = RagApp(report=saved, report_path=path, cases=CASES, log_directory=Path(directory) / 'logs')
            async with legacy.run_test(size=(180, 60)):
                text = str(legacy.query_one('#review', Static).content)
                self.assertIn('Оценка LLM отсутствует', text)
                self.assertIn('Старая ручная оценка', text)

    async def test_missing_key_does_not_start(self):
        with tempfile.TemporaryDirectory() as directory, patch('tui.DAY', Path(directory)), patch.dict('os.environ', {'DEEPSEEK_API_KEY': ''}):
            app = RagApp(cases=CASES, runner_factory=lambda **kwargs: self.fail('Unexpected run'), log_directory=Path(directory) / 'logs')
            async with app.run_test(size=(180, 60)) as pilot:
                self.assertIn('НЕТ КЛЮЧА', str(app.query_one('#status', Static).content))
                await pilot.click('#ask')
                self.assertFalse(app.running)
                self.assertIn('ЗАПУСК ОСТАНОВЛЕН', str(app.query_one('#status', Static).content))

    async def test_ten_questions_refusal_and_invalid_answers_are_visible(self):
        def factory(**kwargs):
            def invalid(messages, key, model):
                if 'sources' not in json.loads(messages[1]['content']):
                    return completion(messages, key, model)
                return {'answer': 'Выдуманная цитата', 'usage': {}, 'llm_seconds': 0}
            return Runner(**kwargs, completion=invalid, retriever_factory=FakeRetriever, reranker_factory=FakeReranker)
        with tempfile.TemporaryDirectory() as directory, patch('tui.DAY', Path(directory)), patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test'}):
            app = RagApp(cases=CASES, runner_factory=factory, log_directory=Path(directory) / 'logs')
            async with app.run_test(size=(180, 60)) as pilot:
                await pilot.click('#demo')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(len(app.report['items']), 10)
                self.assertEqual(app.report['status'], 'invalid')
                self.assertIn('ОТКЛОНЁН', str(app.query_one('#checks', Static).content))
                self.assertIn('Не выполнялась', str(app.query_one('#review', Static).content))
                app.query_one('#threshold', Input).value = '.9'
                await pilot.click('#ask')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertIn('Не знаю', str(app.query_one('#answer', Static).content))
                self.assertIn('Уточните', str(app.query_one('#answer', Static).content))
                self.assertIn('генерация не вызывалась', str(app.query_one('#checks', Static).content))
