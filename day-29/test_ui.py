import asyncio
import tempfile
import unittest
from pathlib import Path
from textual.widgets import DataTable, Input, Select, Static, Switch, TextArea

from core import load_report
from engine import Engine
from test_optimization import CASE, FakeJudge, FakeLocal, FakeMonitor
from tui import OptimizeApp


class FakeManager:
    owned = False
    def __init__(self):
        self.requests = 0
    async def snapshot(self):
        self.requests += 1
        return {'available': False, 'models': [], 'running': [], 'owned': False}
    async def stop(self):
        pass


class UITests(unittest.IsolatedAsyncioTestCase):
    async def test_compare_and_open_without_calls_at_three_sizes(self):
        for size in [(180, 60), (120, 40), (80, 24)]:
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                local = FakeLocal()
                engine = Engine(local=local, judge=FakeJudge(), monitor_factory=FakeMonitor,
                                directory=Path(directory))
                app = OptimizeApp(engine=engine, manager=FakeManager())
                async with app.run_test(size=size) as pilot:
                    app.case_values = [dict(CASE, id='opt-empty')]
                    app.query_one('#case', Select).value = 'opt-empty'
                    await pilot.pause()
                    app.action_compare()
                    await asyncio.wait_for(app.generation_task, 5)
                    await pilot.pause()
                    self.assertEqual(engine.report['comparisons'][0]['after']['status'], 'complete')
                    self.assertIn('Ответ', app.query_one('#after-answer')._markdown)
                    self.assertIn('S1', app.query_one('#sources', TextArea).text)
                    self.assertEqual(engine.report['cases'][0]['id'], 'opt-empty')
                    self.assertFalse(app.query_one('#compare').disabled)
                    first_parameters = str(app.query_one('#before-profile', Static).content)
                    second_parameters = str(app.query_one('#after-profile', Static).content)
                    self.assertIn('Prompt: Исходный | Рассуждения: включены', first_parameters)
                    self.assertIn('Prompt: Короткий | Рассуждения: выключены', second_parameters)
                    for parameters in [first_parameters, second_parameters]:
                        self.assertIn('Модель: qwen3:14b', parameters)
                        self.assertIn('Температура: 0', parameters)
                        self.assertIn('Контекст: 16384 токенов', parameters)
                        self.assertIn('Лимит ответа: 3072 токенов', parameters)
                    for identifier, title in [('before', 'Первый профиль'), ('after', 'Второй профиль')]:
                        selector = app.query_one('#' + identifier, Select)
                        editor = app.query_one('#' + identifier + '-editor')
                        self.assertEqual(editor.border_title, title)
                        self.assertGreaterEqual(editor.region.width, len(title)+2)
                        self.assertEqual(selector.parent, editor)
                        for name in ('temperature', 'context', 'limit', 'thinking'):
                            field = app.query_one('#' + identifier + '-' + name)
                            self.assertEqual(field.parent.parent, editor)
                        self.assertIn(title, str(app.query_one('#' + identifier + '-title', Static).content))
                    columns = [str(c.label) for c in app.query_one('#comparisons', DataTable).columns.values()]
                    self.assertIn('Первый профиль', columns)
                    self.assertIn('Второй профиль', columns)
                path = next(Path(directory).glob('*.json'))
                count = len(local.calls)
                manager = FakeManager()
                loaded = OptimizeApp(engine=engine, report=load_report(path), manager=manager)
                async with loaded.run_test(size=size) as pilot:
                    await pilot.pause()
                    self.assertEqual(len(local.calls), count)
                    self.assertEqual(len(loaded.engine.report['comparisons']), 1)
                    self.assertEqual(manager.requests, 0)
                    caption = str(loaded.query_one('#result-caption', Static).content)
                    self.assertIn('opt-empty', caption)
                    self.assertIn('повтор 1', caption)
                    saved_title = str(loaded.query_one('#after-title', Static).content)
                    self.assertEqual(str(loaded.query_one('#before-profile', Static).content), first_parameters)
                    self.assertEqual(str(loaded.query_one('#after-profile', Static).content), second_parameters)
                    loaded.query_one('#after', Select).value = 'temperature-02'
                    await pilot.pause()
                    self.assertEqual(str(loaded.query_one('#after-title', Static).content), saved_title)
                    self.assertEqual(str(loaded.query_one('#result-caption', Static).content), caption)
                    self.assertEqual(str(loaded.query_one('#after-profile', Static).content), second_parameters)

    async def test_independent_profile_settings_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(local=FakeLocal(), judge=FakeJudge(), monitor_factory=FakeMonitor,
                            directory=Path(directory))
            app = OptimizeApp(engine=engine, manager=FakeManager())
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one('#after', Select).value = 'temperature-02'
                await pilot.pause()
                self.assertEqual(app.query_one('#after-temperature', Input).value, '0.2')
                self.assertEqual(float(app.query_one('#before-temperature', Input).value), 0)
                app.query_one('#before-temperature', Input).value = '0.5'
                app.query_one('#before-limit', Input).value = '1536'
                app.query_one('#before-thinking', Switch).value = False
                app.query_one('#after-context', Input).value = '8192'
                app.query_one('#after-limit', Input).value = '3072'
                before, after = app.chosen_profiles()
                self.assertEqual(before['settings']['temperature'], 0.5)
                self.assertEqual(before['settings']['num_predict'], 1536)
                self.assertFalse(before['settings']['thinking'])
                self.assertEqual(before['settings']['num_ctx'], 16384)
                self.assertEqual(after['settings']['temperature'], 0.2)
                self.assertEqual(after['settings']['num_predict'], 3072)
                self.assertEqual(after['settings']['num_ctx'], 8192)
                self.assertEqual(app.options['baseline']['settings']['temperature'], 0)
                app.query_one('#question', TextArea).load_text('Тест независимых настроек')
                app.query_one('#case', Select).set_options([('Тест', 'test')])
                app.case_values = [dict(CASE, id='test')]
                app.query_one('#case', Select).value = 'test'
                await pilot.pause()
                app.action_compare()
                for side in ('before', 'after'):
                    self.assertTrue(app.query_one('#' + side + '-context', Input).disabled)
                await asyncio.wait_for(app.generation_task, 5)
                await pilot.pause()
                comparison = engine.report['comparisons'][0]
                self.assertEqual(comparison['before']['profile'], before)
                self.assertEqual(comparison['after']['profile'], after)
                app.query_one('#before', Select).value = 'no-thinking'
                await pilot.pause()
                self.assertEqual(float(app.query_one('#before-temperature', Input).value), 0)
                self.assertEqual(app.query_one('#before-limit', Input).value, '3072')
                self.assertEqual(app.query_one('#after-context', Input).value, '8192')
                app.query_one('#before-limit', Input).value = '8192'
                app.query_one('#before-context', Input).value = '8192'
                with self.assertRaisesRegex(ValueError, 'Первый профиль'):
                    app.chosen_profiles()
                app.query_one('#before-limit', Input).value = '1536'
                app.query_one('#after-limit', Input).value = '8192'
                with self.assertRaisesRegex(ValueError, 'Второй профиль'):
                    app.chosen_profiles()


if __name__ == '__main__':
    unittest.main()
