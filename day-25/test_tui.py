import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from textual.widgets import Static, TextArea, TabbedContent, DataTable, Select
from tui import ChatApp
from test_chat import factory
from scenarios import SCENARIOS


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_row_stays_on_review_and_preserves_selection_during_progress(self):
        import copy
        from benchmark import run_scenario
        from pipeline import save_report

        with tempfile.TemporaryDirectory() as tmp, patch('tui.DAY', Path(tmp)):
            path = Path(tmp) / 'sessions' / 'scenario.json'
            session = run_scenario(factory(key='fake'), SCENARIOS[0], path, validation_only=True)
            session['turns'][1]['assessment']['result']['explanation'] = 'Оценка второго хода'
            save_report(path, session)
            activations = []

            def capture(message):
                # The hook sees the same message at each parent as it bubbles.
                if isinstance(message, TabbedContent.TabActivated) and not any(
                    message is previous for previous in activations
                ):
                    activations.append(message)

            app = ChatApp(session_path=path)
            async with app.run_test(size=(180, 60), message_hook=capture) as pilot:
                tabs = app.query_one('#tabs', TabbedContent)
                tabs.active = 'checks-tab'
                await pilot.pause()
                activations.clear()
                await pilot.click('#summary', offset=(5, 2))
                await pilot.pause()
                self.assertEqual(app.selected, 1)
                self.assertEqual(tabs.active, 'checks-tab')
                self.assertEqual(activations, [])
                self.assertIn('Оценка второго хода', str(app.query_one('#assessment', Static).content))
                app.receive(copy.deepcopy(session), 'judge')
                await pilot.pause()
                self.assertEqual(app.selected, 1)
                self.assertEqual(app.query_one('#summary', DataTable).cursor_row, 1)
                self.assertEqual(tabs.active, 'checks-tab')
                self.assertEqual(activations, [])
                await pilot.press('enter')
                await pilot.pause()
                self.assertEqual(tabs.active, 'checks-tab')
                self.assertEqual(activations, [])
                await pilot.click('#inspect-sources')
                await pilot.pause()
                self.assertEqual(tabs.active, 'sources-tab')
                self.assertEqual(app.query_one('#turns', Select).value, 1)
                self.assertIn('Реплика 2:', str(app.query_one('#query', Static).content))
                self.assertEqual(len(activations), 1)

    async def test_worker_progress_does_not_start_multiprocessing_tracker(self):
        from tqdm import tqdm
        from test_chat import FakeRetriever, FakeReranker, scripted_completion
        from pipeline import Runner

        class ProgressRetriever(FakeRetriever):
            def search(self, query, k):
                # SentenceTransformer constructs tqdm even when progress is disabled.
                list(tqdm([query], disable=True))
                return super().search(query, k)

        def runner(**kwargs):
            return Runner(**kwargs, completion=scripted_completion,
                          retriever_factory=ProgressRetriever, reranker_factory=FakeReranker)

        with tempfile.TemporaryDirectory() as tmp, patch('tui.DAY', Path(tmp)), patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'fake'}):
            with patch('tqdm.std.TqdmDefaultWriteLock.create_mp_lock', side_effect=AssertionError('Unexpected multiprocessing')):
                app = ChatApp(runner_factory=runner)
                async with app.run_test(size=(180, 60)) as pilot:
                    app.query_one('#input', TextArea).load_text(SCENARIOS[0]['turns'][0]['question'])
                    await pilot.click('#send')
                    await app.workers.wait_for_complete()
                    self.assertEqual(app.session['turns'][-1]['status'], 'complete')

    async def test_fullscreen_manual_chat_inspection_and_offline_replay(self):
        with tempfile.TemporaryDirectory() as tmp, patch('tui.DAY', Path(tmp)), patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'fake'}):
            app = ChatApp(runner_factory=factory)
            async with app.run_test(size=(180, 60)) as pilot:
                self.assertFalse(app.query('#demo'))
                self.assertFalse(app.query('#scenario'))
                app.query_one('#input', TextArea).load_text(SCENARIOS[0]['turns'][0]['question'])
                await pilot.press('enter')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(len(app.session['turns']), 1)
                self.assertIn('Рабочая память', str(app.query_one('#chat', Static).content))
                self.assertIn('реплика 1', str(app.query_one('#memory', Static).content))
                self.assertGreater(app.query_one('#chat-scroll').size.width, 100)
                app.query_one('#tabs', TabbedContent).active = 'sources-tab'
                await pilot.pause()
                self.assertEqual(app.query_one('#chunks', DataTable).row_count, 1)
                self.assertIn('c1', str(app.query_one('#chunk', Static).content))
                app.query_one('#tabs', TabbedContent).active = 'dialog-tab'
                for case in SCENARIOS[0]['turns'][1:]:
                    app.query_one('#input', TextArea).load_text(case['question'])
                    await pilot.click('#send')
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                self.assertEqual(len(app.session['turns']), 12)
                self.assertEqual(app.query_one('#summary', DataTable).row_count, 12)
                path = app.path
                self.assertEqual(path.parent, Path(tmp) / 'sessions')
                await pilot.click('#new')
                self.assertFalse(app.session['turns'])
                self.assertFalse(app.session['state']['goal'])
                app.query_one('#sessions', Select).value = str(path)
                await pilot.click('#open')
                self.assertEqual(len(app.session['turns']), 12)
                self.assertEqual(app.path, path)
            with patch('urllib.request.urlopen', side_effect=AssertionError('Network during replay')):
                replay = ChatApp(session_path=path, runner_factory=lambda **kw: self.fail('Unexpected runner'))
                async with replay.run_test(size=(100, 40)) as pilot:
                    await pilot.pause()
                    self.assertFalse(replay.query_one('#memory-scroll').display)
                    self.assertIn('первоначальной цели', str(replay.query_one('#chat', Static).content))
                    replay.query_one('#turns', Select).value = 0
                    await pilot.pause()
                    self.assertIn('реплика 1', str(replay.query_one('#memory', Static).content))

    async def test_missing_key_does_not_start(self):
        with tempfile.TemporaryDirectory() as tmp, patch('tui.DAY', Path(tmp)), patch.dict('os.environ', {'DEEPSEEK_API_KEY': ''}):
            app = ChatApp(runner_factory=lambda **kw: self.fail('Unexpected runner'))
            async with app.run_test(size=(180, 60)) as pilot:
                app.query_one('#input', TextArea).load_text('test')
                await pilot.click('#send')
                self.assertFalse(app.running)
                self.assertIn('НЕТ КЛЮЧА', str(app.query_one('#status', Static).content))
