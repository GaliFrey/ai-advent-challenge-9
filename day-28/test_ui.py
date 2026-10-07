"""UI regressions: live context, display-only JSON streaming and compact layouts."""
import asyncio
import json
import tempfile
import unittest

import httpx
from textual.widgets import Collapsible, Input, Select, Static, Switch, TabbedContent, TextArea

from core import display_answer, streamed_answer
from ollama import OllamaManager
from pipeline import Runner
from test_rag import FakeDocs, FakeModel, answer
from tui import RagApp


class StreamTextTests(unittest.TestCase):
    def test_invalid_refusal_shows_explanation_without_accepting_citations(self):
        raw = json.dumps({'status': 'unknown', 'answer': '',
            'gaps': 'Trim и DeleteTutor не удаляют дубли из массива.',
            'citations': [{'source_id': 'S1', 'quote': 'wrong'}]}, ensure_ascii=False)
        result = {'status': 'invalid', 'raw': raw}
        text = display_answer(result)
        self.assertIn('Модель не нашла ответа', text)
        self.assertIn('Trim и DeleteTutor', text)
        self.assertIn('Ответ не прошёл проверку', text)
        self.assertNotIn('citations', text)
        self.assertNotIn('wrong', text)
        self.assertNotIn('response', result)
        result = {'status': 'complete', 'response': {'status': 'unknown', 'answer': '',
                                                    'gaps': 'Недостаточно документов.'}}
        self.assertIn('Недостаточно документов.', display_answer(result))
        self.assertNotIn('не прошёл проверку', display_answer(result))

    def test_incomplete_answer_decodes_escapes_without_exposing_json(self):
        raw = json.dumps({'status':'answered','answer':'Строка\n"цитата" \\ путь 😀','citations':[]}, ensure_ascii=True)
        expected = 'Строка\n"цитата" \\ путь 😀'
        for i in range(len(raw)+1):
            actual = streamed_answer(raw[:i])
            self.assertTrue(expected.startswith(actual), (i,actual))
            self.assertNotIn('"status"', actual)
            self.assertFalse(any(0xD800 <= ord(c) <= 0xDFFF for c in actual))
        self.assertEqual(streamed_answer(raw), expected)
        self.assertEqual(streamed_answer('{"answer":"hello\\'), 'hello')
        self.assertEqual(streamed_answer('{"answer":"hello\\u12'), 'hello')
        self.assertEqual(streamed_answer('{"answer":"hello\\u123z'), '')

    def test_only_root_answer_and_invalid_output_remains_diagnostic(self):
        raw = '{"nested":{"answer":"wrong"},"answer":"right"}'
        self.assertEqual(streamed_answer(raw),'right')
        self.assertEqual(streamed_answer('{"answer":[],'),'')
        self.assertEqual(streamed_answer('not json'),'')
        invalid = display_answer({'status':'invalid','raw':'{"answer":"draft","citations":[]}'})
        self.assertIn('Ответ не прошёл проверку',invalid)
        self.assertIn('draft',invalid)
        self.assertNotIn('citations',invalid)


class DelayedDocs(FakeDocs):
    def __init__(self):
        super().__init__()
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.sources[0]['title'] = 'Источник нового вопроса'

    async def retrieve(self, queries, settings, notify):
        self.entered.set()
        await self.release.wait()
        return await super().retrieve(queries, settings, notify)


class DraftModel(FakeModel):
    async def generate(self, messages, settings, update, *, planning=False):
        update('{"status":"answered","answer":"Новый ответ\\nс пояснением [S1]", "citations":[', '')
        self.entered.set()
        await asyncio.Event().wait()


class UiRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_refusal_remains_readable_and_check_errors_stay_in_diagnostics(self):
        raw = json.dumps({'status': 'unknown', 'answer': '', 'gaps': 'Нужной функции нет в найденных документах.',
                          'citations': [{'source_id': 'S1', 'quote': 'wrong'}]}, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as tmp:
            app = RagApp(directory=tmp, ollama=self.manager(), runner=Runner(FakeDocs(), FakeModel(raw=raw), FakeModel()))
            async with app.run_test(size=(120, 40)) as pilot:
                app.query_one('#planner', Switch).value = False
                app.query_one('#question', TextArea).load_text('Как удалить дубли?')
                app.action_send()
                await app.generation_task
                await pilot.pause()
                self.assertEqual(app.selected()['results']['local']['status'], 'invalid')
                text = app.query_one('#local-answer', Static).content.markup
                self.assertIn('Нужной функции нет', text)
                self.assertIn('Ответ не прошёл проверку', text)
                self.assertNotIn('Цитата отсутствует', str(app.query_one('#local-metrics', Static).content))
                self.assertIn('Цитата отсутствует', str(app.query_one('#local-diagnostics', Static).content))

    def manager(self):
        def offline(request):
            raise httpx.ConnectError('offline',request=request)
        return OllamaManager(transport=httpx.MockTransport(offline))

    async def test_current_turn_and_source_picker_update_before_generation_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = RagApp(directory=tmp,ollama=self.manager(),runner=Runner(FakeDocs(),FakeModel(),FakeModel()))
            async with app.run_test(size=(120,40)) as pilot:
                await pilot.pause()
                app.query_one('#planner',Switch).value=False
                app.query_one('#question',TextArea).load_text('Первый вопрос')
                app.action_send(); await app.generation_task; await pilot.pause()
                delayed, draft = DelayedDocs(), DraftModel()
                app.runner.docs, app.runner.local = delayed, draft
                app.query_one('#question',TextArea).load_text('Второй вопрос')
                app.action_send(); await delayed.entered.wait(); await pilot.pause(.25)
                try:
                    self.assertEqual(app.query_one('#turns',Select).value,1)
                    self.assertNotIn('ArrayOptFirstElem',str(app.query_one('#sources',Select)._options))
                    self.assertNotIsInstance(app.query_one('#sources',Select).value,int)
                    delayed.release.set(); await draft.entered.wait(); await pilot.pause(.25)
                    picker = app.query_one('#sources',Select)
                    self.assertIn('Источник нового вопроса',str(picker._options))
                    self.assertEqual(picker.value,0)
                    self.assertIn('Источник нового вопроса',str(app.query_one('#source-text',Static).content))
                    text = app.query_one('#local-answer',Static).content.markup
                    self.assertEqual(text,'Новый ответ\nс пояснением [S1]')
                    self.assertNotIn('citations',text)
                    self.assertIn('citations',str(app.query_one('#local-diagnostics',Static).content))
                    self.assertIn('ещё не проверено',str(app.query_one('#local-metrics',Static).content))
                    # Browsing the old turn remains possible, but stream updates must not reset its picker.
                    app.query_one('#turns',Select).value=0; await pilot.pause()
                    self.assertEqual(app.index,0)
                    self.assertIn('ArrayOptFirstElem',str(app.query_one('#sources',Select)._options))
                    app.changed(); await pilot.pause(.25)
                    self.assertEqual(app.query_one('#turns',Select).value,0)
                finally:
                    app.action_stop(); await app.generation_task

    async def test_human_readable_memory_search_dynamic_models_and_compact_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = RagApp(directory=tmp,ollama=self.manager(),runner=Runner(FakeDocs(),FakeModel(),FakeModel()))
            async with app.run_test(size=(180,60)) as pilot:
                await pilot.pause()
                app.query_one('#local_model',Input).value='local-new'
                app.query_one('#cloud_model',Input).value='cloud-new'
                app.query_one('#cloud',Switch).value=True
                app.query_one('#notes',TextArea).load_text('Только серверный SP-XML')
                app.query_one('#question',TextArea).load_text('Первый вопрос')
                app.action_send(); await app.generation_task; await pilot.pause()
                brand = str(app.query_one('#brand',Static).content)
                self.assertIn('local-new ↔ cloud-new',brand)
                self.assertNotIn('Qwen3',brand)
                self.assertIn('Завершено',str(app.query_one('#local-metrics',Static).content))
                self.assertNotIn('complete',str(app.query_one('#turns',Select)._options))
                self.assertIn('• ArrayOptFirstElem',str(app.query_one('#search-details',Static).content))
                self.assertNotIn('queries',str(app.query_one('#search-details',Static).content))
                self.assertIn('Только серверный SP-XML',str(app.query_one('#memory-details',Static).content))
                self.assertNotIn('user_questions',str(app.query_one('#memory-details',Static).content))
                self.assertTrue(all(c.collapsed for c in app.query(Collapsible)))
                await pilot.resize_terminal(80,24); await pilot.pause()
                self.assertGreaterEqual(app.query_one('#local-scroll').region.height,2)
                self.assertGreaterEqual(app.query_one('#cloud-scroll').region.height,2)
                self.assertLessEqual(app.query_one('#send').region.bottom,24)
                self.assertIn('Для пустого массива',app.query_one('#local-answer',Static).content.markup)
                await pilot.resize_terminal(120,40); await pilot.pause()
                self.assertFalse(app._compact)
                app.query_one('#cloud',Switch).value=False; await pilot.pause()
                self.assertIn('облако выключено',str(app.query_one('#brand',Static).content))


if __name__ == '__main__':
    unittest.main()
