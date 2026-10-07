"""Ollama lifecycle and TUI tests use only mock HTTP and a fake subprocess."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from textual.widgets import Button, Input, Select, Static, TabbedContent, TextArea

from core import Settings, load_session
from ollama import OllamaManager
from pipeline import Runner
from test_rag import FakeDocs, FakeModel
from tui import RagApp


class FakeServer:
    def __init__(self, available=False):
        self.available = available
        self.requests = []
        self.models = [
            {'name': 'qwen3:14b', 'size': 8*1024**3, 'details': {'parameter_size':'14B','quantization_level':'Q4_K_M'}},
            {'name': 'qwen3:8b', 'size': 5*1024**3, 'details': {'parameter_size':'8B','quantization_level':'Q4_K_M'}}]

    def request(self, request):
        self.requests.append(request.url.path)
        if not self.available:
            raise httpx.ConnectError('offline', request=request)
        return httpx.Response(200, json={'models': self.models if request.url.path == '/api/tags' else [
            {'name': 'qwen3:8b', 'size_vram': 4*1024**3, 'context_length': 16384}]})


class FakeProcess:
    def __init__(self, server, code=None):
        self.returncode, self.server = code, server
        self.terminations = 0
        self.kills = 0

    def terminate(self):
        self.terminations += 1
        self.returncode = 0
        self.server.available = False

    def kill(self):
        self.kills += 1
        self.returncode = -9
        self.server.available = False

    async def wait(self):
        while self.returncode is None:
            await asyncio.sleep(.01)
        return self.returncode


class ManagerTests(unittest.IsolatedAsyncioTestCase):
    def installation(self, directory):
        root = Path(directory)
        (root / 'bin').mkdir()
        (root / 'bin/ollama').write_text('fake binary, never executed')
        (root / 'models').mkdir()
        return root

    async def test_external_server_is_never_started_or_stopped_and_cloud_models_filtered(self):
        server = FakeServer(True)
        server.models.append({'name':'remote:cloud','remote_host':'https://example.invalid'})
        manager = OllamaManager(transport=httpx.MockTransport(server.request))
        with patch('ollama.asyncio.create_subprocess_exec', new_callable=AsyncMock) as spawn:
            state = await manager.start(Settings())
            self.assertTrue(state['available'])
            self.assertFalse(state['owned'])
            self.assertEqual([m['name'] for m in state['models']], ['qwen3:14b','qwen3:8b'])
            await manager.stop()
            spawn.assert_not_awaited()
            self.assertTrue(server.available)
            self.assertEqual(server.requests, ['/api/tags','/api/ps'])

    async def test_explicit_start_sets_local_paths_and_stop_only_owned_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.installation(tmp)
            server, process = FakeServer(), None
            async def spawn(*args, **kwargs):
                nonlocal process
                self.assertEqual(args, (str(root / 'bin/ollama'), 'serve'))
                self.assertEqual(kwargs['env']['OLLAMA_MODELS'], str(root / 'models'))
                self.assertEqual(kwargs['env']['OLLAMA_NO_CLOUD'], '1')
                self.assertEqual(kwargs['env']['OLLAMA_CONTEXT_LENGTH'], '8192')
                self.assertNotIn('DEEPSEEK_API_KEY', kwargs['env'])
                server.available = True
                process = FakeProcess(server)
                return process
            manager = OllamaManager(installation=root, transport=httpx.MockTransport(server.request))
            with patch('ollama.asyncio.create_subprocess_exec', side_effect=spawn) as factory:
                self.assertFalse((await manager.snapshot())['available'])
                factory.assert_not_awaited()
                state = await manager.start(Settings(num_ctx=8192))
                self.assertTrue(state['owned'])
                await manager.stop()
                await manager.stop()
                self.assertEqual(process.terminations, 1)
                self.assertFalse(manager.owned)
                self.assertFalse(server.available)

    async def test_missing_binary_early_exit_and_start_cancellation(self):
        server = FakeServer()
        with tempfile.TemporaryDirectory() as tmp:
            manager = OllamaManager(installation=tmp, transport=httpx.MockTransport(server.request))
            with self.assertRaisesRegex(ValueError, 'Нет установки'):
                await manager.start(Settings())
            root = self.installation(tmp)
            process = FakeProcess(server, code=1)
            with patch('ollama.asyncio.create_subprocess_exec', new=AsyncMock(return_value=process)):
                with self.assertRaisesRegex(ValueError, 'код 1'):
                    await manager.start(Settings())
            process = FakeProcess(server)
            with patch('ollama.asyncio.create_subprocess_exec', new=AsyncMock(return_value=process)):
                task = asyncio.create_task(manager.start(Settings()))
                for _ in range(50):
                    if manager.owned:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(manager.owned)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(process.terminations, 1)
                self.assertFalse(manager.owned)

    async def test_malformed_api_does_not_trigger_start_over_reachable_server(self):
        manager = OllamaManager(transport=httpx.MockTransport(lambda r: httpx.Response(503, text='private-response')))
        with patch('ollama.asyncio.create_subprocess_exec', new_callable=AsyncMock) as factory:
            state = await manager.start(Settings())
            self.assertTrue(state['reachable'])
            self.assertFalse(state['available'])
            self.assertNotIn('private-response', state['error'])
            factory.assert_not_awaited()


class OllamaTuiTests(unittest.IsolatedAsyncioTestCase):
    async def settled(self, app, pilot):
        for _ in range(50):
            await pilot.pause(.02)
            if app.ollama_task is None:
                return
        self.fail('Ollama action did not finish')

    async def test_tab_has_manual_start_model_selection_and_owned_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'installation'
            (root / 'bin').mkdir(parents=True)
            (root / 'bin/ollama').write_text('fake')
            (root / 'models').mkdir()
            server = FakeServer()
            process = FakeProcess(server)
            async def spawn(*args, **kwargs):
                server.available = True
                return process
            manager = OllamaManager(installation=root, transport=httpx.MockTransport(server.request))
            directory = Path(tmp) / 'sessions'
            app = RagApp(directory=directory, ollama=manager, runner=Runner(FakeDocs(),FakeModel(),FakeModel()))
            with patch('ollama.asyncio.create_subprocess_exec', side_effect=spawn) as factory:
                async with app.run_test(size=(180,60)) as pilot:
                    await self.settled(app, pilot)
                    factory.assert_not_awaited()
                    app.query_one('#tabs', TabbedContent).active = 'ollama-tab'
                    await pilot.pause()
                    self.assertFalse(app.query_one('#ollama-start', Button).disabled)
                    self.assertTrue(app.query_one('#ollama-stop', Button).disabled)
                    await pilot.click('#ollama-start')
                    await self.settled(app, pilot)
                    factory.assert_awaited_once()
                    self.assertTrue(manager.owned)
                    self.assertFalse(app.query_one('#ollama-stop', Button).disabled)
                    app.query_one('#ollama-model', Select).value = 'qwen3:8b'
                    await pilot.pause()
                    self.assertEqual(app.query_one('#local_model', Input).value, 'qwen3:8b')
                    self.assertEqual(app.session['settings']['local_model'], 'qwen3:8b')
                    saved = load_session(next(directory.glob('*.json')))
                    self.assertEqual(saved['settings']['local_model'], 'qwen3:8b')
                    await pilot.resize_terminal(120,40)
                    await pilot.pause()
                    for name in ('ollama-start','ollama-stop','ollama-refresh'):
                        self.assertLessEqual(app.query_one('#'+name).region.right, 120)
                    app.query_one('#question', TextArea).load_text('ArrayOptFirstElem')
                    app.action_send()
                    await app.generation_task
                    await pilot.pause()
                    self.assertEqual(app.session['turns'][0]['results']['local']['model'], 'qwen3:8b')
                self.assertFalse(manager.owned)
                self.assertEqual(process.terminations, 1)

    async def test_external_server_stop_disabled_refresh_and_setting_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = FakeServer(True)
            manager = OllamaManager(transport=httpx.MockTransport(server.request))
            app = RagApp(directory=tmp, ollama=manager, runner=Runner(FakeDocs(),FakeModel(),FakeModel()))
            with patch('ollama.asyncio.create_subprocess_exec', new_callable=AsyncMock) as factory:
                async with app.run_test(size=(120,40)) as pilot:
                    await self.settled(app, pilot)
                    self.assertTrue(app.query_one('#ollama-stop', Button).disabled)
                    self.assertTrue(app.query_one('#ollama-start', Button).disabled)
                    app.query_one('#local_model', Input).value = 'qwen3:8b'
                    await pilot.pause()
                    self.assertEqual(app.query_one('#ollama-model', Select).value, 'qwen3:8b')
                    app.query_one('#tabs', TabbedContent).active = 'ollama-tab'
                    await pilot.pause()
                    await pilot.click('#ollama-refresh')
                    await self.settled(app, pilot)
                    self.assertIn('GPU: 4.00 ГиБ', str(app.query_one('#ollama-loaded', Static).render()))
                self.assertTrue(server.available)
                factory.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
