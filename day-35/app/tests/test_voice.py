import asyncio
from dataclasses import replace
import importlib.util
import json
import os
import signal
from pathlib import Path
import tempfile
import threading
import wave
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock

import httpx
from personal_assistent.runtime import Runtime
from personal_assistent.storage import Store
from personal_assistent.telegram import Telegram
from personal_assistent.voice import VoiceConfig, VoiceClient, VoiceError
from personal_assistent.voice_models import file_hash, verify_models
from test_assistant import NOW, OWNER, update


def voice_update(number=1):
    incoming = update(number=number)
    incoming['message'].pop('text')
    incoming['message']['voice'] = {'file_id': 'voice-file', 'duration': 12, 'file_size': 100}
    return incoming


class VoiceConfigTests(unittest.TestCase):
    def test_defaults_do_not_enable_voice(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(VoiceConfig.load().enabled)

    def test_invalid_settings(self):
        for key, value in [('VOICE_SERVICE_URL', 'http://user:secret@localhost'),
                           ('VOICE_SERVICE_URL', 'http://localhost/path'),
                           ('VOICE_SERVICE_URL', 'http://localhost:bad'),
                           ('VOICE_SERVICE_URL', 'http://other:8765'),
                           ('VOICE_SERVICE_HOST', '0.0.0.0'),
                           ('VOICE_TIMEOUT', 'nan'), ('VOICE_MAX_BYTES', '0'),
                           ('VOICE_CPU_THREADS', '0'), ('ASSISTANT_VOICE_ENABLED', 'yes')]:
            with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}, clear=True):
                with self.assertRaises(ValueError):
                    VoiceConfig.load()

    def test_remote_service_with_token(self):
        with patch.dict(os.environ, {'VOICE_SERVICE_URL': 'https://voice.example',
                                    'VOICE_SERVICE_TOKEN': 'secret'}, clear=True):
            self.assertEqual(VoiceConfig.load().url, 'https://voice.example')


class VoiceClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_bytes_with_token_and_returns_text(self):
        def handler(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:8765/transcribe')
            self.assertEqual(request.content, b'audio')
            self.assertEqual(request.headers['Authorization'], 'Bearer secret')
            return httpx.Response(200, json={'text': ' Напомни завтра ', 'model': 'test'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(await VoiceClient(client, replace(VoiceConfig(), token='secret')).transcribe(b'audio'),
                             'Напомни завтра')

    async def test_errors_never_expose_server_response_or_url(self):
        for status, payload in [(503, {'error': 'secret'}), (200, {'text': ''}),
                                (200, []), (500, {'text': 'secret'})]:
            with self.subTest(status=status, payload=payload):
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda _: httpx.Response(status, json=payload))) as client:
                    with self.assertRaises(VoiceError) as exc:
                        await VoiceClient(client, VoiceConfig()).transcribe(b'audio')
                    self.assertNotIn('secret', str(exc.exception))

    async def test_telegram_download_checks_actual_size_and_preserves_path(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.method == 'POST':
                return httpx.Response(200, json={'ok': True, 'result': {'file_path': 'voice/file.oga'}})
            return httpx.Response(200, content=b'12345')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            telegram = Telegram(client, 'fake-token')
            self.assertEqual(await telegram.download_voice({'file_id': 'abc'}, 5, 60), b'12345')
            self.assertTrue(str(requests[-1].url).endswith('/voice/file.oga'))
            with self.assertRaises(VoiceError):
                await telegram.download_voice({'file_id': 'abc'}, 4, 60)
            count = len(requests)
            with self.assertRaises(VoiceError):
                await telegram.download_voice({'file_id': 'abc', 'duration': 61}, 5, 60)
            self.assertEqual(len(requests), count)


class InputTelegram:
    def __init__(self):
        self.downloads = 0

    async def download_voice(self, voice, max_bytes, max_seconds):
        self.downloads += 1
        return b'audio'


class Transcriber:
    config = VoiceConfig()

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.calls = 0
        self.error = None

    async def transcribe(self, audio):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if self.error:
            raise self.error
        return 'Напомни через час проверить духовку'


class CaptureAgent:
    def __init__(self, store):
        self.store = store
        self.inputs = []
        self.interrupt = False

    async def process(self, job):
        message = json.loads(job['payload'])['message']
        self.inputs.append(message)
        if self.interrupt:
            raise asyncio.CancelledError
        self.store.finish(job['id'], message['text'], 'Готово', NOW)


class VoiceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store(':memory:')
        self.telegram = InputTelegram()
        self.voice = Transcriber()
        self.agent = CaptureAgent(self.store)
        self.runtime = Runtime(self.store, self.telegram, self.agent, OWNER, lambda: NOW, voice=self.voice)

    async def asyncTearDown(self):
        self.store.db.close()

    def status(self, number=1):
        return self.store.db.execute('SELECT status FROM jobs WHERE id=?', (number,)).fetchone()[0]

    async def test_voice_keeps_source_metadata_and_goes_through_agent(self):
        self.store.enqueue(voice_update(), NOW + 120)
        await self.runtime.process_once()
        message = self.agent.inputs[0]
        self.assertEqual(message['date'], int(NOW))
        self.assertEqual(message['message_id'], 1)
        self.assertEqual(message['voice']['file_id'], 'voice-file')
        self.assertEqual(message['text'], 'Напомни через час проверить духовку')
        self.assertEqual(self.store.history()[0]['content'], message['text'])
        self.assertEqual(self.status(), 'done')

    async def test_unauthorized_voice_is_not_downloaded(self):
        incoming = voice_update()
        incoming['message']['from']['id'] = OWNER + 1
        self.store.enqueue(incoming, NOW)
        await self.runtime.process_once()
        self.assertEqual(self.telegram.downloads, 0)
        self.assertEqual(self.voice.calls, 0)
        self.assertEqual(self.status(), 'ignored')

    async def test_unsupported_attachment_describes_enabled_input_types(self):
        incoming = voice_update()
        incoming['message'].pop('voice')
        incoming['message']['photo'] = [{'file_id': 'photo'}]
        self.store.enqueue(incoming, NOW)
        await self.runtime.process_once()
        self.assertIn('текстовые и голосовые', self.store.history()[-1]['content'])
        self.assertEqual(self.agent.inputs, [])
        self.assertEqual(self.telegram.downloads, 0)

    async def test_voice_failure_and_disabled_voice_do_not_start_agent(self):
        self.voice.error = VoiceError('Сервис распознавания недоступен.')
        self.store.enqueue(voice_update(), NOW)
        await self.runtime.process_once()
        self.runtime.voice = None
        self.store.enqueue(voice_update(2), NOW)
        await self.runtime.process_once()
        self.assertEqual(self.agent.inputs, [])
        self.assertEqual(self.store.history(), [])
        self.assertEqual([self.status(1), self.status(2)], ['done', 'done'])

    async def test_restart_uses_saved_transcription_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.sqlite3'
            store = Store(path)
            try:
                agent = CaptureAgent(store)
                agent.interrupt = True
                runtime = Runtime(store, self.telegram, agent, OWNER, lambda: NOW, voice=self.voice)
                store.enqueue(voice_update(), NOW)
                with self.assertRaises(asyncio.CancelledError):
                    await runtime.process_once()
            finally:
                store.db.close()
            store = Store(path)
            try:
                agent = CaptureAgent(store)
                await Runtime(store, self.telegram, agent, OWNER, lambda: NOW, voice=self.voice).process_once()
                self.assertEqual(len(agent.inputs), 1)
                self.assertEqual(self.telegram.downloads, 1)
                self.assertEqual(self.voice.calls, 1)
            finally:
                store.db.close()

    async def test_text_does_not_overtake_voice_and_controls_still_work(self):
        self.voice.release.clear()
        self.store.enqueue(voice_update(), NOW)
        task = asyncio.create_task(self.runtime.process_once())
        try:
            await asyncio.wait_for(self.voice.started.wait(), 1)
            self.store.enqueue(update('Следующее сообщение', 2), NOW)
            self.store.enqueue(update('/profile', 3), NOW)
            async with asyncio.timeout(1):
                while self.status(3) == 'pending':
                    await asyncio.sleep(.01)
            self.assertEqual(self.status(2), 'pending')
            self.assertEqual(self.agent.inputs, [])
            self.voice.release.set()
            await asyncio.wait_for(task, 1)
            await self.runtime.process_once()
            self.assertEqual([m['message_id'] for m in self.agent.inputs], [1, 2])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancel_during_transcription_never_runs_agent(self):
        self.voice.release.clear()
        self.store.enqueue(voice_update(), NOW)
        task = asyncio.create_task(self.runtime.process_once())
        try:
            await asyncio.wait_for(self.voice.started.wait(), 1)
            self.store.enqueue(update('/cancel', 2), NOW)
            await asyncio.wait_for(task, 1)
            self.assertEqual(self.agent.inputs, [])
            self.assertEqual([self.status(1), self.status(2)], ['done', 'done'])
            self.assertNotIn('text', json.loads(self.store.db.execute('SELECT payload FROM jobs WHERE id=1').fetchone()[0])['message'])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_timeout_covers_transcription(self):
        self.voice.release.clear()
        self.runtime.processing_timeout = .02
        self.store.enqueue(voice_update(), NOW)
        await asyncio.wait_for(self.runtime.process_once(), 1)
        self.assertEqual(self.status(), 'done')
        self.assertEqual(self.agent.inputs, [])

    async def test_cancel_at_transcription_completion_prevents_agent_start(self):
        async def transcribe(audio):
            self.store.enqueue(update('/cancel', 2), NOW)
            return 'Сохрани факт'
        self.voice.transcribe = transcribe
        self.store.enqueue(voice_update(), NOW)
        await self.runtime.process_once()
        self.assertEqual(self.agent.inputs, [])
        self.assertEqual([self.status(1), self.status(2)], ['done', 'done'])
        self.assertEqual(self.store.history()[0]['content'], 'Сохрани факт')

    async def test_saved_slash_transcription_is_not_a_control_command(self):
        incoming = voice_update()
        incoming['message']['text'] = '/new confirm'
        self.store.enqueue(incoming, NOW)
        await self.runtime.process_once()
        self.assertEqual(self.agent.inputs[0]['text'], '/new confirm')
        self.assertEqual(self.store.get_meta('session', 0), 0)


class ModelPreparationTests(unittest.TestCase):
    def test_missing_and_corrupted_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'test'
            model.mkdir()
            file = model / 'encoder.onnx'
            file.write_bytes(b'correct')
            spec = {'test': {'files': {'encoder.onnx': file_hash(file)}}}
            with patch('personal_assistent.voice_models.MODELS', spec):
                verify_models(root)
                file.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    verify_models(root)
                file.unlink()
                with self.assertRaises(ValueError):
                    verify_models(root)

    @unittest.skipUnless(importlib.util.find_spec('huggingface_hub'), 'optional voice dependencies not installed')
    def test_preparation_downloads_pinned_files_and_skips_verified_ones(self):
        from personal_assistent.setup_voice import prepare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_file = root / 'expected'
            expected_file.write_bytes(b'correct')
            spec = {'test': {'repo': 'publisher/model', 'revision': 'pinned-revision',
                             'files': {'encoder.onnx': file_hash(expected_file)}}}
            def download(repo, filename, *, revision, local_dir, force_download):
                self.assertEqual((repo, revision), ('publisher/model', 'pinned-revision'))
                (local_dir / filename).write_bytes(b'correct')
            with patch('personal_assistent.setup_voice.MODELS', spec), \
                    patch('personal_assistent.voice_models.MODELS', spec), \
                    patch('personal_assistent.setup_voice.shutil.which', return_value='/usr/bin/ffmpeg'), \
                    patch('huggingface_hub.hf_hub_download', side_effect=download) as fetch:
                config = replace(VoiceConfig(), model_dir=root)
                prepare(config)
                self.assertEqual(fetch.call_count, 1)
                prepare(config)
                self.assertEqual(fetch.call_count, 1)
                (root / 'test/encoder.onnx').write_bytes(b'broken')
                prepare(config)
                self.assertTrue(fetch.call_args.kwargs['force_download'])


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional voice dependencies not installed')
class DecodeTests(unittest.TestCase):
    def test_corrupt_audio_is_rejected(self):
        from personal_assistent.voice_service import decode_audio, AudioError
        with self.assertRaises(AudioError) as exc:
            decode_audio(b'not an audio file', VoiceConfig())
        self.assertEqual(exc.exception.code, 400)

    def test_real_decoded_duration_is_checked_without_silent_truncation(self):
        from personal_assistent.voice_service import decode_audio, AudioError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\0\0' * 20000)
            with self.assertRaises(AudioError) as exc:
                decode_audio(path.read_bytes(), replace(VoiceConfig(), max_seconds=1))
            self.assertEqual(exc.exception.code, 413)


class LauncherTests(unittest.IsolatedAsyncioTestCase):
    async def test_sigterm_stops_child_process_and_restores_handler(self):
        from personal_assistent.launch import launch
        spawned = asyncio.Event()
        callbacks = {}
        class Process:
            returncode = None
            def __init__(self):
                self.exited = asyncio.Event()
            async def wait(self):
                await self.exited.wait()
                return self.returncode
            def terminate(self):
                self.returncode = -15
                self.exited.set()
        process = Process()
        async def spawn(*args):
            spawned.set()
            return process
        loop = asyncio.get_running_loop()
        with patch('personal_assistent.launch.asyncio.create_subprocess_exec', side_effect=spawn), \
                patch.object(loop, 'add_signal_handler', side_effect=lambda sig, callback: callbacks.update({sig: callback})), \
                patch.object(loop, 'remove_signal_handler') as remove:
            task = asyncio.create_task(launch(SimpleNamespace(voice=VoiceConfig())))
            try:
                await asyncio.wait_for(spawned.wait(), 1)
                self.assertIn(signal.SIGTERM, callbacks)
                callbacks[signal.SIGTERM]()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(process.returncode, -15)
                remove.assert_called_once_with(signal.SIGTERM)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_bot_exit_stops_local_service(self):
        from personal_assistent.launch import launch
        class Process:
            def __init__(self, pid, code=None):
                self.pid, self.returncode = pid, code
                self.done = asyncio.Event()
                self.terminated = False
                if code is not None:
                    self.done.set()
            async def wait(self):
                await self.done.wait()
                return self.returncode
            def terminate(self):
                self.terminated = True
                self.returncode = -15
                self.done.set()
        service, bot = Process(123), Process(124, 0)
        def health(request):
            return httpx.Response(200, json={'ready': True, 'pid': 123})
        client = httpx.AsyncClient(transport=httpx.MockTransport(health))
        with patch('personal_assistent.launch.asyncio.create_subprocess_exec', AsyncMock(side_effect=[service, bot])), \
                patch('personal_assistent.launch.httpx.AsyncClient', return_value=client):
            self.assertEqual(await launch(SimpleNamespace(voice=replace(VoiceConfig(), enabled=True))), 0)
        self.assertTrue(service.terminated)

    async def test_external_service_is_not_spawned_by_local_launcher(self):
        from personal_assistent.launch import launch
        with patch('personal_assistent.launch.asyncio.create_subprocess_exec', AsyncMock()) as spawn:
            with self.assertRaises(ValueError):
                await launch(SimpleNamespace(voice=replace(VoiceConfig(), enabled=True, url='https://remote')))
            spawn.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('aiohttp'), 'optional voice dependencies not installed')
class VoiceServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient, TestServer
        from personal_assistent.voice_service import create_app
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        owner = self
        class Recognizer:
            def transcribe(self, audio):
                owner.started.set()
                owner.release.wait(3)
                return {'text': 'Тест', 'model': 'test', 'duration': 1}
        self.config = replace(VoiceConfig(), token='secret', max_bytes=8)
        self.client = TestClient(TestServer(create_app(self.config, Recognizer())))
        self.addAsyncCleanup(self.client.close)
        await self.client.start_server()
        self.headers = {'Authorization': 'Bearer secret'}

    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()

    async def test_http_auth_limits_empty_and_success(self):
        response = await self.client.post('/transcribe', data=b'audio')
        self.assertEqual(response.status, 401)
        for data, status in [(b'', 400), (b'123456789', 413), (b'1234567890', 413), (b'audio', 200)]:
            response = await self.client.post('/transcribe', data=data, headers=self.headers)
            self.assertEqual(response.status, status)
        response = await self.client.get('/health', headers=self.headers)
        self.assertTrue((await response.json())['ready'])

    async def test_busy_does_not_start_second_inference(self):
        self.release.clear()
        task = asyncio.create_task(self.client.post('/transcribe', data=b'audio', headers=self.headers))
        try:
            async with asyncio.timeout(1):
                while not self.started.is_set():
                    await asyncio.sleep(.01)
            response = await self.client.post('/transcribe', data=b'audio', headers=self.headers)
            self.assertEqual(response.status, 503)
            self.release.set()
            response = await task
            self.assertEqual(response.status, 200)
        finally:
            self.release.set()
            await asyncio.gather(task, return_exceptions=True)
