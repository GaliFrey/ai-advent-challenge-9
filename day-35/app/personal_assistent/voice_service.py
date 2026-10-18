"""Local, offline GigaAM HTTP service. Install the optional voice dependencies."""
import asyncio
import hmac
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from .voice import ROOT, MODEL_NAME, VoiceConfig
from .voice_models import verify_models

log = logging.getLogger(__name__)


class AudioError(Exception):
    def __init__(self, code):
        self.code = code


def decode_audio(audio, config):
    # Bound both compressed input and decoded PCM; never silently truncate.
    if not audio or len(audio) > config.max_bytes:
        raise AudioError(413 if audio else 400)
    with tempfile.TemporaryDirectory(prefix='assistant-voice-') as directory:
        path = Path(directory) / 'input.audio'
        path.write_bytes(audio)
        try:
            completed = subprocess.run([
                'ffmpeg', '-v', 'error', '-nostdin', '-protocol_whitelist', 'file,pipe',
                '-i', str(path), '-map', '0:a:0', '-t', str(config.max_seconds + 1),
                '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20, check=True)
        except (subprocess.SubprocessError, OSError):
            raise AudioError(400) from None
    import numpy as np
    samples = np.frombuffer(completed.stdout, dtype='<f4')
    if len(samples) > config.max_seconds * 16000:
        raise AudioError(413)
    if not len(samples) or not np.isfinite(samples).all():
        raise AudioError(400)
    return samples


class Recognizer:
    def __init__(self, config):
        if not shutil.which('ffmpeg'):
            raise ValueError('Нужен системный ffmpeg.')
        verify_models(config.model_dir)
        import onnx_asr
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        kwargs = {'providers': ['CPUExecutionProvider'], 'sess_options': options}
        model = onnx_asr.load_model(MODEL_NAME, config.model_dir / 'gigaam', **kwargs)
        vad = onnx_asr.load_vad('silero', config.model_dir / 'silero', **kwargs)
        self.model = model.with_vad(vad, batch_size=1, max_speech_duration_s=20,
                                   min_silence_duration_ms=350, speech_pad_ms=100)
        self.config = config

    def transcribe(self, audio):
        samples = decode_audio(audio, self.config)
        segments = list(self.model.recognize(samples, sample_rate=16000))
        text = ' '.join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise AudioError(422)
        return {'text': text, 'model': MODEL_NAME, 'duration': len(samples) / 16000}


def create_app(config, recognizer):
    from aiohttp import web
    work = None
    accepting = False

    @web.middleware
    async def authenticate(request, handler):
        if config.token and not hmac.compare_digest(
                request.headers.get('Authorization', '').encode(), ('Bearer ' + config.token).encode()):
            return web.json_response({'error': 'unauthorized'}, status=401)
        return await handler(request)

    async def health(request):
        return web.json_response({'ready': True, 'model': MODEL_NAME,
                                  'pid': os.getpid(),
                                  'busy': accepting or (work is not None and not work.done())})

    async def transcribe(request):
        nonlocal work, accepting
        if accepting or (work is not None and not work.done()):
            return web.json_response({'error': 'busy'}, status=503)
        accepting = True
        try:
            # A streaming client cannot occupy the only slot indefinitely.
            async with asyncio.timeout(20):
                audio = await request.read()
            if len(audio) > config.max_bytes:
                return web.json_response({'error': 'too_large'}, status=413)
            if not audio:
                return web.json_response({'error': 'empty_audio'}, status=400)
            work = asyncio.create_task(asyncio.to_thread(recognizer.transcribe, audio))
            # Retrieve exceptions even if the HTTP client has disconnected.
            work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            result = await asyncio.shield(work)
            return web.json_response(result)
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({'error': 'too_large'}, status=413)
        except AudioError as exc:
            return web.json_response({'error': 'invalid_audio'}, status=exc.code)
        except TimeoutError:
            return web.json_response({'error': 'upload_timeout'}, status=408)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning('Распознавание не завершено: %s', type(exc).__name__)
            return web.json_response({'error': 'recognition_failed'}, status=500)
        finally:
            accepting = False

    async def cleanup(app):
        if work is not None:
            await asyncio.gather(work, return_exceptions=True)

    app = web.Application(client_max_size=config.max_bytes + 1, middlewares=[authenticate])
    app.router.add_get('/health', health)
    app.router.add_post('/transcribe', transcribe)
    app.on_cleanup.append(cleanup)
    return app


def main():
    os.umask(0o077)
    load_dotenv(ROOT / '.env')
    logging.basicConfig(level=logging.WARNING)
    try:
        config = VoiceConfig.load()
        from aiohttp import web
        recognizer = Recognizer(config)
        # No access log: do not retain audio metadata, tokens or request bodies.
        web.run_app(create_app(config, recognizer), host=config.host, port=config.port,
                    access_log=None, handler_cancellation=True)
    except ImportError:
        print('Установите зависимости: uv sync --locked --extra voice', file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print('Не удалось запустить сервис голоса: ' + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
