"""Voice configuration and HTTP client; no inference dependencies in the bot."""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
import math
import os

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = 'gigaam-v3-e2e-rnnt'


class VoiceError(Exception):
    """Safe user-facing error. Never wrap request URLs or server responses."""


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = False
    url: str = 'http://127.0.0.1:8765'
    token: str = ''
    model_dir: Path = ROOT / 'data/voice-models'
    host: str = '127.0.0.1'
    port: int = 8765
    max_bytes: int = 10 * 1024 * 1024
    max_seconds: int = 300
    timeout: float = 90
    threads: int = 8

    @classmethod
    def load(cls):
        try:
            enabled = os.getenv('ASSISTANT_VOICE_ENABLED', 'false').lower().strip()
            if enabled not in ('true', 'false'):
                raise ValueError
            path = Path(os.getenv('VOICE_MODEL_DIR', 'data/voice-models'))
            config = cls(
                enabled=enabled == 'true',
                url=os.getenv('VOICE_SERVICE_URL', cls.url).rstrip('/'),
                token=os.getenv('VOICE_SERVICE_TOKEN', '').strip(),
                model_dir=path if path.is_absolute() else ROOT / path,
                host=os.getenv('VOICE_SERVICE_HOST', cls.host),
                port=int(os.getenv('VOICE_SERVICE_PORT', '8765')),
                max_bytes=int(os.getenv('VOICE_MAX_BYTES', str(cls.max_bytes))),
                max_seconds=int(os.getenv('VOICE_MAX_SECONDS', '300')),
                timeout=float(os.getenv('VOICE_TIMEOUT', '90')),
                threads=int(os.getenv('VOICE_CPU_THREADS', '8')))
            parsed = urlsplit(config.url)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path or parsed.port == 0):
                raise ValueError
            if (not 1 <= config.port <= 65535 or not 1 <= config.max_bytes <= 20 * 1024 * 1024
                    or not 1 <= config.max_seconds <= 1800 or not 1 <= config.threads <= 64
                    or not math.isfinite(config.timeout) or config.timeout <= 0):
                raise ValueError
            local = ('127.0.0.1', 'localhost', '::1')
            if ((config.host not in local or parsed.hostname not in local) and not config.token):
                raise ValueError
            return config
        except ValueError:
            raise ValueError('Неверные настройки VOICE_* / ASSISTANT_VOICE_ENABLED. '
                             'Вне localhost требуется VOICE_SERVICE_TOKEN.') from None


class VoiceClient:
    def __init__(self, client, config):
        self.client, self.config = client, config

    async def transcribe(self, audio):
        if not audio or len(audio) > self.config.max_bytes:
            raise VoiceError('Голосовое пустое или превышает допустимый размер.')
        headers = {'Content-Type': 'application/octet-stream'}
        if self.config.token:
            headers['Authorization'] = 'Bearer ' + self.config.token
        try:
            response = await self.client.post(self.config.url + '/transcribe', content=audio,
                headers=headers, timeout=self.config.timeout)
            if response.status_code != 200:
                messages = {400: 'Не удалось прочитать аудио.', 413: 'Голосовое превышает лимит размера или длительности.',
                            422: 'Не удалось распознать речь. Пришли текст или другую запись.',
                            503: 'Сервис распознавания занят. Попробуй позже.'}
                raise VoiceError(messages.get(response.status_code, 'Сервис распознавания недоступен. Попробуй позже.'))
            result = response.json()
            text = result.get('text') if isinstance(result, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) > 100000:
                raise VoiceError('Сервис распознавания вернул некорректный текст.')
            return text.strip()
        except VoiceError:
            raise
        except Exception:
            raise VoiceError('Не удалось связаться с сервисом распознавания. Попробуй позже.') from None
