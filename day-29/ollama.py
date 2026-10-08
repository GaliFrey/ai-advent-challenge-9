"""Read-only Ollama status and explicit ownership of a locally started server."""
import asyncio
import os
import time
from pathlib import Path

import httpx

from core import DAY


class OllamaManager:
    def __init__(self, *, transport=None, installation=None):
        self.transport = transport
        self.installation = Path(installation or DAY.parent / 'day-28/resources/ollama-local')
        self.process = None

    @property
    def owned(self):
        return self.process is not None and self.process.returncode is None

    async def snapshot(self):
        state = {'available': False, 'reachable': False, 'owned': self.owned,
                 'models': [], 'running': [], 'error': ''}
        try:
            async with httpx.AsyncClient(base_url='http://127.0.0.1:11434', trust_env=False,
                    timeout=httpx.Timeout(2), transport=self.transport) as client:
                response = await client.get('/api/tags')
                state['reachable'] = True
                response.raise_for_status()
                models = response.json()['models']
                if not isinstance(models, list):
                    raise ValueError
                state['models'] = [m for m in models if isinstance(m, dict) and isinstance(m.get('name'), str)
                    and not m.get('remote_model') and not m.get('remote_host')
                    and not m['name'].endswith(('-cloud', ':cloud'))]
                state['available'] = True
                try:
                    running = await client.get('/api/ps')
                    running.raise_for_status()
                    state['running'] = running.json()['models']
                    if not isinstance(state['running'], list):
                        raise ValueError
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    state['error'] = 'Сервер доступен; состояние моделей в памяти получить не удалось.'
        except httpx.ConnectError:
            if self.owned:
                state['error'] = 'Процесс TUI запущен, но HTTP ещё недоступен.'
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            state['error'] = 'Не удалось прочитать состояние Ollama на 127.0.0.1:11434.'
        return state

    async def start(self, settings):
        settings.validate()
        current = await self.snapshot()
        if current['reachable']:
            return current  # Never spawn over an external server.
        if self.owned:
            raise ValueError('Процесс уже запущен; дождитесь готовности или остановите его.')
        binary = self.installation / 'bin/ollama'
        if not binary.is_file() or not (self.installation / 'models').is_dir():
            raise ValueError('Нет установки или весов в day-28/resources/ollama-local/.')
        # Exclude cloud keys and proxy settings from the subprocess environment/logs.
        env = {k: v for k, v in os.environ.items() if k in (
            'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH',
            'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
            'GGML_VK_VISIBLE_DEVICES', 'OLLAMA_VULKAN', 'OLLAMA_LLM_LIBRARY')}
        env.update(OLLAMA_MODELS=str((self.installation / 'models').resolve()),
                   OLLAMA_HOST='127.0.0.1:11434', OLLAMA_CONTEXT_LENGTH=str(settings.num_ctx),
                   OLLAMA_NUM_PARALLEL='1', OLLAMA_KEEP_ALIVE='30m', OLLAMA_NO_CLOUD='1')
        try:
            self.process = await asyncio.create_subprocess_exec(str(binary), 'serve',
                cwd=self.installation, env=env, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
        except OSError:
            raise ValueError('Не удалось запустить локальный бинарный файл Ollama.') from None
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    raise ValueError(f'Процесс Ollama завершился: код {self.process.returncode}. Проверьте порт и установку.')
                state = await self.snapshot()
                if state['available'] and self.owned:
                    return state
                await asyncio.sleep(.2)
            raise ValueError('Ollama не стала доступна за 20 секунд; запущенный процесс остановлен.')
        except BaseException:
            await self.stop()
            raise

    async def stop(self):
        process = self.process
        if process is None:
            return  # No process handle means no authority to stop an external server.
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        self.process = None
