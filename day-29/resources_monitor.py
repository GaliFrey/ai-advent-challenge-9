"""Observed process RSS and device memory peaks; missing measurements stay null."""
import asyncio
import time
from pathlib import Path
import httpx


def ollama_rss(root=Path('/proc')):
    total, pids = 0, []
    for folder in root.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            # Inspect executable identity, never environment or command-line secrets.
            name = (folder / 'comm').read_text().strip()
            if name not in ('ollama', 'llama-server') and not name.startswith('ollama_llama'):
                continue
            status = (folder / 'status').read_text()
            line = next(line for line in status.splitlines() if line.startswith('VmRSS:'))
            total += int(line.split()[1]) * 1024
            pids.append(int(folder.name))
        except (OSError, ValueError, StopIteration):
            continue
    return (total if pids else None), pids


async def gpu_memory():
    try:
        process = await asyncio.create_subprocess_exec(
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 2)
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None
        if process.returncode:
            return None
        values = []
        for line in output.decode().splitlines():
            index, name, used, total = [v.strip() for v in line.split(',')]
            values.append({'index': index, 'name': name, 'used_bytes': int(used) * 1024**2,
                           'total_bytes': int(total) * 1024**2})
        return values or None
    except (OSError, ValueError):
        return None


class ResourceMonitor:
    def __init__(self, sampler=None):
        self.sampler = sampler or self.sample
        self.samples = []
        self.task = None
        self.started = None

    async def sample(self):
        ram, pids = ollama_rss()
        devices = await gpu_memory()
        running, error = None, None
        try:
            async with httpx.AsyncClient(base_url='http://127.0.0.1:11434', trust_env=False, timeout=1) as client:
                response = await client.get('/api/ps')
                response.raise_for_status()
                running = response.json()['models']
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            error = 'Состояние /api/ps недоступно.'
        return {'ram_bytes': ram, 'pids': pids, 'devices': devices,
                'gpu_bytes': sum(d['used_bytes'] for d in devices) if devices else None,
                'running_models': running, 'ps_error': error}

    async def collect(self):
        try:
            sample = await self.sampler()
        except Exception:
            sample = {'ram_bytes': None, 'gpu_bytes': None, 'sample_error': 'Измерение ресурсов недоступно.'}
        sample['elapsed_seconds'] = time.perf_counter() - self.started
        self.samples.append(sample)

    async def loop(self):
        while True:
            start = time.perf_counter()
            await self.collect()
            await asyncio.sleep(max(0, .5 - (time.perf_counter() - start)))

    async def __aenter__(self):
        self.started = time.perf_counter()
        await self.collect()
        self.task = asyncio.create_task(self.loop())
        return self

    async def __aexit__(self, *args):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        await self.collect()

    def result(self):
        def peak(key):
            values = [s[key] for s in self.samples if s.get(key) is not None]
            return max(values) if values else None
        return {'max_ram_bytes': peak('ram_bytes'), 'max_gpu_bytes': peak('gpu_bytes'),
                'samples': self.samples, 'interval_seconds': .5,
                'limitations': ['RAM — сумма RSS процессов Ollama, общие страницы могут учитываться повторно.',
                                'GPU — вся память устройств, включая другие приложения.',
                                'Пик — максимум наблюдений, короткие всплески между отсчётами могут быть пропущены.']}
