"""Start and supervise the local voice service and bot together."""
import asyncio
import os
import signal
import sys
import httpx
from .config import Config


async def launch(config):
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    previous = signal.getsignal(signal.SIGTERM)
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await _launch(config)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        signal.signal(signal.SIGTERM, previous)


async def _launch(config):
    processes = []
    try:
        if config.voice.enabled:
            if config.voice.url != f'http://{config.voice.host}:{config.voice.port}' or config.voice.host not in ('127.0.0.1', 'localhost'):
                raise ValueError('Совместный запуск требует локальные согласованные VOICE_SERVICE_URL/HOST/PORT. '
                                 'Для внешнего сервиса запускайте python -m personal_assistent.')
            service = await asyncio.create_subprocess_exec(sys.executable, '-m', 'personal_assistent.voice_service')
            processes.append(service)
            headers = {'Authorization': 'Bearer ' + config.voice.token} if config.voice.token else {}
            async with httpx.AsyncClient(trust_env=False) as client:
                async with asyncio.timeout(60):
                    while True:
                        if service.returncode is not None:
                            raise ValueError('Сервис голоса завершился при запуске.')
                        try:
                            response = await client.get(config.voice.url + '/health', headers=headers, timeout=1)
                            if (response.status_code == 200 and response.json().get('ready')
                                    and response.json().get('pid') == service.pid):
                                break
                        except (httpx.HTTPError, ValueError):
                            pass
                        await asyncio.sleep(.25)
        bot = await asyncio.create_subprocess_exec(sys.executable, '-m', 'personal_assistent')
        processes.append(bot)
        waits = [asyncio.create_task(process.wait()) for process in processes]
        try:
            done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            return next(iter(done)).result() or (1 if len(processes) > 1 and bot.returncode is None else 0)
        finally:
            for task in waits:
                task.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
    finally:
        for process in reversed(processes):
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()


def main():
    os.umask(0o077)
    try:
        return asyncio.run(launch(Config.load()))
    except KeyboardInterrupt:
        return 0
    except asyncio.CancelledError:
        return 128 + signal.SIGTERM
    except Exception as exc:
        print(str(exc) if isinstance(exc, ValueError) else 'Совместный запуск не выполнен: ' + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
