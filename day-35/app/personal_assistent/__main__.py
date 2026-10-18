import asyncio
import fcntl
import logging
import os
import sys
import httpx
from .config import Config
from .storage import Store
from .migrate import requires_migration
from .telegram import Telegram
from .agent import Agent, DeepSeek
from .runtime import Runtime
from .context import Budget
from .voice import VoiceClient


async def run(config):
    db_path=config.db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(str(db_path) + '.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Уже запущен экземпляр с этой базой') from None
        if requires_migration(db_path):
            raise ValueError('Нужна миграция остановленной базы с резервной копией: python -m personal_assistent.migrate --database PATH --backup NEW_PATH')
        store = Store(db_path)
        os.chmod(db_path, 0o600)
        async with httpx.AsyncClient() as telegram_http, httpx.AsyncClient() as model_http, httpx.AsyncClient(trust_env=False) as voice_http:
            telegram = Telegram(telegram_http, config.telegram_token)
            # Check configuration before starting tasks. Never drop pending updates.
            webhook = await telegram.call('getWebhookInfo')
            if webhook.get('url'):
                raise ValueError('У бота включён webhook. Отключите его перед запуском long polling.')
            await telegram.call('getMe')
            logging.getLogger(__name__).info('Бот запущен; остановка: Ctrl+C')
            try:
                model = DeepSeek(model_http, config.deepseek_key,
                    temperature=config.temperature, max_tokens=config.max_tokens,
                    on_call=store.record_llm_call)
                agent = Agent(store, model, budget=Budget(config.context_budget, config.context_reserve))
                await Runtime(store, telegram, agent, config.owner_id,
                              processing_timeout=config.processing_timeout,
                              voice=VoiceClient(voice_http, config.voice) if config.voice.enabled else None).run()
            finally:
                store.db.close()


def main():
    os.umask(0o077)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    logging.getLogger(__name__).setLevel(logging.INFO)
    try:
        asyncio.run(run(Config.load()))
    except KeyboardInterrupt:
        pass
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print('Не удалось запустить бота: ' + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
