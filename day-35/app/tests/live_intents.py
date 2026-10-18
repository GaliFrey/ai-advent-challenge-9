import asyncio
import time
import httpx
from personal_assistent.config import Config
from personal_assistent.context import Budget
from personal_assistent.agent import Agent, DeepSeek
from personal_assistent.storage import Store

CASES = [
    ('natural infinitive', 'мне нужно напомнить через 3 минуты почистить зубы', 1, 0, 180),
    ('reminder synonym', 'Поставь напоминание через 7 минут проверить окно', 1, 0, 420),
    ('save fact', 'Сохрани факт: 15 октября 2026 года Оля обещала не пить газировку', 0, 1, None),
    ('remember infinitive', 'Хочу запомнить, что вчера я заменил фильтр', 0, 1, None),
    ('negation', 'Не ставь напоминание про зубы', 0, 0, None),
    ('quoted discussion', 'Что означает фраза «напомни через три минуты»?', 0, 0, None),
    ('ordinary personal story', 'Сегодня Оля обещала не пить газировку.', 0, 0, None),
]

async def main():
    config = Config.load()
    failed = []
    async with httpx.AsyncClient() as client:
        model = DeepSeek(client, config.deepseek_key, temperature=config.temperature, max_tokens=config.max_tokens)
        for number, (name, text, reminders, notes, delta) in enumerate(CASES, 1):
            store = Store(':memory:')
            try:
                now = int(time.time())
                payload = {'update_id': number, 'message': {'message_id': number, 'date': now,
                    'from': {'id': 1}, 'chat': {'id': 1, 'type': 'private'}, 'text': text}}
                store.enqueue(payload, now)
                await Agent(store, model, budget=Budget(config.context_budget, config.context_reserve)).process(store.next_job())
                actual_r = store.db.execute('SELECT count(*) FROM reminders').fetchone()[0]
                actual_n = store.db.execute('SELECT count(*) FROM notes').fetchone()[0]
                ok = (actual_r, actual_n) == (reminders, notes)
                if delta is not None and actual_r:
                    due = store.db.execute('SELECT due_at FROM reminders').fetchone()[0]
                    ok = ok and abs(due - (now + delta)) <= 1
                print(name + ': ' + ('PASS' if ok else 'FAIL'), flush=True)
                if not ok:
                    failed.append(name)
            except Exception as exc:
                print(name + ': ERROR ' + type(exc).__name__, flush=True)
                failed.append(name)
            finally:
                store.db.close()
    if failed:
        raise SystemExit('Live cases failed: ' + ', '.join(failed))
    print('7/7 live semantic checks passed; no Telegram requests; only temporary databases', flush=True)

asyncio.run(main())
