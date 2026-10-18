import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import httpx
from personal_assistent.config import Config
from personal_assistent.context import Budget
from personal_assistent.agent import Agent, DeepSeek
from personal_assistent.storage import Store
from personal_assistent.runtime import Runtime
from personal_assistent.requests import Requests

class BoundedDeepSeek(DeepSeek):
    calls = 0
    async def _request(self, messages, purpose):
        self.calls += 1
        if self.calls > 20:
            raise RuntimeError('Live call limit')
        return await super()._request(messages, purpose)

async def main(scenario_filter=None):
    """Paid, opt-in smoke check: synthetic inputs, in-memory databases, no Telegram."""
    config = Config.load()
    async with httpx.AsyncClient() as client:
        model = BoundedDeepSeek(client, config.deepseek_key, temperature=config.temperature, max_tokens=config.max_tokens)
        for scenario, messages in [
            ('clarification', ['Напомни завтра купить молоко.', 'В девять утра.']),
            ('correction', ['Напомни завтра в девять утра.', 'Нет, лучше в десять.', 'Почистить зубы.']),
            ('cancellation', ['Напомни завтра проверить фильтр.', 'Больше не нужно, отмени это поручение.']),
        ]:
            if scenario_filter and scenario != scenario_filter:
                continue
            store = Store(':memory:')
            runtime = Runtime(store, None, Agent(store, model, budget=Budget(config.context_budget, config.context_reserve)), 1)
            tomorrow = (datetime.now(ZoneInfo('Europe/Kirov')) + timedelta(days=1)).date()
            try:
                for number, message in enumerate(messages, 1):
                    now = int(time.time())
                    store.enqueue({'update_id':number, 'message':{'message_id':number,'date':now,'from':{'id':1},'chat':{'id':1,'type':'private'},'text':message}},now)
                    await runtime.process_once()
                    active = Requests(store).active()
                    rows = [dict(r) for r in store.db.execute('SELECT * FROM requests')]
                    reminders = [dict(r) for r in store.db.execute('SELECT * FROM reminders')]
                    reply = store.db.execute('SELECT text FROM outbox ORDER BY rowid DESC LIMIT 1').fetchone()
                    print(f'{scenario}, step {number}: {reply[0] if reply else None}', flush=True)
                    assert store.db.execute('SELECT count(*) FROM notes').fetchone()[0] == 0, 'Unexpected note'
                    if scenario == 'correction' and number < 3:
                        assert reply and reply[0] == 'О чём напомнить?', 'Wrong clarification question'
                    if number < len(messages):
                        assert active and active['status']=='waiting', 'No waiting draft'
                        assert not reminders, 'Premature reminder'
                    if scenario == 'correction' and number == 2:
                        params = json.loads(active['parameters'])
                        assert params.get('due_time','').startswith('10:') or 'T10:' in params.get('due_at',''), 'Correction lost'
                assert len(rows)==1, 'Draft replaced during continuation'
                if scenario == 'cancellation':
                    assert rows[0]['status']=='cancelled' and not reminders and not store.get_meta('pending'), 'Cancellation failed'
                else:
                    assert rows[0]['status']=='done' and len(reminders)==1, 'Request not executed'
                    due = datetime.fromtimestamp(reminders[0]['due_at'],ZoneInfo('Europe/Kirov'))
                    assert due.date()==tomorrow and due.hour==(9 if scenario=='clarification' else 10) and due.minute==0, 'Wrong date/time'
                    assert reminders[0]['source_message_id']==1, 'Original source lost'
                print(scenario+': PASS',flush=True)
            finally:
                store.db.close()
        print(f'PASS; {model.calls} API calls; no Telegram; temporary DBs',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paid DeepSeek smoke check; synthetic data only')
    parser.add_argument('--scenario', choices=('clarification', 'correction', 'cancellation'))
    asyncio.run(main(parser.parse_args().scenario))
