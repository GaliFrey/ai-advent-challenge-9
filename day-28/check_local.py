"""Opt-in integration: actual TUI, stdio MCP and three local Ollama turns; no cloud."""
import asyncio
import json
from pathlib import Path

from clients import CloudClient, DocsClient, LocalClient
from core import DAY, load_session
from pipeline import Runner
from tui import RagApp
from textual.widgets import Input, Switch, TextArea


async def main():
    # Deliberately do not load .env or cloud credentials in integration checks.
    app = RagApp(runner=Runner(DocsClient(), LocalClient(), CloudClient('')))
    async with app.run_test(size=(180, 60)) as pilot:
        app.query_one('#cloud', Switch).value = False
        app.query_one('#top_k', Input).value = '3'
        app.query_one('#notes', TextArea).load_text('Серверный SP-XML; нужны документированные контракты функций.')
        for question in ['Как ведёт себя ArrayOptFirstElem для пустого массива без второго аргумента?',
                         'А если сам массив undefined?']:
            app.query_one('#question', TextArea).load_text(question)
            app.action_send()
            if app.generation_task is None:
                raise RuntimeError('TUI не запустил запрос.')
            await asyncio.wait_for(app.generation_task, 180)
            await pilot.pause()
        app.action_send(repeat=0)
        await asyncio.wait_for(app.generation_task, 180)
        await pilot.pause()
    restored = load_session(DAY / 'sessions' / (app.session['id'] + '.json'))
    report = {'session': str(Path('sessions') / (restored['id'] + '.json')), 'cloud_calls': 0,
              'turns': [{'question': t['question'], 'status': t['status'], 'plan': t.get('plan'),
                         'sources': len(t['sources']), 'context_sha256': t.get('context', {}).get('sha256'),
                         'metrics': t['results']['local']['metrics'],
                         'checks': t['results']['local'].get('checks'), 'error': t.get('error')}
                        for t in restored['turns']]}
    report['same_context_on_repeat'] = bool(report['turns'][0]['context_sha256']) and (
        report['turns'][0]['context_sha256'] == report['turns'][2]['context_sha256'])
    path = DAY / 'resources/local-check.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(t['status'] == 'complete' for t in report['turns']) or not report['same_context_on_repeat']:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
