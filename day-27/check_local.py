"""Explicit local integration check: three Ollama requests, no cloud services."""
import asyncio
import json
from pathlib import Path

from textual.widgets import Input, Static, Switch, TextArea
from chat import DAY, load_session
from tui import ChatApp


async def main():
    app = ChatApp()
    async with app.run_test(size=(180, 60)) as pilot:
        app.query_one('#num_predict', Input).value = '512'
        questions = ['Запомни: мой проект называется Север. Ответь одним коротким предложением.',
                     'Как называется мой проект? Ответь только названием.',
                     'Сколько будет 17 умножить на 6? Ответь кратко.']
        for index, question in enumerate(questions):
            app.query_one('#thinking', Switch).value = index == 2
            app.query_one('#input', TextArea).load_text(question)
            app.action_send()
            await asyncio.wait_for(app.generation_task, timeout=120)
            await pilot.pause()
            turn = app.session['turns'][-1]
            assert turn['status'] == 'complete', turn.get('error', turn['status'])
            assert turn['metrics']['tokens_per_second'] > 0
        assert 'север' in app.session['turns'][1]['content'].lower()
        assert '102' in app.session['turns'][2]['content']
        assert app.session['turns'][2]['thinking']
        path = DAY / 'sessions' / (app.session['id'] + '.json')
        restored = load_session(path)
        assert len(restored['turns']) == 3
        assert restored['turns'][1]['context']['included'] == 1
        assert app.query_one('#send').region.bottom <= 60
        sidebar = app.query_one('#sidebar')
        assert sidebar.max_scroll_y == 0, 'Боковая панель не помещается при 180×60'
        geometry = {'sidebar_height': sidebar.size.height, 'sidebar_content_height': sidebar.virtual_size.height,
                    'sidebar_scroll_max': sidebar.max_scroll_y}
        summary = {'session_file': str(path), 'model': 'qwen3:14b',
                   'size': [180, 60], 'geometry': geometry,
                   'turns': [{'status': t['status'], 'thinking': t['settings']['thinking'],
                              'metrics': t['metrics']} for t in restored['turns']]}
        (DAY / 'resources' / 'local-check.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
