"""CLI entry point for chat, persisted replay and two long scenarios."""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from evidence import answer_text
from pipeline import DAY, Runner, load_session
from benchmark import run_scenario
from scenarios import SCENARIOS
from retrieval import Reranker


def main(argv=None):
    parser = argparse.ArgumentParser(description='День 25: RAG-чат с историей и памятью задачи')
    commands = parser.add_subparsers(dest='command', required=True)
    ui = commands.add_parser('tui', help='Полноэкранный чат; до 2 LLM-вызовов на реплику')
    ui.add_argument('session', nargs='?', type=Path)
    show = commands.add_parser('show', help='Офлайн-просмотр диалога')
    show.add_argument('session', type=Path)
    ask = commands.add_parser('ask', help='Продолжить диалог; до 2 LLM-вызовов')
    ask.add_argument('session', type=Path)
    ask.add_argument('question')
    bench = commands.add_parser('benchmark', help='Два сценария, до 72 LLM-вызовов')
    bench.add_argument('--scenario', choices=['memory', 'mcp', 'all'], default='all')
    commands.add_parser('prepare', help='Подготовить локальный reranker')
    args = parser.parse_args(argv)
    try:
        if args.command == 'show':
            session = load_session(args.session)
            if session.get('validation_only'):
                print('ЛОКАЛЬНАЯ ПРОВЕРКА: синтетические LLM-ответы, качество не оценивалось.')
            for turn in session['turns']:
                print(f"\nРеплика {turn['number']}: {turn['question']} [{turn['status']}]")
                print(answer_text(turn.get('response')))
                print('Источники:', json.dumps((turn.get('response') or {}).get('sources', []), ensure_ascii=False))
                print('Оценка LLM:', json.dumps(turn.get('assessment', {} ).get('result'), ensure_ascii=False))
            return 0
        if args.command == 'tui':
            from tui import ChatApp
            ChatApp(session_path=args.session).run()
            return 0
        if args.command == 'prepare':
            Reranker()
            print('Reranker готов')
            return 0
        load_dotenv(DAY / '.env', override=False)
        runner = Runner(key=os.getenv('DEEPSEEK_API_KEY', ''), model=os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'))
        if args.command == 'ask':
            from pipeline import new_session
            session = load_session(args.session) if args.session.exists() else new_session()
            if session.get('validation_only'):
                raise ValueError('Нельзя продолжать синтетический отчёт живыми ответами')
            turn = runner.send(session, args.question, args.session)
            print(answer_text(turn.get('response')))
            return 0 if turn['status'] == 'complete' else 1
        if not runner.key.strip():
            raise ValueError('Нет ключа')
        failed = False
        for scenario in SCENARIOS:
            if args.scenario not in ('all', scenario['id']):
                continue
            from datetime import datetime, timezone
            path = DAY / 'sessions' / f"scenario-{scenario['id']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
            print(f'До 36 LLM-вызовов: {path}')
            session = run_scenario(runner, scenario, path)
            good = len(session['turns']) == 12 and all(
                t.get('assessment', {}).get('result', {}).get('verdict') == 'pass' for t in session['turns'])
            failed |= not good
            print('pass' if good else 'fail / неполный прогон')
        return int(failed)
    except Exception as error:
        print(f'Ошибка: {type(error).__name__}. Проверьте настройки и сохранённую диагностику.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
