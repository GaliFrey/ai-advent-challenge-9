"""CLI for preflight, one paired case, full experiment and offline report review."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from cases import load_cases
from clients import LocalClient
from core import BASELINE, DAY, Settings, load_report, profiles, save_report, status_label
from engine import Engine
from judge import DIMENSIONS, Judge, decision, quality, validate_review
from ollama import OllamaManager
from resources_monitor import gpu_memory


def configured_judge(disabled=False):
    load_dotenv(DAY / '.env')
    return Judge('' if disabled else os.getenv('DEEPSEEK_API_KEY', ''),
                 os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'))


def report_text(report):
    lines = [f'Отчёт: {report["id"]} | {report["kind"]} | {status_label(report["status"])}']
    if report.get('error'):
        lines.append(report['error'])
    for step in report.get('steps', []):
        lines.append(f'{step["profile"]["id"]}: {status_label(step.get("decision", {}).get("status", step.get("status")))} — '
                     + step.get('decision', {}).get('reason', ''))
    if report.get('final') and report['final'].get('decision'):
        final = report['final']['decision']
        lines.append('Итог: ' + final['reason'])
        lines.append('Выбран профиль: ' + report.get('winner', BASELINE)['id'])
    lines.append('Случай | Повтор | Сторона | Профиль | Статус | Время, с | Токенов/с | RAM MiB | GPU MiB | Оценка')
    def number(value, divisor=1):
        return f'{value/divisor:.2f}' if isinstance(value, (float, int)) else 'н/д'
    for comparison in report['comparisons']:
        for side in ('before', 'after'):
            result = comparison.get(side, {})
            metrics = result.get('metrics', {})
            review = quality(comparison, side)
            score = '/'.join(str(review[k]) for k in DIMENSIONS) if review else 'нужна проверка'
            role = 'Первый профиль' if side == 'before' else 'Второй профиль'
            lines.append(' | '.join([comparison['case_id'], str(comparison['repeat']+1), role, result.get('profile', {}).get('id', side),
                status_label(result.get('status', 'pending')), number(metrics.get('wall_seconds')), number(metrics.get('tokens_per_second')),
                number(metrics.get('max_ram_bytes'), 1024**2), number(metrics.get('max_gpu_bytes'), 1024**2), score]))
    return '\n'.join(lines)


async def execute(args):
    judge = configured_judge(getattr(args, 'no_judge', False))
    if args.command == 'preflight':
        state = await OllamaManager().snapshot()
        result = {'ollama_available': state['available'], 'models': state['models'],
                  'running': state['running'], 'gpu': await gpu_memory(), 'judge_key_present': judge.available}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if state['available'] else 1
    manager = OllamaManager()
    try:
        if args.start_ollama:
            await manager.start(Settings())
        if args.command == 'prepare':
            import httpx
            import time
            async with httpx.AsyncClient(base_url='http://127.0.0.1:11434', trust_env=False,
                    timeout=httpx.Timeout(600, connect=5)) as client:
                tags = await client.get('/api/tags')
                tags.raise_for_status()
                if any(m['name'] == 'qwen3:14b-q8_0' for m in tags.json()['models']):
                    print('Q8_0 уже установлена; загрузка пропущена.')
                else:
                    last = 0
                    async with client.stream('POST', '/api/pull', json={'model': 'qwen3:14b-q8_0', 'stream': True}) as response:
                        response.raise_for_status()
                        successful = False
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            if data.get('error'):
                                raise ValueError('Ollama не смогла загрузить Q8_0.')
                            if time.monotonic() - last > 5 or data.get('status') == 'success':
                                print(data.get('status', 'Загрузка'), data.get('completed', ''), '/', data.get('total', ''), flush=True)
                                last = time.monotonic()
                            successful |= data.get('status') == 'success'
                        if not successful:
                            raise ValueError('Загрузка не завершена; модель не считается установленной.')
            metadata = await LocalClient().metadata('qwen3:14b-q8_0')
            path = DAY / 'resources' / 'model-preparation.json'
            path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            print('Модель проверена; метаданные сохранены: ' + str(path))
            return 0
        engine = Engine(judge=judge, notify=lambda value: print(value, flush=True))
        if args.command == 'compare':
            if args.case:
                case = next(c for c in load_cases() if c['id'] == args.case)
            else:
                case = await engine.retrieve_manual(args.question, args.notes, args.source)
            options = profiles()
            after = options[args.after]
            if args.profile_report:
                saved = load_report(args.profile_report)
                after = saved.get('winner') or saved.get('selected_profile')
                if after is None:
                    raise ValueError('В отчёте ещё нет выбранного профиля.')
            report = await engine.manual(case, options[args.before], after)
        else:
            report = await engine.experiment(load_report(args.resume) if args.resume else None)
        print(report_text(report))
        print('Сохранено: ' + str(DAY / 'reports' / (report['id'] + '.json')))
        if report['status'] != 'complete':
            return 1
        if report['kind'] == 'experiment':
            return 0
        return 0 if all(c['status'] == 'complete' for c in report['comparisons']) else 1
    finally:
        await manager.stop()


def parser():
    value = argparse.ArgumentParser(description='День 29: оптимизация локального WebTutor RAG')
    commands = value.add_subparsers(dest='command', required=True)
    commands.add_parser('preflight', help='Проверка Ollama, GPU и наличия ключа без генерации')
    prepare = commands.add_parser('prepare', help='Загрузить отсутствующую Qwen3:14b Q8_0 (~16 GB) и проверить метаданные')
    prepare.add_argument('--start-ollama', action='store_true')
    compare = commands.add_parser('compare', help='Последовательно сравнить два профиля')
    source = compare.add_mutually_exclusive_group(required=True)
    source.add_argument('--case', choices=[c['id'] for c in load_cases()])
    source.add_argument('--question')
    compare.add_argument('--notes', default='')
    compare.add_argument('--source', choices=['docs', 'datex', 'portal', 'schema'], default='docs')
    compare.add_argument('--before', choices=list(profiles()), default='baseline')
    compare.add_argument('--after', choices=list(profiles()), default='compact-prompt')
    compare.add_argument('--profile-report', type=Path)
    compare.add_argument('--no-judge', action='store_true', help='Без внешних API; предметная оценка остаётся ручной')
    compare.add_argument('--start-ollama', action='store_true', help='Запустить собственный сервер из установки дня 28')
    experiment = commands.add_parser('experiment', help='Шесть этапов настройки и итоговые повторы; использует DeepSeek')
    experiment.add_argument('--resume', type=Path)
    experiment.add_argument('--start-ollama', action='store_true')
    report = commands.add_parser('report', help='Читать отчёт без API')
    report.add_argument('path', type=Path)
    review = commands.add_parser('review', help='Сохранить ручную оценку спорного сравнения без API')
    review.add_argument('path', type=Path)
    review.add_argument('--comparison', type=int, required=True, help='Индекс сравнения, начиная с 0')
    review.add_argument('--side', choices=['before', 'after'], required=True)
    review.add_argument('--scores', required=True, help='correctness,completeness,grounding,refusal: четыре числа 0–2')
    review.add_argument('--critical-error', action='append', default=[])
    review.add_argument('--reason', required=True)
    return value


def main():
    args = parser().parse_args()
    try:
        if args.command == 'report':
            print(report_text(load_report(args.path)))
            return 0
        if args.command == 'review':
            report = load_report(args.path)
            if not 0 <= args.comparison < len(report['comparisons']):
                raise ValueError('Индекс сравнения вне отчёта.')
            scores = [int(v) for v in args.scores.split(',')]
            if len(scores) != 4:
                raise ValueError('Нужны четыре оценки.')
            review = dict(zip(DIMENSIONS, scores), critical_errors=args.critical_error,
                          needs_manual_review=False, reason=args.reason, reviewer='human')
            validate_review(json.dumps({'A': review, 'B': review}))
            comparison = report['comparisons'][args.comparison]
            comparison.setdefault('manual_history', []).append({'side': args.side, 'review': review})
            comparison.setdefault('manual', {})[args.side] = review
            comparison['decision'] = decision([comparison])
            save_report(report, args.path.parent)
            print('Ручная оценка сохранена. Для пересчёта выбора: experiment --resume с этим отчётом.')
            return 0
        return asyncio.run(execute(args))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        from engine import safe_error
        print(safe_error(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
