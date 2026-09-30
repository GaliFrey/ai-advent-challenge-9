"""Real local retrieval/reranking check with deliberately substituted LLM."""
import json
from datetime import datetime, timezone

from main import load_cases
from pipeline import DAY, Runner
from session_log import SessionLog


def local_completion(messages, key, model):
    question = json.loads(messages[1]['content'])['question']
    is_rewrite = 'query' in messages[0]['content']
    return {'answer': json.dumps({'query': question}, ensure_ascii=False) if is_rewrite else
            'Подменный ответ для локальной проверки; качество генерации не оценивалось.',
            'usage': {'total_tokens': 0}, 'llm_seconds': 0, 'finish_reason': 'stop'}


if __name__ == '__main__':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path = DAY / 'resources' / f'local-check-{stamp}.json'
    report = Runner(key='local-placeholder', completion=local_completion).run(load_cases(), path, SessionLog())
    report['validation_only'] = True
    from pipeline import save_report
    save_report(path, report)
    print('Подмена LLM; настоящие индекс, эмбеддинги и reranker.')
    print('Статус:', report['status'])
    print('Результат:', path)
    if report.get('error'):
        print(report['error'])
    for item in report['items']:
        print(item['id'], {mode: len(result['chunks']) for mode, result in item['results'].items()})
    raise SystemExit(0 if report['status'] == 'complete' else 1)
