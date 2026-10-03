"""Real cached retrieval, synthetic LLM: proves integration, not answer quality."""
import json
from datetime import datetime, timezone
from benchmark import run_scenario
from pipeline import DAY, Runner
from scenarios import SCENARIOS


def synthetic(messages, key, model):
    payload = json.loads(messages[1]['content'])
    if 'current_turn' in payload:
        case = next(c for s in SCENARIOS for c in s['turns'] if c['question'] == payload['question'])
        data = {'query': case['resolved'], 'resolved_question': case['resolved'], 'state': case['expected_state'],
                'searches': [{'query': case['resolved'], 'source': None}]}
    elif 'expected_state' in payload:
        data = {'checks': {'support': True, 'memory': True, 'continuity': False}, 'verdict': 'fail',
                'explanation': 'Синтетическая оценка: выдержка проверяет цитату, но не отвечает на вопрос. Качество LLM не оценивалось.'}
    else:
        c = payload['sources'][0]
        data = {'status': 'answered', 'answer': [{'text': 'Тестовая выдержка; это не ответ на вопрос.',
                'citations': [{'source_id': c['source_id'], 'quote': c['text'][:240]}]}],
                'sources': [{k: c[k] for k in ('source_id', 'source', 'section', 'chunk_id')}], 'clarification': ''}
    return {'answer': json.dumps(data, ensure_ascii=False), 'usage': {}, 'llm_seconds': 0}


if __name__ == '__main__':
    runner = Runner(key='offline-placeholder', completion=synthetic)
    failed = False
    for scenario in SCENARIOS:
        path = DAY / 'sessions' / f"local-{scenario['id']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
        session = run_scenario(runner, scenario, path, validation_only=True)
        ok = len(session['turns']) == 12 and all(t['status'] == 'complete' for t in session['turns'])
        print(f"{scenario['id']}: {len(session['turns'])} ходов, интеграция {'OK' if ok else 'FAIL'}, синтетические ответы: {path}", flush=True)
        if not ok:
            print(session['turns'][-1].get('error'))
        failed |= not ok
    raise SystemExit(int(failed))
