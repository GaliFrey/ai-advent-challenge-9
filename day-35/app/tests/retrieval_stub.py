"""Fixture-only adapter for unrelated tests; never an LLM quality evaluator.

Answers come from scripted main responses retained by the fake model, not from
collector payloads. Dedicated collection tests script each decision explicitly.
"""
import json


def retrieval_reply(messages, model=None):
    data = json.loads(messages[-1]['content'])
    ids = [n['id'] for n in data.get('candidates', data.get('evidence', []))]
    if data['task'] == 'scope':
        response = dict(scope='single', target='Предмет тестового вопроса',
                        criterion='Факт для тестового вопроса', query='')
    elif data['task'] == 'select':
        response = dict(sufficient=bool(ids), ids=ids)
    else:
        fixtures = getattr(model, 'retrieval_fixtures', [])
        index = len(fixtures) - len(getattr(model, 'responses', [])) - 1
        text = fixtures[index].get('content', '') if index >= 0 else 'Ответ на вопрос'
        try:
            text = json.loads(text).get('text', text)
        except (ValueError, AttributeError):
            pass
        reviews = [dict(id=n['id'], decision='include',
                        quote=next(p['content'][:100] for p in n['fragments'] if p.get('content')),
                        relation='Заданный тестовый факт', status='unknown') for n in data['evidence']]
        response = dict(answer=text, ids=ids, reviews=reviews)
    return {'role': 'assistant', 'content': json.dumps(response, ensure_ascii=False)}
