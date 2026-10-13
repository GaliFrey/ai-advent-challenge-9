"""Review contract, auditable context bundle, and Markdown report."""
import html
import json
from pathlib import Path

from provider import complete
from rag import MODEL_ID, MODEL_REVISION, retrieve


def prepare(snapshot, model):
    retrieved, associations, count = retrieve(snapshot, model)
    sources = snapshot['direct'] + retrieved
    payload = {'base': snapshot['base'], 'merge_base': snapshot['merge_base'],
               'head': snapshot['head'], 'changes': [
                   {k: v for k, v in change.items() if k != 'lines'} |
                   {'rag_sources': associations[change['path']]}
                   for change in snapshot['changes']],
               'sources': [s.payload() for s in sources]}
    # Bound total request data before making a paid call.
    if len(json.dumps(payload, ensure_ascii=False).encode()) > 300_000:
        raise ValueError('Контекст больше 300 КБ; ревью остановлено без усечения.')
    return payload, sources, {'model': MODEL_ID, 'revision': MODEL_REVISION,
                             'documents': len(snapshot['corpus']), 'chunks': count,
                             'retrieved': len(retrieved), 'associations': associations}


def validate(content, snapshot, sources):
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        raise ValueError('Ревью не является JSON.') from None
    if (not isinstance(value, dict) or set(value) != {'summary', 'findings'}
            or not isinstance(value['summary'], str) or not value['summary'].strip()
            or len(value['summary']) > 1500 or not isinstance(value['findings'], list)
            or len(value['findings']) > 20):
        raise ValueError('Неверный контракт ревью.')
    changed = {c['path']: c for c in snapshot['changes']}
    available = {s.id: s for s in sources}
    for finding in value['findings']:
        if not isinstance(finding, dict) or set(finding) != {
                'category', 'priority', 'file', 'side', 'line', 'description', 'evidence'}:
            raise ValueError('Неверный контракт замечания.')
        if (not isinstance(finding['file'], str) or finding['file'] not in changed
                or finding['side'] not in ('head', 'base')
                or finding['category'] not in ('bug', 'architecture', 'recommendation')
                or finding['priority'] not in ('P1', 'P2', 'P3')
                or type(finding['line']) is not int
                or finding['line'] not in changed[finding['file']]['lines'][finding['side']]
                or not isinstance(finding['description'], str)
                or not 20 <= len(finding['description'].strip()) <= 3000
                or not isinstance(finding['evidence'], list) or not finding['evidence']):
            raise ValueError('Замечание не относится к изменённой строке или имеет неверные поля.')
        direct_id = changed[finding['file']]['sources'].get(finding['side'])
        cited = set()
        for evidence in finding['evidence']:
            if (not isinstance(evidence, dict) or set(evidence) != {'source', 'quote'}
                    or not isinstance(evidence['source'], str)
                    or evidence['source'] not in available
                    or not isinstance(evidence['quote'], str)
                    or not evidence['quote'].strip()
                    or evidence['quote'] not in available[evidence['source']].text):
                raise ValueError('Цитата отсутствует в указанном источнике.')
            cited.add(evidence['source'])
        if direct_id not in cited:
            raise ValueError('Нет цитаты из изменённого файла на указанной стороне diff.')
    return value


def markdown(result):
    snapshot, answer = result['snapshot'], result.get('answer')
    out = ['# AI-ревью PR', '', f"Base: `{snapshot['base']}`",
           f"Merge base: `{snapshot['merge_base']}`", f"Head: `{snapshot['head']}`", '']
    if answer is None:
        out += ['**Ревью не выполнено:** ' + html.escape(result.get('error', 'retrieve-only')), '']
    else:
        out += [html.escape(answer['summary']), '']
        for category, title in [('bug', 'Потенциальные баги'), ('architecture', 'Архитектурные проблемы'),
                                ('recommendation', 'Рекомендации')]:
            out += ['## ' + title, '']
            findings = [f for f in answer['findings'] if f['category'] == category]
            if not findings:
                out += ['Замечаний не найдено.', '']
            for f in findings:
                out += [f"- **{f['priority']} · {html.escape(f['file'])}:{f['line']} ({f['side']})** — "
                        + html.escape(f['description']), '']
                for evidence in f['evidence']:
                    source = next(s for s in result['sources'] if s.id == evidence['source'])
                    quote = ' '.join(evidence['quote'].split())
                    out += [f"  Источник {source.id}: {html.escape(source.path)}:{source.start}–{source.end} "
                            f"({source.revision[:12]}) — {html.escape(quote)}", '']
    out += ['## Контекст и ограничения', '',
            f"RAG: {result['rag']['documents']} файлов, {result['rag']['chunks']} чанков, "
            f"{result['rag']['retrieved']} выбранных фрагментов.", '',
            'Изменённые файлы: ' + ', '.join(html.escape(c['path']) for c in snapshot['changes']), '',
            'Код PR не выполнялся. Цитаты и позиции проверяются механически; '
            'правильность выводов требует проверки человеком. Отсутствие замечаний не доказывает отсутствие багов.', '']
    for change in snapshot['changes']:
        if change.get('skipped'):
            out += ['Не прочитано как текст: ' + html.escape(change['path']) + ' — '
                    + ', '.join(change['skipped']) + '.', '']
    return '\n'.join(out)


def run(snapshot, model, key='', llm_model='deepseek-flash', retrieve_only=False, completion=complete):
    payload, sources, stats = prepare(snapshot, model)
    result = {'snapshot': snapshot, 'sources': sources, 'rag': stats, 'payload': payload,
              'status': 'retrieved' if retrieve_only else 'failed'}
    if retrieve_only:
        return result
    try:
        content, metadata = completion(payload, key, llm_model)
        result['llm'] = metadata
        result['answer'] = validate(content, snapshot, sources)
        result['status'] = 'accepted'
    except (ValueError, RuntimeError) as error:
        result['error'] = str(error)
    return result


def save(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Do not persist the whole corpus. Persist the exact prompt and reviewed answer.
    report = {k: v for k, v in result.items() if k not in {'snapshot', 'sources'}}
    (output / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'review.md').write_text(markdown(result), encoding='utf-8')
