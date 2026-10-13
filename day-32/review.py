"""Review contract, auditable context bundle, and Markdown report."""
import html
import json
from pathlib import Path

from provider import complete
from rag import MODEL_ID, MODEL_REVISION, retrieve


class ValidationError(ValueError):
    """A rejected citation with enough context to reproduce the rejection."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


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
    for finding_number, finding in enumerate(value['findings'], 1):
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
        for evidence_number, evidence in enumerate(finding['evidence'], 1):
            details = {'finding': finding_number, 'evidence': evidence_number,
                       'file': finding['file'], 'side': finding['side'], 'line': finding['line']}
            if (not isinstance(evidence, dict) or set(evidence) != {'source', 'quote'}
                    or not isinstance(evidence['source'], str)):
                raise ValidationError('Неверный контракт цитаты.', details | {'reason': 'citation_contract'})
            source_id = evidence['source']
            details['source'] = source_id
            if source_id not in available:
                raise ValidationError(
                    f'Неизвестный источник цитаты: {json.dumps(source_id[:80], ensure_ascii=False)}.',
                    details | {'reason': 'unknown_source'})
            source = available[source_id]
            details.update(source_file=source.path, source_start=source.start, source_end=source.end)
            quote = evidence['quote']
            if not isinstance(quote, str) or not quote.strip():
                raise ValidationError('Цитата должна быть непустой строкой.',
                                      details | {'reason': 'invalid_quote'})
            if quote not in source.text:
                details.update(reason='quote_not_found', quote=quote)
                preview = json.dumps(quote[:240], ensure_ascii=False)
                raise ValidationError(
                    f'Цитата отсутствует в указанном источнике: замечание {finding_number}, '
                    f'цитата {evidence_number}, {source_id}, '
                    f'{source.path}:{source.start}–{source.end}; получено {preview}.', details)
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
        if 'rejected_response' in result:
            out += ['Отклонённый ответ сохранён в `review.json` → `rejected_response`; '
                    'он не является принятым ревью.', '']
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
        try:
            answer = validate(content, snapshot, sources)
        except ValueError as error:
            result['rejected_response'] = content
            if isinstance(error, ValidationError):
                result['validation_error'] = error.details
            raise
        result['answer'] = answer
        result['status'] = 'accepted'
    except (ValueError, RuntimeError) as error:
        result['error'] = str(error)
        # The key is not sent to the model; still prevent an echoed key from
        # appearing in newly added diagnostic fields or console messages.
        if key:
            for field in ('error', 'rejected_response', 'validation_error'):
                if field in result:
                    # A raw JSON answer or a quote preview may contain the
                    # escaped representation of the key rather than its text.
                    variants = {key, json.dumps(key)[1:-1], json.dumps(key, ensure_ascii=False)[1:-1]}
                    for variant in variants:
                        diagnostic = result[field]
                        result[field] = (diagnostic.replace(variant, '[REDACTED]')
                            if isinstance(diagnostic, str) else {
                                name: value.replace(variant, '[REDACTED]') if isinstance(value, str) else value
                                for name, value in diagnostic.items()})
    return result


def save(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Do not persist the whole corpus. Persist the exact prompt and reviewed answer.
    report = {k: v for k, v in result.items() if k not in {'snapshot', 'sources'}}
    (output / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'review.md').write_text(markdown(result), encoding='utf-8')
