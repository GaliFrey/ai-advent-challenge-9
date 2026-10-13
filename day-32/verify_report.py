"""Revalidate a saved Actions artifact against local immutable Git objects."""
import argparse
import json
from pathlib import Path

from review import validate
from snapshot import Source, collect


def check(project, report):
    if report.get('status') != 'accepted':
        raise ValueError('Отчёт не содержит принятого ревью.')
    payload = report['payload']
    snapshot = collect(project, payload['base'], payload['head'])
    if payload['merge_base'] != snapshot['merge_base']:
        raise ValueError('Merge-base отчёта не совпадает с Git.')
    expected = [{k: v for k, v in c.items() if k != 'lines'} for c in snapshot['changes']]
    actual = [{k: v for k, v in c.items() if k != 'rag_sources'} for c in payload['changes']]
    if actual != expected:
        raise ValueError('Diff/список изменений отчёта не совпадает с Git.')
    sources = [Source(**s) for s in payload['sources']]
    direct = {s.id: s for s in snapshot['direct']}
    seen = set()
    corpus = {s.path: s for s in snapshot['corpus']}
    for source in sources:
        if source.id in seen:
            raise ValueError('Повторный ID источника.')
        seen.add(source.id)
        if source.id in direct:
            if source != direct[source.id]:
                raise ValueError('Изменённый текст отчёта не совпадает с Git.')
        else:
            if (not source.id.startswith('R') or source.revision != snapshot['head']
                    or source.path not in corpus or source.kind != corpus[source.path].kind):
                raise ValueError('Неверная принадлежность RAG-источника.')
            lines = corpus[source.path].text.splitlines(keepends=True)
            if not 1 <= source.start <= source.end <= len(lines):
                raise ValueError('Неверные строки RAG-источника.')
            if source.text not in ''.join(lines[source.start - 1:source.end]):
                raise ValueError('RAG-фрагмент отсутствует в указанном диапазоне Git-снимка.')
    if not set(direct).issubset(seen):
        raise ValueError('В отчёте отсутствуют изменённые файлы.')
    answer = validate(json.dumps(report['answer']), snapshot, sources)
    if 'model_response' in report:
        resolved = validate(report['model_response'], snapshot, sources, resolve_quotes=True)
        if resolved != answer:
            raise ValueError('Сохранённое ревью не совпадает с ответом модели и разрешёнными источниками.')
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    answer = check(args.project, json.loads(args.report.read_text()))
    print(f"Отчёт подтверждён Git-снимком: {len(answer['findings'])} замечаний. LLM-вызовов: 0.")


if __name__ == '__main__':
    main()
