"""Synthetic memory evaluation. Offline by default; never opens the bot database."""
import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from .agent import Agent, DeepSeek
from .context import Budget, Context
from .search import INDEX_VERSION, QUERY_VERSION, index_values
from .storage import Store
from .tools import ToolRunner

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'evaluations' / 'memory.json'
RESULTS = ROOT / 'evaluations' / 'results'


def clone(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Время набора должно содержать часовой пояс')
    return parsed.timestamp()


def load_dataset():
    raw = DATASET.read_bytes()
    data = json.loads(raw)
    notes = {n['id']: n for n in data['notes']}
    cases = data['cases']
    if (len(notes) != len(data['notes']) or len(cases) != 32
            or len({q['id'] for q in cases}) != 32):
        raise ValueError('Нужны уникальные памятки и 32 уникальных вопроса')
    ZoneInfo(data['timezone'])
    timestamp(data['now'])
    timestamp(data['default_saved_at'])
    for note in notes.values():
        if not note['content'].strip():
            raise ValueError('Пустая памятка')
        timestamp(note.get('saved_at', data['default_saved_at']))
        if note.get('event_date'):
            date.fromisoformat(note['event_date'])
    for case in cases:
        if not case['question'].strip() or not case['required_facts'] or not case['sources']:
            raise ValueError('Неполные ожидания вопроса')
        if not set(case['sources']) <= notes.keys():
            raise ValueError('Неизвестный источник')
        for evidence in case['evidence']:
            note = notes[evidence['note_id']]
            if (evidence['note_id'] not in case['sources']
                    or evidence['quote'] not in note.get(evidence.get('field', 'content'), '')):
                raise ValueError('Фрагмент доказательства не найден в источнике')
    return data, hashlib.sha256(raw).hexdigest()


@contextmanager
def memory_store(data):
    # No path option, Config.load(), Runtime, Telegram or scheduler in this runner.
    store = Store(':memory:')
    try:
        with store.db:
            store.set_meta('profile', {'timezone': data['timezone']})
            for number, note in enumerate(data['notes'], 1):
                saved = timestamp(note.get('saved_at', data['default_saved_at']))
                store.db.execute(
                    'INSERT INTO notes(id,content,original,event_date,created_at,updated_at,source_message_id) '
                    'VALUES(?,?,?,?,?,?,?)',
                    (number, note['content'], note['content'], note.get('event_date'), saved, saved, number))
                store.db.execute('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)',
                                 (number, *index_values(note['content'])))
        yield store
    finally:
        store.db.close()


def enqueue(store, question, now):
    store.enqueue({'update_id': 1, 'message': {
        'message_id': 1, 'date': int(now), 'from': {'id': 1},
        'chat': {'id': 1, 'type': 'private'}, 'text': question}}, now)
    return store.next_job()


def source_ids(result, identities):
    if not isinstance(result, dict):
        return []
    items = result.get('items', [result])
    return [identities[item['id']] for item in items
            if isinstance(item, dict) and item.get('id') in identities]


def coverage(sources, expected):
    found = sorted(set(sources) & set(expected))
    return {'found': found, 'missing': sorted(set(expected) - set(found)),
            'fraction': len(found) / len(expected)}


class TracedTools(ToolRunner):
    def __init__(self, store):
        super().__init__(store)
        self.trace = []

    def run(self, call, job, now):
        record = {'call': clone(call)}
        self.trace.append(record)
        started = time.perf_counter()
        try:
            result = super().run(call, job, now)
            record['result'] = clone(result)
            return result
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['seconds'] = time.perf_counter() - started


class CallLimit(RuntimeError):
    pass


class TracedModel:
    def __init__(self, model, allowance):
        self.model, self.allowance = model, allowance
        self.max_tokens = model.max_tokens
        self.trace = []

    async def _call(self, messages, purpose):
        if self.allowance['remaining'] <= 0:
            raise CallLimit('Лимит API-вызовов исчерпан')
        self.allowance['remaining'] -= 1
        record = {'purpose': purpose, 'messages': clone(messages)}
        self.trace.append(record)
        try:
            method = {'work': 'complete', 'summary': 'summarize', 'retrieval': 'retrieve'}[purpose]
            response = await getattr(self.model, method)(messages)
            record['response'] = clone(response)
            return response
        except BaseException as exc:
            # Do not store exception text: HTTP errors can include request details.
            record['error_type'] = type(exc).__name__
            raise

    async def complete(self, messages):
        return await self._call(messages, 'work')

    async def summarize(self, messages):
        return await self._call(messages, 'summary')

    async def retrieve(self, messages):
        return await self._call(messages, 'retrieval')


def model_sources(trace, identities):
    sources = set()
    for record in trace:
        if record['purpose'] == 'retrieval' and 'response' in record:
            for message in record['messages']:
                if message['role'] == 'user':
                    payload = json.loads(message['content'])
                    sources.update(source_ids({'items': payload.get('candidates', []) + payload.get('evidence', [])}, identities))
            continue
        if record['purpose'] != 'work' or 'response' not in record:
            continue
        messages = record['messages']
        names = {c['id']: c['function']['name'] for m in messages
                 for c in (m.get('tool_calls') or [])}
        for message in messages:
            if (message['role'] == 'tool' and names.get(message.get('tool_call_id'))
                    in ('search_notes', 'list_notes', 'get_note')):
                sources.update(source_ids(json.loads(message['content']), identities))
    return sorted(sources)


def token_total(metrics, field):
    values = [m.get(field) for m in metrics]
    return sum(values) if values and all(type(v) is int for v in values) else None


async def evaluate_case(data, case, *, model=None, allowance=None, budget=None, timeout=120, record=None):
    identities = {i: n['id'] for i, n in enumerate(data['notes'], 1)}
    now = timestamp(data['now'])
    result = record if record is not None else {}
    result.update({'id': case['id'], 'group': case['group'], 'question': case['question'],
              'expectations': {k: case[k] for k in ('behavior', 'expected_answer', 'required_facts',
                                                   'sources', 'evidence', 'critical_errors')},
              'status': 'running', 'answer': None,
              'review': {'evidence': None, 'answer': None, 'grounding': None,
                         'critical_error': None, 'cause': None, 'note': None}})
    with memory_store(data) as store:
        job = enqueue(store, case['question'], now)
        tools = TracedTools(store)
        traced = None
        started = time.perf_counter()
        try:
            if model is None:
                call = {'id': 'baseline', 'type': 'function', 'function': {
                    'name': 'search_notes', 'arguments': json.dumps({'query': case['question'], 'limit': 5})}}
                raw = tools.run(call, job, now)
                result['preview'] = json.loads(Context.model_messages([
                    {'role': 'assistant', 'tool_calls': [call]},
                    {'role': 'tool', 'tool_call_id': 'baseline', 'content': json.dumps(raw)}])[-1]['content'])
            else:
                traced = TracedModel(model, allowance)
                agent = Agent(store, traced, clock=lambda: now, budget=budget)
                agent.tools = tools
                await asyncio.wait_for(agent.process(job), timeout)
                reply = store.db.execute("SELECT content FROM history WHERE role='assistant' ORDER BY id DESC LIMIT 1").fetchone()
                if reply is None:
                    raise RuntimeError('Нет окончательного ответа ассистента')
                result['answer'] = reply[0]
            result['status'] = 'complete'
        except Exception as exc:
            result['status'] = 'failed'
            result['error_type'] = type(exc).__name__
        except BaseException as exc:
            result['status'] = 'interrupted'
            result['error_type'] = type(exc).__name__
            raise
        finally:
            result['seconds'] = time.perf_counter() - started
            result['tools'] = tools.trace
            searches = [t for t in tools.trace if t['call']['function']['name'] == 'search_notes']
            first = searches[0] if searches else None
            hits = source_ids(first.get('result', {}), identities) if first else []
            result['first_search'] = {'performed': first is not None, 'sources': hits,
                                      'source_coverage': coverage(hits, case['sources'])}
            result['model_calls'] = traced.trace if traced else []
            checkpoint = store.db.execute('SELECT checkpoint FROM jobs WHERE id=?', (job['id'],)).fetchone()
            result['collection'] = json.loads(checkpoint[0]).get('memory_collection') if checkpoint and checkpoint[0] else None
            sent = model_sources(result['model_calls'], identities)
            result['model_sources'] = sent
            result['model_source_coverage'] = coverage(sent, case['sources']) if traced else None
            result['tool_counts'] = dict(Counter(t['call']['function']['name'] for t in tools.trace))
    return result


def save_report(path, report):
    fd, temporary = tempfile.mkstemp(prefix='.memory-eval-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


async def run(args, data, digest, path):
    cases = [c for c in data['cases'] if not args.case or c['id'] in args.case]
    implementation = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        implementation.update(source.name.encode() + b'\0' + source.read_bytes() + b'\0')
    report = {'schema_version': 1, 'mode': 'assistant' if args.live else 'first_fts_search',
              'created_at': datetime.now(timezone.utc).isoformat(), 'dataset_sha256': digest,
              'implementation_sha256': implementation.hexdigest(), 'python_version': sys.version,
              'dataset': data, 'index_version': INDEX_VERSION, 'query_version': QUERY_VERSION,
              'db_identities': {str(i): n['id'] for i, n in enumerate(data['notes'], 1)},
              'settings': {'model': 'deepseek-flash' if args.live else None,
                           'temperature': args.temperature, 'max_tokens': args.max_tokens,
                           'context_budget': args.context_budget, 'context_reserve': args.context_reserve,
                           'max_calls': args.max_calls, 'case_timeout': args.case_timeout},
              'status': 'running', 'items': []}
    allowance = {'remaining': args.max_calls}
    budget = Budget(args.context_budget, args.context_reserve)
    save_report(path, report)

    async def evaluate_all(model=None):
        for case in cases:
            metrics = []
            if model is not None:
                model.on_call = metrics.append
            item = {}
            report['items'].append(item)
            try:
                await evaluate_case(data, case, model=model, allowance=allowance,
                                    budget=budget, timeout=args.case_timeout, record=item)
            finally:
                item['api_metrics'] = metrics
                item['tokens'] = {key: token_total(metrics, key) for key in ('prompt_tokens', 'completion_tokens')}
                save_report(path, report)
            print(f"{case['id']}: {item['status']}; first search: {','.join(item['first_search']['sources']) or '—'}", flush=True)
            if item['status'] != 'complete':
                report['status'] = 'failed'
                return
        report['status'] = 'complete'

    try:
        if args.live:
            async with httpx.AsyncClient() as client:
                await evaluate_all(DeepSeek(client, os.environ['DEEPSEEK_API_KEY'],
                                            temperature=args.temperature, max_tokens=args.max_tokens))
        else:
            await evaluate_all()
    except BaseException as exc:
        report['status'] = 'interrupted'
        report['error_type'] = type(exc).__name__
        raise
    finally:
        report['api_calls_attempted'] = args.max_calls - allowance['remaining']
        save_report(path, report)
    return report['status'] == 'complete'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Оценка вымышленной памяти: бесплатно FTS; --live включает платный DeepSeek')
    parser.add_argument('--live', action='store_true', help='Явно разрешить платные вызовы для этого запуска')
    parser.add_argument('--case', action='append', help='Например Q09; можно повторять')
    parser.add_argument('--output', type=Path, help='Новый JSON-файл отчёта; существующий не перезаписывается')
    parser.add_argument('--max-calls', type=int, default=192, help='Общий лимит всех API-вызовов, включая сводки')
    parser.add_argument('--case-timeout', type=float, default=120)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, default=2000)
    parser.add_argument('--context-budget', type=int, default=60000)
    parser.add_argument('--context-reserve', type=int, default=2000)
    args = parser.parse_args(argv)
    data, digest = load_dataset()
    if args.case and set(args.case) - {c['id'] for c in data['cases']}:
        parser.error('Неизвестный номер вопроса')
    if (args.max_calls < 1 or not math.isfinite(args.case_timeout) or args.case_timeout <= 0
            or not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2
            or not 128 <= args.max_tokens <= 8192
            or not max(args.max_tokens, 1000) <= args.context_reserve < args.context_budget <= 1000000):
        parser.error('Неверные лимиты или настройки генерации')
    if args.live:
        load_dotenv(ROOT / '.env')
        if not os.environ.get('DEEPSEEK_API_KEY', '').strip():
            parser.error('Для --live нужен DEEPSEEK_API_KEY в .env или окружении')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = args.output or RESULTS / f'memory-{stamp}-{uuid4().hex[:8]}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new filename before any API call; never overwrite another artifact.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        parser.error('Файл отчёта уже существует; укажи новый путь')
    print(f'Отчёт: {path.resolve()}', flush=True)
    return 0 if asyncio.run(run(args, data, digest, path)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
