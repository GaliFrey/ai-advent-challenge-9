"""Persistent sequential chat: planning, fresh retrieval and verified evidence."""
import copy
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evidence import messages, refusal, validate
from memory import empty_state, history_window, plan_messages, validate_plan, validate_state, prune_resolved_questions
from retrieval import DEFAULT_INDEX, Retriever, Reranker, RERANKER_ID, RERANKER_REVISION
from retrieval import search_candidates, rerank_candidates, select_context
from shared import rag, load_day23
from session_log import SessionLog

_helpers = load_day23('pipeline')
save_report = _helpers.save_report
DAY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    top_k_before: int = 20
    top_k_after: int = 5
    rerank_threshold: float = .1

    def __post_init__(self):
        import math
        if (type(self.top_k_before) is not int or type(self.top_k_after) is not int
            or not 1 <= self.top_k_after <= self.top_k_before <= 100
            or not math.isfinite(self.rerank_threshold) or not 0 <= self.rerank_threshold <= 1):
            raise ValueError('Некорректные настройки поиска')


def new_session(title='Новый диалог', settings=None):
    return {'day': 25, 'schema_version': 1, 'id': uuid4().hex,
            'created_at': datetime.now(timezone.utc).isoformat(), 'title': title,
            'settings': asdict(settings or Settings()), 'state': empty_state(), 'turns': []}


def load_session(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('day') != 25 or data.get('schema_version') != 1 or not isinstance(data.get('turns'), list):
        raise ValueError('Ожидался диалог дня 25')
    Settings(**data['settings'])
    turns = data['turns']
    for number, turn in enumerate(turns, 1):
        if turn.get('number') != number or not isinstance(turn.get('question'), str):
            raise ValueError('Некорректная история')
        validate_state(turn['state_before'], turns[:number - 1])
        if 'state_after' in turn:
            validate_state(turn['state_after'], turns[:number])
        if turn.get('response'):
            _, checks = validate(json.dumps(turn['response'], ensure_ascii=False), turn.get('chunks', []))
            if not checks['passed']:
                raise ValueError('Невалидный сохранённый ответ')
    validate_state(data['state'], turns)
    return data


class Runner:
    def __init__(self, *, key, model='deepseek-flash', completion=rag.complete,
                 retriever_factory=Retriever, reranker_factory=Reranker):
        self.key, self.model, self.completion = key, model, completion
        self.retriever_factory, self.reranker_factory = retriever_factory, reranker_factory
        self.retriever = self.reranker = None

    def send(self, session, question, path, notify=lambda session, stage: None,
             cancelled=lambda: False, log=None):
        if not self.key.strip():
            raise ValueError('Задайте DEEPSEEK_API_KEY')
        if not question.strip() or len(question) > 4000:
            raise ValueError('Нужна реплика длиной 1–4000 символов')
        log = log or SessionLog()
        turn = {'number': len(session['turns']) + 1, 'question': question.strip(),
                'status': 'running', 'state_before': copy.deepcopy(session['state']), 'steps': []}
        session['turns'].append(turn)
        session['model'] = self.model
        session['parameters'] = rag.PARAMETERS
        session['reranker'] = {'model': RERANKER_ID, 'revision': RERANKER_REVISION}
        if turn['number'] == 1 and session['title'] == 'Новый диалог':
            session['title'] = question.strip()[:60]
        stage = 'save'

        def publish():
            save_report(path, session)
            notify(copy.deepcopy(session), stage)

        def step(name, operation):
            nonlocal stage
            stage = name
            if cancelled():
                raise InterruptedError
            entry = {'stage': name, 'status': 'running'}
            turn['steps'].append(entry)
            publish()
            started = time.perf_counter()
            try:
                with log.stage(name, question_number=turn['number']):
                    result = operation()
                entry['status'] = 'complete'
                return result
            except Exception as error:
                entry.update(status='failed', error_type=type(error).__name__)
                raise
            finally:
                entry['seconds'] = round(time.perf_counter() - started, 3)

        def call(name, request):
            turn[name + '_messages'] = request
            output = step(name, lambda: self.completion(request, self.key, self.model))
            turn[name + '_output'] = output
            return output['answer']

        try:
            publish()
            history = history_window(session['turns'][:-1])
            raw = call('plan', plan_messages(question, turn['number'], session['state'], history,
                                            session['turns'][:-1]))
            plan = step('validate', lambda: validate_plan(raw, session['turns'], session['state']))
            turn.update(plan=plan, state_after=copy.deepcopy(plan['state']))
            session['state'] = copy.deepcopy(plan['state'])
            publish()
            if self.retriever is None:
                retriever = step('index', lambda: self.retriever_factory(DEFAULT_INDEX))
                step('embedding_model', retriever.load)
                self.retriever = retriever
            session['index'] = self.retriever.metadata
            if self.reranker is None:
                self.reranker = step('reranker_model', self.reranker_factory)
            settings = Settings(**session['settings'])
            searches = plan['searches']
            turn['searches'] = searches
            chunks = step('search', lambda: search_candidates(self.retriever, searches, settings.top_k_before))
            ranked = step('rerank', lambda: rerank_candidates(self.reranker, searches, chunks))
            candidates, selected = step('filter', lambda: select_context(ranked, settings.top_k_after,
                                                settings.rerank_threshold, len(searches)))
            turn.update(candidates=candidates, chunks=selected)
            publish()
            if selected:
                raw = call('answer', messages(question, plan['resolved_question'], session['state'], history, selected))
            else:
                raw = json.dumps(refusal(), ensure_ascii=False)
                turn['origin'] = 'threshold'
            turn['raw_answer'] = raw
            data, checks = step('validate', lambda: validate(raw, selected))
            turn.update(response=data if checks['passed'] else None, checks=checks,
                        status='complete' if checks['passed'] else 'invalid')
            if checks['passed']:
                session['state'] = prune_resolved_questions(session['state'], session['turns'])
                turn['state_after'] = copy.deepcopy(session['state'])
        except InterruptedError:
            turn['status'] = 'cancelled'
        except Exception as error:
            turn.update(status='failed', error={'stage': stage, 'type': type(error).__name__})
        publish()
        return turn
