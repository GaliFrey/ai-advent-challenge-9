"""One retrieval snapshot, independent providers, reproducible fixed-context repeats."""
import asyncio
import copy
import time
from dataclasses import asdict

from core import fit_sources, memory_snapshot, parse_plan, planner_messages, prepare_messages, provider_memory, validate_answer


def result_state(model):
    return {'model': model, 'status': 'pending', 'raw': '', 'thinking': '', 'metrics': {}}


class Runner:
    def __init__(self, docs, local, cloud):
        self.docs, self.local, self.cloud = docs, local, cloud

    async def run(self, session, question, settings, changed=lambda: None, phase=lambda text: None, repeat=None):
        settings.validate()
        start = time.perf_counter()
        memory = memory_snapshot(session)
        turn = {'question': question, 'status': 'running', 'settings': asdict(settings),
                'memory': memory, 'sources': [], 'results': {'local': result_state(settings.local_model),
                'cloud': result_state(settings.cloud_model)}}
        if repeat is not None:
            turn['repeat_of'] = repeat + 1
        session['turns'].append(turn)
        session['settings'] = asdict(settings)
        session['title'] = session['turns'][0]['question'][:50]
        changed()
        try:
            if repeat is None:
                if settings.planner:
                    phase('Локальный планировщик')
                    request = planner_messages(question, provider_memory(memory, 'local'))
                    # Budget the planner too; long notes/questions must not be silently clipped.
                    if sum(len(m['content'].encode()) for m in request) > (settings.num_ctx - 1536) * 2:
                        raise ValueError('Память слишком большая для локального планировщика.')
                    raw = []
                    metrics = await self.local.generate(request, settings, lambda content, thinking: raw.append(content), planning=True)
                    turn['planner'] = {'messages': request, 'raw': ''.join(raw), 'metrics': metrics}
                    if metrics.get('done_reason') not in ('stop',):
                        raise ValueError('Планировщик не завершил ответ.')
                    plan = parse_plan(''.join(raw))
                else:
                    plan = {'question': question, 'queries': [question]}
                turn['plan'] = plan
                changed()
                phase('Локальный поиск и чтение MCP')
                retrieve_start = time.perf_counter()
                retrieved = await self.docs.retrieve(plan['queries'], settings, phase)
                turn.update(retrieved)
                turn['retrieval_seconds'] = time.perf_counter() - retrieve_start
                context = [{k: v for k, v in s.items() if k != 'pages'} for s in turn['sources']]
                context, generation_memory = fit_sources(plan['question'], memory, context, settings)
                for original, fitted in zip(turn['sources'], context):
                    if fitted.get('budget_truncated'):
                        original['read_text'] = original['text']
                        original.update(fitted)
                turn['provider_messages'], turn['provider_contexts'] = {}, {}
                for name in ('local', 'cloud'):
                    messages, budget = prepare_messages(plan['question'], provider_memory(generation_memory, name), context, settings)
                    turn['provider_messages'][name] = messages
                    turn['provider_contexts'][name] = budget
                # Keep legacy fields for old sessions/tools: these describe the local request.
                turn.update(messages=turn['provider_messages']['local'], context=turn['provider_contexts']['local'])
            else:
                previous = session['turns'][repeat]
                for name in ('sources', 'messages', 'context', 'provider_messages', 'provider_contexts',
                             'memory', 'plan', 'server', 'searches', 'candidates'):
                    if name in previous:
                        turn[name] = copy.deepcopy(previous[name])
                turn['retrieval_seconds'] = 0.0
                # Recheck budget with current model settings, without altering saved messages.
                requests = turn.get('provider_messages', {'local': turn['messages'], 'cloud': turn['messages']})
                if any(sum(len(m['content'].encode()) for m in request) > (settings.num_ctx - settings.num_predict - 1024) * 2
                       for name, request in requests.items() if name == 'local' or settings.cloud):
                    raise ValueError('Сохранённый контекст не помещается в текущие настройки.')
            changed()
            if not turn['sources']:
                for name in ('local', 'cloud'):
                    turn['results'][name].update(status='skipped', error='Поиск не нашёл материалов; генерация пропущена.')
                turn['status'] = 'no_sources'
                return turn
            phase('Генерация ответов' if settings.cloud else 'Локальная генерация')
            turn['results']['cloud']['status'] = 'pending' if settings.cloud else 'disabled'

            async def answer(name, client):
                result = turn['results'][name]
                result['status'] = 'running'
                changed()
                def update(content, thinking):
                    result['raw'] += content
                    result['thinking'] += thinking
                    changed()
                try:
                    messages = turn.get('provider_messages', {}).get(name, turn['messages'])
                    result['metrics'] = await client.generate(copy.deepcopy(messages), settings, update)
                    if result['metrics'].get('done_reason') != 'stop':
                        result.update(status='incomplete', error='Достигнут лимит или ответ не завершён.')
                    else:
                        data, checks = validate_answer(result['raw'], turn['sources'])
                        result['checks'] = checks
                        result['status'] = 'complete' if checks['passed'] else 'invalid'
                        if checks['passed']:
                            result['response'] = data
                except asyncio.CancelledError:
                    result['status'] = 'cancelled'
                    raise
                except Exception as error:
                    result.update(status='error', error=safe_error(error))
                finally:
                    changed()
            tasks = [answer('local', self.local)]
            if settings.cloud:
                tasks.append(answer('cloud', self.cloud))
            await asyncio.gather(*tasks)
            statuses = [r['status'] for r in turn['results'].values() if r['status'] != 'disabled']
            turn['status'] = 'complete' if all(s == 'complete' for s in statuses) else 'attention'
        except asyncio.CancelledError:
            turn['status'] = 'cancelled'
            for result in turn['results'].values():
                if result['status'] in ('running', 'pending'):
                    result['status'] = 'cancelled'
            raise
        except Exception as error:
            turn.update(status='error', error=safe_error(error))
        finally:
            turn['total_seconds'] = time.perf_counter() - start
            changed()
        return turn


def safe_error(error):
    # Known local validation messages are safe; provider/MCP exception groups may contain data.
    return str(error)[:400] if isinstance(error, ValueError) else f'{type(error).__name__}: ошибка этапа, подробности не сохраняются.'
