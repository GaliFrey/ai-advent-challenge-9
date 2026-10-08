"""Shared sequential comparison, staged tuning and held-out validation."""
import asyncio
import copy
import importlib.metadata
import platform

from cases import load_cases
from clients import DocsClient, LocalClient
from core import (BASELINE, STEPS, Settings, candidate, digest, frozen_input,
                  messages_for, new_report, save_report, summary, validate_answer)
from judge import Judge, decision
from resources_monitor import ResourceMonitor, gpu_memory


def safe_error(error):
    return str(error)[:400] if isinstance(error, ValueError) else f'{type(error).__name__}: ошибка этапа; чувствительные детали не сохранены.'


class Engine:
    def __init__(self, local=None, docs=None, judge=None, monitor_factory=ResourceMonitor,
                 notify=lambda text: None, changed=lambda: None, directory=None):
        self.local = local or LocalClient()
        self.docs = docs or DocsClient()
        self.judge = judge or Judge()
        self.monitor_factory = monitor_factory
        self.notify, self.changed = notify, changed
        self.directory = directory
        self.report = None
        self.lock = asyncio.Lock()

    def persist(self):
        if self.report:
            path = save_report(self.report, self.directory) if self.directory else save_report(self.report)
            self.changed()
            return path

    async def environment(self):
        return {'python': platform.python_version(), 'platform': platform.platform(),
                'dependencies': {name: importlib.metadata.version(name) for name in ('httpx', 'mcp', 'textual', 'python-dotenv')},
                'gpu': await gpu_memory(), 'resource_scope': 'local Ollama processes / all GPU devices',
                'protocol': 'sequential-unload-preload-generate',
                'limitations': ['Байтовый бюджет — оценка, не токенизатор.',
                                'Замеры не являются оценкой энергопотребления.',
                                'Примеры SP-XML не выполняются на сервере WebTutor.']}

    async def retrieve_manual(self, question, notes='', source='docs'):
        if not question.strip():
            raise ValueError('Введите вопрос.')
        settings = Settings(source=source).validate()
        retrieved = await self.docs.retrieve([question], settings, self.notify)
        sources = [{k: v for k, v in source.items() if k != 'pages'} for source in retrieved['sources']]
        return {'id': 'manual-' + digest(question)[:8], 'question': question, 'split': 'manual',
                'sources': sources, 'memory': {'notes': notes, 'user_questions': [], 'prior_answers': []},
                'criteria': {'facts': [], 'required': ['Ответ по переданным источникам.'], 'forbidden': ['Выдуманные API и факты.']},
                'retrieval': retrieved}

    async def run_one(self, case, profile, result=None):
        result = result if result is not None else {}
        result.update(profile=copy.deepcopy(profile), input_sha256=digest(frozen_input(case)),
                      raw='', thinking='', metrics={}, status='pending')
        self.changed()
        try:
            result['messages'] = messages_for(case, profile)
            result['messages_sha256'] = digest(result['messages'])
            if not case['sources']:
                result.update(status='no_sources', error='Нет источников; генерация пропущена.')
                return result
            settings = Settings(**profile['settings'])
            self.notify('Проверка модели: ' + profile['id'])
            result['model_metadata'] = await self.local.metadata(settings.local_model)
            self.persist()
            self.notify('Предварительная загрузка: ' + profile['id'])
            result['warmup'] = await self.local.warmup(settings)
            result['status'] = 'running'
            self.persist()
            self.notify('Генерация: ' + profile['id'])
            def update(content, thinking):
                result['raw'] += content
                result['thinking'] += thinking
                self.changed()
            monitor = self.monitor_factory()
            try:
                async with monitor:
                    result['metrics'] = await self.local.generate(copy.deepcopy(result['messages']), settings, update)
            finally:
                resources = monitor.result()
                result['resources'] = resources
                result['metrics'].update({key: resources[key] for key in ('max_ram_bytes', 'max_gpu_bytes')})
            data, checks = validate_answer(result['raw'], case['sources'])
            result['checks'] = checks
            if result['metrics'].get('done_reason') != 'stop':
                result.update(status='incomplete', error='Ответ не завершён; достигнут лимит либо другая причина остановки.')
            elif not checks['passed']:
                result['status'] = 'invalid'
            else:
                result.update(status='complete', response=data)
            # An unexpectedly high measured prompt count invalidates the estimate.
            actual = result['metrics'].get('input_tokens')
            if isinstance(actual, int) and actual + settings.num_predict + 1024 > settings.num_ctx:
                result.update(status='context_risk', error='Фактическое число входных токенов превышает резерв контекста; профиль непригоден.')
        except asyncio.CancelledError:
            result['status'] = 'cancelled'
            raise
        except Exception as error:
            result.update(status='context_limit' if isinstance(error, ValueError) and 'оценочный бюджет' in str(error) else 'error',
                          error=safe_error(error))
        finally:
            self.persist()
        return result

    async def compare(self, case, before, after, *, repeat=0, swap=False, cached_before=None):
        comparison = {'case_id': case['id'], 'repeat': repeat, 'order': ['after', 'before'] if swap else ['before', 'after'],
                      'input_sha256': digest(frozen_input(case)), 'before': {}, 'after': {}, 'status': 'running'}
        self.report['comparisons'].append(comparison)
        if not any(c['id'] == case['id'] for c in self.report['cases']):
            self.report['cases'].append(copy.deepcopy(case))
        self.persist()
        try:
            for side in comparison['order']:
                if side == 'before' and cached_before is not None:
                    comparison['before'] = copy.deepcopy(cached_before)
                    comparison['before']['reused_screening_result'] = True
                else:
                    await self.run_one(case, before if side == 'before' else after, comparison[side])
            if all(comparison[side].get('raw') and comparison[side]['status'] in ('complete', 'invalid', 'incomplete')
                   for side in ('before', 'after')):
                self.notify('DeepSeek: обезличенная оценка в двух порядках')
                comparison['judge'] = await self.judge.compare(case, comparison['before'], comparison['after'])
                comparison['status'] = 'complete'
            else:
                comparison.update(status='attention', judge={'status': 'skipped', 'reason': 'Ошибка локальной генерации/проверки.'})
            comparison['decision'] = decision([comparison])
            return comparison
        except asyncio.CancelledError:
            comparison['status'] = 'cancelled'
            raise
        except Exception as error:
            comparison.update(status='error', error=safe_error(error))
            raise
        finally:
            self.persist()

    async def manual(self, case, before, after):
        async with self.lock:
            self.report = new_report('manual')
            try:
                self.report['environment'] = await self.environment()
                await self.compare(case, before, after)
                self.report['status'] = 'complete'
            except asyncio.CancelledError:
                self.report['status'] = 'cancelled'
                raise
            except Exception as error:
                self.report.update(status='error', error=safe_error(error))
            finally:
                self.persist()
            return self.report

    async def experiment(self, report=None):
        async with self.lock:
            self.report = report or new_report('experiment')
            if self.report['kind'] != 'experiment':
                raise ValueError('Продолжить можно только пакетный эксперимент.')
            try:
                self.report['status'] = 'running'
                if not self.report['environment']:
                    self.report['environment'] = await self.environment()
                if not self.judge.available:
                    self.report.update(status='manual_review', error='Для автоматического выбора нужен DEEPSEEK_API_KEY. Ручные сравнения доступны без ключа.')
                    return self.report
                if not self.report.get('dataset_frozen'):
                    saved = {case['id']: case for case in self.report['cases']}
                    dataset = [dict(case, split=split) for split in ('tune', 'holdout') for case in load_cases(split)]
                    self.report['cases'] = [copy.deepcopy(saved.get(case['id'], case)) for case in dataset]
                    self.report['dataset_frozen'] = True
                    self.report['dataset_sha256'] = digest(self.report['cases'])
                if digest(self.report['cases']) != self.report['dataset_sha256']:
                    raise ValueError('Сохранённый набор случаев изменён; начните новый эксперимент.')
                tune = [case for case in self.report['cases'] if case['split'] == 'tune']
                holdout = [case for case in self.report['cases'] if case['split'] == 'holdout']
                current = copy.deepcopy(BASELINE)
                cache = self.report.setdefault('screening_results', {})
                base_key = digest(BASELINE)
                if base_key not in cache:
                    cache[base_key] = {}
                for case in tune:
                    previous = cache[base_key].get(case['id'])
                    if previous and previous.get('status') not in ('pending', 'running', 'cancelled'):
                        if previous['input_sha256'] != digest(frozen_input(case)):
                            raise ValueError('Сохранённый базовый ответ относится к другому входу; начните новый эксперимент.')
                        continue
                    cache[base_key][case['id']] = {}
                    await self.run_one(case, BASELINE, cache[base_key][case['id']])
                    if cache[base_key][case['id']]['status'] not in ('complete', 'invalid', 'incomplete'):
                        self.report.update(status='error', error='Базовый профиль не выполнил запрос; эксперимент остановлен.')
                        return self.report
                for index, step in enumerate(STEPS):
                    proposed = candidate(current, step)
                    existing = self.report['steps'][index] if index < len(self.report['steps']) else None
                    if existing and digest(existing['profile']) != digest(proposed):
                        raise ValueError('Ручная оценка изменила ранее выбранный профиль после следующих этапов; начните новый эксперимент.')
                    if existing is None:
                        existing = {'index': index, 'profile': copy.deepcopy(proposed), 'comparisons': [], 'status': 'running'}
                        self.report['steps'].append(existing)
                    existing_comparisons = [self.report['comparisons'][i] for i in existing['comparisons']]
                    done_ids = {c['case_id'] for c in existing_comparisons if c['status'] not in ('running', 'cancelled')}
                    for case in tune:
                        if case['id'] in done_ids:
                            continue
                        self.notify(f'Настройка {index+1}/6: {proposed["id"]}, {case["id"]}')
                        old = cache[digest(current)][case['id']]
                        comparison_index = len(self.report['comparisons'])
                        existing['comparisons'].append(comparison_index)
                        await self.compare(case, current, proposed, cached_before=old)
                    comparisons = [self.report['comparisons'][i] for i in existing['comparisons']
                                   if self.report['comparisons'][i]['status'] not in ('cancelled', 'running')]
                    existing['decision'] = decision(comparisons)
                    existing['status'] = existing['decision']['status']
                    self.persist()
                    if existing['status'] == 'manual_review':
                        self.report['status'] = 'manual_review'
                        return self.report
                    if existing['decision']['accepted']:
                        current = proposed
                        cache[digest(current)] = {c['case_id']: copy.deepcopy(c['after']) for c in comparisons}
                self.report['selected_profile'] = copy.deepcopy(current)
                final = self.report.get('final')
                if final is None:
                    final = self.report['final'] = {'comparisons': [], 'profile': copy.deepcopy(current)}
                completed = {(self.report['comparisons'][i]['case_id'], self.report['comparisons'][i]['repeat'])
                             for i in final['comparisons'] if self.report['comparisons'][i]['status'] not in ('cancelled', 'running')}
                for repeat in range(3):
                    for case_index, case in enumerate(holdout):
                        if (case['id'], repeat) in completed:
                            continue
                        self.notify(f'Итог {repeat+1}/3: {case["id"]}')
                        final['comparisons'].append(len(self.report['comparisons']))
                        await self.compare(case, BASELINE, current, repeat=repeat, swap=bool((repeat+case_index) % 2))
                comparisons = [self.report['comparisons'][i] for i in final['comparisons']
                               if self.report['comparisons'][i]['status'] not in ('cancelled', 'running')]
                final['decision'] = decision(comparisons)
                if digest(current) == digest(BASELINE):
                    final['decision'] = {'accepted': False, 'status': 'no_change',
                        'reason': 'Ни одно изменение не принято; итоговые повторы проверяют базу. Оптимизация не подтверждена.',
                        'before': summary([c['before'] for c in comparisons]),
                        'after': summary([c['after'] for c in comparisons])}
                self.report['winner'] = copy.deepcopy(current if final['decision']['accepted'] else BASELINE)
                self.report['status'] = 'manual_review' if final['decision']['status'] == 'manual_review' else 'complete'
            except asyncio.CancelledError:
                self.report['status'] = 'cancelled'
                raise
            except Exception as error:
                self.report.update(status='error', error=safe_error(error))
            finally:
                self.persist()
            return self.report
