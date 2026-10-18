"""Persistent action drafts. Model extracts values; code owns transitions."""
import json
import re
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
from .storage import dump
from .profile import timezone_name, zone_label, load_profile
from .events import EventData, Events
from .recurrence import RuleDraft, Rule, occurrence, rule_text, local_timestamp

ACTIONS = {
    'create_event': ('data',),
    'change_event': ('id','operation','scope'),
    'set_event_rule': ('id','rule','scope'),
    'create_note': ('content',),
    'create_reminder': ('content', 'due_at'),
    'update_note': ('id', 'content'),
    'delete_note': ('id',),
    'update_reminder': ('id', 'content', 'due_at'),
    'cancel_reminder': ('id',),
    'create_schedule': ('content', 'recurrence', 'due_time'),
    'update_schedule': ('id', 'content', 'recurrence', 'due_time'),
    'cancel_schedule': ('id',),
    'skip_schedule': ('id',),
}
ALLOWED = {
    'create_event': {'data'},
    'change_event': {'id','operation','scope','occurrence_id','from_position','data','rule_id'},
    'set_event_rule': {'id','rule','scope','occurrence_id','from_position','replace_rule_id'},
    'create_note': {'content', 'event_date'},
    'create_reminder': {'content', 'due_at', 'due_date', 'due_time','after_seconds','time_mode','timezone','local_date'},
    'update_note': {'id', 'content', 'event_date'},
    'delete_note': {'id'},
    'update_reminder': {'id', 'content', 'due_at', 'due_date', 'due_time','time_mode','timezone','local_date'},
    'cancel_reminder': {'id'},
    'create_schedule': {'content', 'recurrence', 'due_time', 'timezone', 'time_mode'},
    'update_schedule': {'id', 'content', 'recurrence', 'due_time', 'timezone', 'time_mode'},
    'cancel_schedule': {'id'},
    'skip_schedule': {'id'},
}
REMINDER_TIME_FIELDS = {'due_at', 'due_date', 'due_time', 'time_mode', 'local_date', 'timezone'}


def same_snapshot(expected, actual):
    # Additive migrations do not invalidate saved pre-migration requests.
    if expected is None or actual is None:
        return expected is actual
    return all(key in actual and actual[key]==value for key,value in expected.items())


def record_type(action):
    return 'event' if action in ('create_event','change_event','set_event_rule') else 'schedule' if action.endswith('schedule') else 'note' if action.endswith('note') else 'reminder'


def table_for(action):
    return {'event':'events', 'schedule': 'schedules', 'note': 'notes', 'reminder': 'reminders'}[record_type(action)]


class RequestError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details


def error(code, message, request=None):
    result = {'error': {'code': code, 'message': message}}
    if request is not None:
        result['request'] = request
    return result


class Requests:
    def __init__(self, store, models=None, execute=None):
        self.store, self.models, self.apply = store, models, execute

    def active(self):
        row = self.store.db.execute("SELECT * FROM requests WHERE session=? AND status IN ('waiting','ready','executing') ORDER BY id DESC LIMIT 1",
                                    (self.store.get_meta('session', 0),)).fetchone()
        return dict(row) if row else None

    def view(self, row):
        return {'request_id': row['id'], 'action': row['action'], 'status': row['status'],
                'parameters': json.loads(row['parameters']), 'missing_fields': json.loads(row['missing']),
                'source_time': json.loads(row['origin']).get('date'),
                'error': json.loads(row['error']) if row['error'] else None}

    def pending(self, row=None):
        row = row or self.active()
        if row is None:
            return None
        return {'request_id': row['id'], 'kind': 'note' if row['action'].endswith('note') else 'reminder',
                'origin': json.loads(row['origin']), 'question': self.question(json.loads(row['missing']), row['action'])}

    @staticmethod
    def question(missing, action):
        questions = {'id': 'Какую запись нужно изменить или удалить?', 'content': ('О чём напомнить?' if record_type(action) != 'note' else 'Какой текст нужно сохранить?'),
                     'due_at': 'На какую дату и время поставить напоминание?',
                     'due_date': 'На какую дату поставить напоминание?', 'due_time': 'Во сколько напомнить?',
                     'changes': 'Что именно нужно изменить?',
                     'recurrence': 'Как часто повторять напоминание?',
                     'weekdays': 'В какие дни недели напоминать?',
                     'days': 'По каким числам месяца напоминать?',
                     'dates': 'В какие даты года напоминать?'}
        questions.update(data='Какое событие сохранить?', operation='Что нужно сделать с событием?', scope='Один экземпляр или будущую серию?', rule='Когда напоминать?', occurrence_id='Выберите экземпляр в календаре.', from_position='С какого экземпляра менять серию?', **{'data.content':'Как назвать событие?', 'data.start_date':'На какую дату событие?', 'data.start_time':'Во сколько событие, либо оно на весь день?'})
        questions['rule_id'] = 'Какое правило напоминания отключить?'
        return ' '.join(questions[key] for key in missing) if missing else 'Поручение готово к выполнению.'

    def _target(self, action, values):
        if not action.startswith(('update_', 'delete_', 'cancel_', 'skip_', 'change_', 'set_')) or not values.get('id'):
            return None
        table = table_for(action)
        row = self.store.db.execute(f'SELECT * FROM {table} WHERE id=?', (values['id'],)).fetchone()
        if row is None or (table in ('notes','events') and row['deleted']) or (table == 'reminders' and row['status'] not in ('pending', 'queued')) or (table == 'schedules' and (row['kind']!='recurring' or not (row['status']=='active' or row['status']=='completed' and action in ('cancel_schedule','skip_schedule') and self.store.db.execute("SELECT 1 FROM reminders WHERE schedule_id=? AND status IN ('pending','queued')",(row['id'],)).fetchone()))):
            raise RequestError({'events':'event_not_found', 'notes': 'note_not_found', 'reminders': 'reminder_not_active', 'schedules': 'schedule_not_active'}[table],
                               'Выбранная запись отсутствует, удалена или уже не активна.')
        return dict(row)

    def _validate(self, action, values, now, provided=()):
        missing = [key for key in ACTIONS[action] if values.get(key) is None or values.get(key) == '']
        if action in ('create_reminder', 'update_reminder') and 'due_at' in missing:
            missing.remove('due_at')
            missing.extend(key for key in ('due_date', 'due_time') if not values.get(key))
        if action.startswith('update_') and not (set(provided) & (ALLOWED[action] - {'id'})):
            missing.append('changes')
        if action in ('create_schedule', 'update_schedule'):
            rule = values.get('recurrence') or {}
            if not rule.get('frequency'):
                if 'recurrence' not in missing:
                    missing.append('recurrence')
            else:
                key = {'weekly': 'weekdays', 'monthly': 'days', 'yearly': 'dates'}.get(rule['frequency'])
                if key and not rule.get(key) and not (rule['frequency']=='monthly' and rule.get('ordinal_weekdays')):
                    missing.append(key)
        if action in ('create_event','change_event','set_event_rule'):
            if action == 'change_event' and values.get('operation') == 'disable_rule' and not values.get('rule_id'):
                missing.append('rule_id')
            if action=='create_event' or values.get('operation')=='move':
                data=values.get('data') or {}
                if data.get('kind') == 'birthday' and data.get('person') and not data.get('content'):
                    data['content'] = 'День рождения: ' + data['person']
                if 'data' in missing:
                    missing.remove('data')
                missing.extend('data.'+key for key in ('content','start_date') if not data.get(key))
                if not data.get('all_day') and data.get('kind')!='birthday' and not data.get('start_time'):
                    missing.append('data.start_time')
            if action!='create_event' and values.get('scope'):
                key='occurrence_id' if values['scope']=='one' else 'from_position'
                if values.get(key) is None:
                    missing.append(key)
        issues = []
        if action == 'update_reminder' and set(provided) & REMINDER_TIME_FIELDS and values.get('time_mode') == 'default' and values.get('local_date') and values.get('timezone'):
            zone = ZoneInfo(values['timezone'])
            due = local_timestamp(date.fromisoformat(values['local_date']), time.fromisoformat(load_profile(self.store).notification_time), zone)
            values['due_at'] = datetime.fromtimestamp(due, zone).isoformat()
        if action=='update_reminder' and set(provided) & REMINDER_TIME_FIELDS and values.get('due_at') is not None and datetime.fromisoformat(values['due_at']).timestamp() <= now:
            if 'due_at' not in missing:
                missing.append('due_at')
            issues.append({'code': 'past_due', 'message': 'Указанный срок уже прошёл. Нужны новые дата и время.'})
        if not missing:
            # Reuse the actual action contract (including its shorter reminder limit).
            self.models[action].model_validate(self.action_values(values, action))
            if action=='create_schedule':
                at = load_profile(self.store).notification_time if values.get('time_mode') == 'default' else values['due_time']
                if occurrence(Rule.model_validate(values['recurrence']), at, values['timezone'], now) is None:
                    raise RequestError('schedule_out_of_range', 'Не удалось вычислить следующий срок расписания.')
        return missing, issues

    @staticmethod
    def action_values(values, action=None):
        excluded = ('due_date',) if action in ('create_schedule', 'update_schedule') else ('due_date', 'due_time', 'after_seconds')
        return {key: value for key, value in values.items() if key not in excluded}

    def schedule(self, values, patch):
        zone = ZoneInfo(values.get('timezone') or timezone_name(self.store))
        if not {'due_at', 'due_date', 'due_time'} & patch.keys() and values.get('due_at'):
            local = datetime.fromisoformat(values['due_at']).astimezone(zone)
            values.update(due_date=local.date().isoformat(), due_time=local.time().isoformat())
            return
        if 'due_at' in patch and ('due_date' in patch or 'due_time' in patch):
            raise RequestError('conflicting_schedule', 'Передай либо due_at, либо due_date/due_time, без противоречащих представлений срока.')
        if 'due_at' in patch:
            if patch['due_at'] is not None:
                local = datetime.fromisoformat(patch['due_at']).astimezone(zone)
                values.update(due_date=local.date().isoformat(), due_time=local.time().isoformat())
                return
            else:
                values.update(due_date=None, due_time=None)
        elif 'due_date' in patch or 'due_time' in patch:
            if values.get('due_at'):
                previous = datetime.fromisoformat(values['due_at']).astimezone(zone)
                values.setdefault('due_date', previous.date().isoformat())
                values.setdefault('due_time', previous.time().isoformat())
            values.pop('due_at', None)
        if values.get('due_date') and (values.get('due_time') or values.get('time_mode') == 'default'):
            at = values.get('due_time') or load_profile(self.store).notification_time
            naive = datetime.fromisoformat(values['due_date'] + 'T' + at)
            candidates = [naive.replace(tzinfo=zone, fold=fold) for fold in (0, 1)]
            valid = [candidate for candidate in candidates if candidate.astimezone(ZoneInfo('UTC')).astimezone(zone).replace(tzinfo=None) == naive]
            if values.get('time_mode')=='default':
                values['due_at']=datetime.fromtimestamp(local_timestamp(naive.date(),naive.time(),zone),zone).isoformat()
                return
            if not valid:
                raise RequestError('nonexistent_local_time', 'Такого местного времени нет из-за перевода часов. Укажите другое время.')
            if len({candidate.utcoffset() for candidate in valid}) > 1:
                raise RequestError('ambiguous_local_time', 'Это время повторяется при переводе часов. Уточните смещение UTC и передайте полный due_at.')
            values['due_at'] = valid[0].isoformat()
        elif values.get('due_at') and 'due_date' not in values:
            local = datetime.fromisoformat(values['due_at']).astimezone(zone)
            values.update(due_date=local.date().isoformat(), due_time=local.time().isoformat())

    def prepare(self, action, patch, job, now, continued=False):
        with self.store.atomic():
            current = self.active()
            selection = (job.get('pending') or {}).get('selection')
            message = json.loads(job['payload'])['message']
            numbered = bool(re.fullmatch(r'(?:вариант\s*)?\d+[.)]?', message.get('text', '').strip(), re.I))
            if selection and numbered and current:
                if action is not None and action != current['action']:
                    raise RequestError('selection_mismatch', 'Ответ номером выбирает существующее поручение, не создаёт новое.')
                continued = True
            if continued:
                if current is None:
                    raise RequestError('no_active_request', 'Нет незавершённого поручения. Создай новую просьбу.')
                if action is not None and action != current['action']:
                    raise RequestError('request_action_mismatch', 'При уточнении нельзя менять действие. Создай отдельное поручение.')
                action = current['action']
                values = json.loads(current['parameters'])
                provided = set(json.loads(current['provided']))
                origin = json.loads(current['origin'])
            else:
                if action not in ACTIONS:
                    raise RequestError('action_required', 'Для нового поручения нужно указать действие.')
                values, provided = {}, set()
                origin = json.loads(job['payload'])['message']
            if set(patch) - ALLOWED[action]:
                raise RequestError('invalid_request_fields', 'Эти поля не относятся к выбранному действию.', sorted(set(patch) - ALLOWED[action]))
            selection = selection if continued or numbered else None
            if selection:
                if action.startswith('create_'):
                    raise RequestError('selection_mismatch', 'Номер варианта не разрешает создавать новую запись.')
                kind = record_type(action)
                if kind != selection['record_type'] or (patch.get('id') is not None and patch['id'] != selection['id']):
                    raise RequestError('selection_mismatch', 'Используй выбранную пользователем запись.')
                patch = {**patch, 'id': selection['id']}
            previous_rule = values.get('recurrence')
            previous_due_at = values.get('due_at') if 'due_at' in provided else None
            if patch.get('data') is not None:
                patch = {**patch,'data':{**(values.get('data') or {}),**patch['data']}}
            values.update(patch)
            provided.update(patch)
            target = self._target(action, values)
            if action == 'update_reminder':
                calendar_patch = {'due_date', 'due_time'} & patch.keys()
                if 'due_at' in patch and not calendar_patch:
                    provided.difference_update({'due_date', 'due_time', 'local_date'})
                elif calendar_patch and 'due_at' not in patch:
                    if previous_due_at:
                        zone = values.get('timezone') or timezone_name(self.store)
                        if target and 'timezone' not in provided:
                            parent = self.store.db.execute('SELECT * FROM schedules WHERE id=?',
                                (target['standalone_id'] or target['schedule_id'],)).fetchone()
                            if parent and (parent['kind'] == 'recurring' or parent['local_date']):
                                zone = parent['timezone']
                        local = datetime.fromisoformat(previous_due_at).astimezone(ZoneInfo(zone))
                        for key, value in (('due_date', local.date().isoformat()), ('due_time', local.time().isoformat())):
                            if key not in patch:
                                values[key] = value
                                provided.add(key)
                    provided.discard('due_at')
            if action == 'update_reminder' and target and continued and not current['target']:
                # Before selection, calendar defaults came from the profile.
                # Replay only supplied fields against the selected source.
                explicit_time = {key: values[key] for key in provided & REMINDER_TIME_FIELDS if key in values}
                for key in REMINDER_TIME_FIELDS - provided:
                    values.pop(key, None)
                patch = {**patch, **explicit_time}
            if action == 'update_schedule' and target and continued and not current['target']:
                default_clock = 'due_time' in provided and 'due_time' not in patch and values.get('time_mode') == 'default'
                for key in ('due_time', 'time_mode', 'timezone'):
                    if key not in provided:
                        values.pop(key, None)
                if default_clock:
                    values['due_time'] = None
            if target:
                # Do not silently refresh an earlier target snapshot during a follow-up.
                if continued and current['target'] and not same_snapshot(json.loads(current['target']),target):
                    raise RequestError('request_stale', 'Запись изменилась после начала поручения. Прочитай её заново и начни новую просьбу.')
                for key in ('content', 'due_at', 'event_date', 'due_time', 'timezone'):
                    if key in ALLOWED[action] and key not in values:
                        value = target.get(key)
                        if value is None and key in ('due_time', 'timezone'):
                            continue
                        values[key] = datetime.fromtimestamp(value, ZoneInfo(timezone_name(self.store))).isoformat() if key == 'due_at' and value is not None else value
            if action in ('create_schedule', 'update_schedule'):
                if not values.get('due_time'):
                    values['due_time']=load_profile(self.store).notification_time
                    values['time_mode']='default'
                elif 'due_time' in patch:
                    values['time_mode']='explicit'
                elif target:
                    values['time_mode']=target['time_mode']
                deferred_target = action == 'update_schedule' and not target
                if not values.get('timezone') and not deferred_target:
                    values['timezone'] = timezone_name(self.store)
                base = previous_rule or (json.loads(target['rule']) if target else {})
                if target and continued and not current['target'] and previous_rule:
                    # Before an ambiguous target is selected, keep only explicit
                    # changes; defaults must not replace that schedule's rule.
                    base = json.loads(target['rule'])
                    if previous_rule.get('frequency') and previous_rule['frequency'] != base['frequency']:
                        base = {k: v for k, v in base.items() if k not in ('weekdays', 'days', 'dates', 'ordinal_weekdays', 'interval')}
                    if previous_rule.get('count') is not None and 'until' not in previous_rule:
                        base.pop('until', None)
                    if previous_rule.get('until') is not None and 'count' not in previous_rule:
                        base.pop('count', None)
                    base = {**base, **previous_rule}
                patch_rule = patch.get('recurrence')
                if patch_rule is not None:
                    if patch_rule.get('frequency') and patch_rule['frequency'] != base.get('frequency'):
                        base = {k: v for k, v in base.items() if k not in ('weekdays', 'days', 'dates', 'ordinal_weekdays', 'interval')}
                    if patch_rule.get('count') is not None and 'until' not in patch_rule:
                        base.pop('until', None)
                    if patch_rule.get('until') is not None and 'count' not in patch_rule:
                        base.pop('count', None)
                    base = {**base, **patch_rule}
                elif 'recurrence' in patch:
                    base = {}
                if deferred_target:
                    if base:
                        values['recurrence'] = RuleDraft.model_validate(base).model_dump(mode='json', exclude_unset=True)
                    else:
                        values.pop('recurrence', None)
                else:
                    rule = RuleDraft.model_validate(base).model_dump(mode='json')
                    rule['start_date'] = rule.get('start_date') or datetime.fromtimestamp(now, ZoneInfo(values['timezone'])).date().isoformat()
                    values['recurrence'] = rule
            if action in ('create_reminder', 'update_reminder'):
                if action == 'update_reminder':
                    if target and (target['standalone_id'] or target['schedule_id']):
                        parent = self.store.db.execute('SELECT * FROM schedules WHERE id=?', (target['standalone_id'] or target['schedule_id'],)).fetchone()
                        if parent['kind'] == 'recurring' or parent['local_date']:
                            if not values.get('timezone'):
                                values['timezone'] = parent['timezone']
                        if target['standalone_id']:
                            values.setdefault('time_mode', parent['time_mode'])
                if 'due_time' in patch and patch['due_time'] is None and 'due_at' not in patch:
                    if values.get('due_at') and not values.get('due_date'):
                        zone = ZoneInfo(values.get('timezone') or timezone_name(self.store))
                        values['due_date'] = datetime.fromisoformat(values['due_at']).astimezone(zone).date().isoformat()
                    values.pop('due_at', None)
                if action=='create_reminder' and patch.get('after_seconds') is not None:
                    if any(patch.get(k) is not None for k in ('due_at','due_date','due_time')):
                        raise RequestError('conflicting_schedule','Укажите интервал от сообщения либо календарный срок.')
                    if not isinstance(origin.get('date'),(int,float)):
                        raise RequestError('message_time_missing','Неизвестно время исходного сообщения.')
                    values['due_at']=datetime.fromtimestamp(origin['date']+patch['after_seconds'],ZoneInfo(timezone_name(self.store))).isoformat()
                    values['time_mode'] = 'explicit'
                    for key in ('due_date', 'due_time', 'local_date'):
                        values.pop(key, None)
                if values.get('due_date') and not values.get('due_time') and not values.get('due_at'):
                    if action != 'update_reminder' or 'due_time' not in provided:
                        values['due_time'] = load_profile(self.store).notification_time
                    values.update(time_mode='default',timezone=values.get('timezone') or timezone_name(self.store),local_date=values['due_date'])
                elif 'due_time' in patch or 'due_at' in patch:
                    values['time_mode']='explicit'
                self.schedule(values, patch)
                if values.get('time_mode') == 'default' and provided & REMINDER_TIME_FIELDS:
                    values['local_date'] = values.get('due_date')
            missing, issues = self._validate(action, values, now, provided)
            status = 'waiting' if missing else 'ready'
            if continued:
                identity = current['id']
                self.store.db.execute('UPDATE requests SET parameters=?,status=?,missing=?,target=?,error=?,updated_at=?,provided=? WHERE id=?',
                    (dump(values), status, dump(missing), dump(target) if target else None,
                     dump(issues) if issues else None, now, dump(sorted(provided)), identity))
            else:
                if current:
                    self._cancel(current, now, 'superseded')
                cur = self.store.db.execute('INSERT INTO requests(session,action,parameters,origin,status,missing,target,error,created_at,updated_at,provided) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (self.store.get_meta('session', 0), action, dump(values), dump(origin), status, dump(missing),
                     dump(target) if target else None, dump(issues) if issues else None, now, now, dump(sorted(provided))))
                identity = cur.lastrowid
            row = dict(self.store.db.execute('SELECT * FROM requests WHERE id=?', (identity,)).fetchone())
            self.store.set_meta('pending', self.pending(row))
            result = self.view(row)
            if missing:
                result.update(awaiting_answer=True, question=self.question(missing, row['action']), pending=self.pending(row))
            return result

    def describe(self):
        row = self.active()
        if row is None:
            return 'Незавершённых поручений нет.'
        values = json.loads(row['parameters'])
        names = {'create_note': 'сохранить памятку', 'create_reminder': 'создать напоминание',
                 'update_note': 'исправить памятку', 'delete_note': 'удалить памятку',
                 'update_reminder': 'изменить напоминание', 'cancel_reminder': 'отменить напоминание'}
        names.update(create_event='создать событие',change_event='изменить событие',set_event_rule='настроить уведомление события')
        names.update(create_schedule='создать повторяющееся напоминание', update_schedule='изменить повторяющееся напоминание',
                     cancel_schedule='отменить повторяющееся напоминание', skip_schedule='пропустить срабатывание')
        parts = ['Поручение: ' + names[row['action']] + '.']
        if values.get('content'):
            parts.append('Текст: «' + values['content'][:1000] + '».')
        if values.get('due_date'):
            parts.append('Дата: ' + values['due_date'] + '.')
        if values.get('due_time'):
            parts.append('Время: ' + values['due_time'] + ' (' + zone_label(values.get('timezone') or timezone_name(self.store)) + ').')
        rule = values.get('recurrence')
        if rule and rule.get('frequency'):
            try:
                parts.append('Повтор: ' + rule_text(Rule.model_validate(rule)) + '.')
            except ValueError:
                parts.append('Повтор: ' + {'daily': 'каждый день', 'weekly': 'по дням недели', 'monthly': 'по числам месяца', 'yearly': 'по датам года'}[rule['frequency']] + '.')
        parts.append(self.question(json.loads(row['missing']), row['action']))
        return '\n'.join(parts)

    def _cancel(self, row, now, reason='cancelled_by_user'):
        self.store.db.execute("UPDATE requests SET status='cancelled',error=?,updated_at=? WHERE id=?",
                              (dump({'code': reason}), now, row['id']))
        pending = self.store.get_meta('pending')
        if pending and pending.get('request_id') == row['id']:
            self.store.set_meta('pending', None)

    def cancel(self, now):
        with self.store.atomic():
            row = self.active()
            if row is None:
                if self.store.get_meta('pending'):
                    self.store.set_meta('pending', None)
                    return {'status': 'cancelled', 'legacy_clarification': True}
                return error('no_active_request', 'Нет незавершённого поручения.')
            self._cancel(row, now)
            return {'request_id': row['id'], 'status': 'cancelled'}

    def _receipt(self, job, row, result, now, replayed=False):
        if replayed:
            previous = self.store.db.execute("SELECT payload FROM action_receipts WHERE request_id=? AND outcome='done' ORDER BY id LIMIT 1", (row['id'],)).fetchone()
            if previous:
                snapshot = json.loads(previous[0])
            else:
                # Historical completed requests predate receipts. Their saved
                # parameters/target/result prove the past operation, not today's state.
                snapshot = {**(json.loads(row['target']) if row['target'] else {}),
                            **result, **json.loads(row['parameters'])}
        elif record_type(row['action'])=='event':
            event_result=result.get('event',result)
            identity=event_result.get('id') or json.loads(row['parameters']).get('id')
            saved=self.store.db.execute('SELECT * FROM events WHERE id=?',(identity,)).fetchone()
            if saved is None:
                raise ValueError('Нет сохранённого события')
            if row['action'] in ('change_event','set_event_rule') and saved['revision']!=json.loads(row['target'])['revision']+1:
                raise ValueError('Изменение события не подтверждено новой ревизией')
            if row['action']=='change_event' and result.get('operation')=='delete':
                if not saved['deleted']:
                    raise ValueError('Событие не удалено')
                snapshot={**result,'content':saved['content']}
            else:
                actual=Events(self.store).view(identity)
                if any(actual.get(k)!=v for k,v in event_result.items() if k not in ('warnings',)):
                    raise ValueError('Результат не совпадает с сохранённым событием')
                if row['action'] == 'change_event' and result.get('operation') == 'disable_rule':
                    parameters = json.loads(row['parameters'])
                    rule = self.store.db.execute('SELECT * FROM event_reminder_rules WHERE id=? AND event_id=?',
                        (parameters['rule_id'], identity)).fetchone()
                    if rule is None or dict(rule) != result.get('disabled_rule'):
                        raise ValueError('Отключение выбранного правила не подтверждено')
                    if parameters['scope'] == 'one':
                        disabled = self.store.db.execute('SELECT 1 FROM event_rule_exceptions WHERE occurrence_id=? AND rule_id=?',
                            (parameters['occurrence_id'], rule['id'])).fetchone() is not None
                    else:
                        disabled = rule['to_position'] is not None and rule['to_position'] <= max(rule['from_position'], parameters['from_position'])
                    if not disabled:
                        raise ValueError('Правило осталось действующим на выбранном участке')
                snapshot={**result,'event':actual}
                if row['action']=='create_event':
                    snapshot={**actual,**({'warnings':result['warnings']} if result.get('warnings') else {})}
        elif row['action'].startswith(('create_', 'update_', 'skip_')):
            table = table_for(row['action'])
            saved = self.store.db.execute(f'SELECT * FROM {table} WHERE id=?', (result['id'],)).fetchone()
            if saved is None:
                raise ValueError('Нет сохранённого результата действия')
            snapshot = dict(saved)
            if table == 'reminders' and saved['schedule_id'] is not None:
                schedule = self.store.db.execute('SELECT timezone FROM schedules WHERE id=?', (saved['schedule_id'],)).fetchone()
                snapshot['timezone'] = schedule['timezone']
            if table=='reminders' and saved['standalone_id']:
                parent=self.store.db.execute('SELECT timezone,local_date FROM schedules WHERE id=?',(saved['standalone_id'],)).fetchone()
                if parent['local_date']:
                    snapshot['timezone']=parent['timezone']
            if table=='schedules' and saved['time_mode']=='default':
                snapshot['due_time']=load_profile(self.store).notification_time
            if row['action'] == 'skip_schedule':
                snapshot['skipped_due_at'] = result['skipped_due_at']
        else:
            snapshot = json.loads(row['target'])
            table = table_for(row['action'])
            saved = self.store.db.execute(f'SELECT * FROM {table} WHERE id=?', (snapshot['id'],)).fetchone()
            if saved is None or (table == 'notes' and not saved['deleted']) or (table in ('reminders', 'schedules') and saved['status'] != 'cancelled'):
                raise ValueError('Состояние записи не подтверждает удаление или отмену')
        self.store.record_receipt(job, row['action'], 'done', snapshot, now, row['id'], replayed)

    def execute(self, identity, job, now):
        with self.store.atomic():
            found = self.store.db.execute('SELECT * FROM requests WHERE id=?', (identity,)).fetchone()
            if found is None or found['session'] != self.store.get_meta('session', 0):
                return error('request_not_found', 'Поручение не найдено в текущей беседе.')
            row = dict(found)
            if row['status'] == 'done':
                saved_result = json.loads(row['result'])
                self._receipt(job, row, saved_result, now, replayed=True)
                return {'request_id': identity, 'status': 'done', 'result': saved_result}
            if row['status'] not in ('waiting', 'ready'):
                return error('request_closed', 'Поручение уже отменено или завершилось ошибкой.')
            values = json.loads(row['parameters'])
            selection = (job.get('pending') or {}).get('selection')
            if selection and not row['action'].startswith('create_'):
                kind = record_type(row['action'])
                if (kind, values.get('id')) != (selection['record_type'], selection['id']):
                    return error('selection_mismatch', 'Поручение относится к другой записи. Используй выбранную пользователем запись.')
            missing, issues = self._validate(row['action'], values, now, json.loads(row['provided']))
            if missing:
                self.store.db.execute("UPDATE requests SET status='waiting',missing=?,error=?,updated_at=? WHERE id=?",
                    (dump(missing), dump(issues) if issues else None, now, identity))
                row = dict(self.store.db.execute('SELECT * FROM requests WHERE id=?', (identity,)).fetchone())
                self.store.set_meta('pending', self.pending(row))
                return error('request_incomplete', self.question(missing, row['action']), self.view(row))
            try:
                target = self._target(row['action'], values)
                if not same_snapshot(json.loads(row['target']) if row['target'] else None,target):
                    raise RequestError('request_stale', 'Запись изменилась после подготовки поручения. Нужна новая просьба.')
            except RequestError as exc:
                self.store.db.execute("UPDATE requests SET status='failed',error=?,updated_at=? WHERE id=?",
                                      (dump({'code': exc.code, 'message': str(exc)}), now, identity))
                self.store.set_meta('pending', None)
                return error(exc.code, str(exc))
            self.store.db.execute("UPDATE requests SET status='executing' WHERE id=?", (identity,))
            args = self.models[row['action']].model_validate(self.action_values(values, row['action']))
            if row['action'] == 'update_reminder':
                args._preserve_due = not bool(set(json.loads(row['provided'])) & REMINDER_TIME_FIELDS)
            source_job = {**job, 'payload': dump({'message': json.loads(row['origin'])}), 'pending': None}
            result = self.apply(row['action'], args, source_job, now)
            status = 'failed' if 'error' in result else 'done'
            self.store.db.execute('UPDATE requests SET status=?,result=?,error=?,updated_at=? WHERE id=?',
                (status, dump(result), dump(result['error']) if 'error' in result else None, now, identity))
            if status == 'done':
                self._receipt(job, row, result, now)
            self.store.set_meta('pending', None)
            return {'request_id': identity, 'status': status, 'result': result}

    def direct(self, action, args, job, now):
        """Legacy complete actions use the same lifecycle without extra model calls."""
        values = args.model_dump(mode='json', exclude={'continue_request'})
        origin = (job.get('pending') or {}).get('origin') if getattr(args, 'continue_request', False) else None
        source_job = {**job, 'payload': dump({'message': origin or json.loads(job['payload'])['message']})}
        current = self.active()
        continued = bool(getattr(args, 'continue_request', False) and current and current['action'] == action)
        prepared = self.prepare(action, values, source_job, now, continued)
        result = self.execute(prepared['request_id'], job, now)
        return result.get('result', result)
