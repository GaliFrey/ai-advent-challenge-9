import json
import re
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, AwareDatetime, field_validator, model_validator, PrivateAttr
from .storage import dump
from .presentation import public_text
from .requests import Requests, RequestError, ACTIONS
from .profile import Profile, ProfilePatch, load_profile, timezone_name
from .search import index_values, match_query, related_query
from .memory import unchecked_scopes, note_page_preview
from .events import Events, EventData, ReminderRuleData, EventError
from .recurrence import Rule, RuleDraft, occurrence, rule_text, local_timestamp, iter_dates

TZ = ZoneInfo('Europe/Kirov')


def local_time(ts, timezone='Europe/Kirov'):
    return datetime.fromtimestamp(ts, ZoneInfo(timezone)).isoformat(timespec='seconds')


class Args(BaseModel):
    model_config = ConfigDict(extra='forbid')


class RequestArgs(Args):
    continue_request: bool = Field(default=False, description='true только для ответа на ожидаемое уточнение; новая просьба всегда false')


class NoteCreate(RequestArgs):
    content: str = Field(min_length=1, max_length=12000)
    event_date: date | None = None


class NoteUpdate(Args):
    id: int = Field(gt=0)
    content: str = Field(min_length=1, max_length=12000)
    event_date: date | None


class Identity(Args):
    id: int = Field(gt=0)


class Page(Args):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)


class Search(Page):
    query: str = Field(min_length=1, max_length=500)


class NotePage(Page):
    limit: int = Field(default=5, ge=1, le=50)


class NoteSearch(NotePage):
    query: str = Field(min_length=1, max_length=500)


class Read(Identity):
    content_offset: int = Field(default=0, ge=0)
    content_limit: int = Field(default=1000, ge=1, le=1000)


class ReminderCreate(RequestArgs):
    content: str = Field(min_length=1, max_length=4000)
    due_at: AwareDatetime
    time_mode: Literal['default','explicit'] = 'explicit'
    timezone: str | None = None
    local_date: date | None = None

    _timezone = field_validator('timezone')(EventData.zone.__func__)


class ReminderUpdate(ReminderCreate):
    id: int = Field(gt=0)
    _preserve_due: bool = PrivateAttr(default=False)


class Choice(Args):
    record_type: Literal['note', 'reminder', 'schedule', 'event']
    id: int = Field(gt=0)


class Clarification(RequestArgs):
    question: str = Field(min_length=1, max_length=2000)
    kind: Literal['note', 'reminder', 'other']
    choices: list[Choice] = Field(default_factory=list, max_length=10, description='При выборе существующей записи: от 2 до 10 вариантов. Номера 1..N и подписи сформирует сервер; не пиши ID в question.')


class DraftValues(Args):
    content: str | None = Field(default=None, min_length=1, max_length=12000)
    due_at: AwareDatetime | None = None
    due_date: date | None = None
    due_time: time | None = None
    event_date: date | None = None
    id: int | None = Field(default=None, gt=0)
    after_seconds: int | None = Field(default=None, ge=1, le=315360000)
    data: dict | None = Field(default=None, description='Событие: content, start_date ISO, start_time HH:MM (обязательно либо all_day=true), end_date/end_time необязательны, timezone IANA, recurrence по схеме повтора. День рождения: kind=birthday, person, start_date (месяц/день), birth_year необязателен.')
    rule: ReminderRuleData | None = None
    operation: Literal['move', 'cancel', 'delete', 'disable', 'disable_rule'] | None = None
    rule_id: int | None = Field(default=None, gt=0, description='Выбранное правило события для operation=disable_rule; ID из get_event')
    scope: Literal['one', 'future'] | None = None
    occurrence_id: int | None = Field(default=None, gt=0)
    from_position: int | None = Field(default=None, ge=0)
    replace_rule_id: int | None = Field(default=None, gt=0)
    recurrence: RuleDraft | None = None
    timezone: str | None = None

    @field_validator('due_time')
    @classmethod
    def local_time_only(cls, value):
        if value is not None and value.tzinfo is not None:
            raise ValueError('due_time задаётся местным временем профиля без смещения')
        return value

    @field_validator('timezone')
    @classmethod
    def known_timezone(cls, value):
        if value is not None:
            try:
                ZoneInfo(value)
            except (KeyError, ValueError):
                raise ValueError('Нужен известный часовой пояс IANA') from None
        return value


class ScheduleCreate(RequestArgs):
    content: str = Field(min_length=1, max_length=4000)
    recurrence: Rule
    due_time: time
    timezone: str
    time_mode: Literal['default','explicit'] = 'explicit'

    _local_time = field_validator('due_time')(DraftValues.local_time_only.__func__)
    _timezone = field_validator('timezone')(DraftValues.known_timezone.__func__)


class ScheduleUpdate(ScheduleCreate):
    id: int = Field(gt=0)


SCHEDULE_MODELS = {'create_schedule': ScheduleCreate, 'update_schedule': ScheduleUpdate,
                   'cancel_schedule': Identity, 'skip_schedule': Identity}


class EventCreate(RequestArgs):
    data: EventData


class EventChange(RequestArgs):
    id: int = Field(gt=0)
    operation: Literal['move', 'cancel', 'delete', 'disable', 'disable_rule']
    rule_id: int | None = Field(default=None, gt=0)
    scope: Literal['one', 'future']
    occurrence_id: int | None = Field(default=None, gt=0)
    from_position: int | None = Field(default=None, ge=0)
    data: EventData | None = None

    @model_validator(mode='after')
    def selected_rule(self):
        if (self.operation == 'disable_rule') != (self.rule_id is not None):
            raise ValueError('rule_id нужен только для отключения выбранного правила через disable_rule')
        return self


class EventRule(RequestArgs):
    id: int = Field(gt=0)
    rule: ReminderRuleData
    scope: Literal['one', 'future']
    occurrence_id: int | None = Field(default=None, gt=0)
    from_position: int | None = Field(default=None, ge=0)
    replace_rule_id: int | None = Field(default=None, gt=0)


class EventRead(Read, Page):
    pass


class CalendarPage(Page):
    start: date
    end: date


class PrepareRequest(Args):
    action: Literal['create_note', 'create_reminder', 'update_note', 'delete_note', 'update_reminder', 'cancel_reminder',
                    'create_schedule', 'update_schedule', 'cancel_schedule', 'skip_schedule', 'create_event', 'change_event', 'set_event_rule'] | None = None
    parameters: DraftValues = Field(default_factory=DraftValues)
    continue_request: bool = False


class RequestIdentity(Args):
    request_id: int = Field(gt=0)


class ProfileChange(ProfilePatch):
    source_text: str = Field(min_length=1, max_length=4096,
        description='Точная цитата явной просьбы изменить постоянные настройки из текущего сообщения пользователя; не из истории или памяток')


EVENT_MODELS = {'create_event':EventCreate,'change_event':EventChange,'set_event_rule':EventRule}

SPECS = {
    'get_event': (EventRead, 'Прочитать событие, участки и правила перед изменением.'),
    'get_event_occurrence': (EventRead, 'Прочитать экземпляр события по ID календаря и его уведомления порциями offset/limit; notifications_incomplete в календаре требует этого чтения.'),
    'search_events': (Search, 'Поиск событий и дней рождения по словам/имени. При неоднозначности уточни; имя не определяет личность.'),
    'calendar': (CalendarPage, 'Календарь за диапазон до 366 дней: события с уведомлениями внутри, самостоятельные напоминания отдельно. offset/limit для продолжения.'),
    'get_profile': (Args, 'Прочитать постоянные настройки пользователя. Не содержит памяток или предположений о привычках.'),
    'update_profile': (ProfileChange, 'Изменить только явно запрошенные постоянные настройки. timezone — IANA; language — ru/en (ответы модели); style — concise/neutral/detailed; address — обращение, null сбрасывает его. Временные просьбы и рассказы не сохранять. Нужна точная source_text из текущего сообщения. /new не сбрасывает профиль.'),
    'prepare_request': (PrepareRequest, 'Создать или дополнить сохраняемый черновик поручения. Для уточнения continue_request=true, action можно не передавать; передавай только изменённые поля. Если известно завтра, но время неизвестно, сохрани due_date без due_time. Сервер вычисляет due_at по дате и местному времени постоянного профиля. Неполное поручение возвращает вопрос, готовое — request_id для execute_request.'),
    'get_request': (Args, 'Прочитать активное структурированное поручение: действие, сохранённые параметры, недостающие поля и состояние. Внутренний request_id пользователю не показывать.'),
    'execute_request': (RequestIdentity, 'Выполнить готовое поручение по внутреннему request_id. Отсутствующие поля, устаревшая цель, отменённое поручение и чужая сессия не допускаются; повтор завершённого поручения не повторяет действие.'),
    'cancel_request': (Args, 'Отменить незавершённое поручение и уточнение; уже созданные памятки и напоминания не удаляет.'),
    'create_note': (NoteCreate, 'Сохранить новую памятку. По явной просьбе сохранить факт, независимо от точных слов, или ответу на её уточнение. event_date — дата события, null если неизвестна.'),
    'search_notes': (NoteSearch, 'Поиск по словам (OR), словоформам и ё/е; только при нуле исходных совпадений — по ограниченному словарю связанных слов. Совпадение не доказывает факт. До 5 превью; детали/оригинал — get_note. При неудаче меняй слова. list_notes вызывай следующим шагом.'),
    'list_notes': (NotePage, 'До 5 превью памяток после оценки поиска. total — все записи; next_offset — продолжение. Детали/оригинал — get_note. Неполный просмотр не доказывает отсутствия записи.'),
    'get_note': (Read, 'Прочитать памятку. Длинные content/original читать по content_offset; next_content_offset указывает продолжение.'),
    'get_reminder': (Read, 'Прочитать напоминание по ID, длинный текст читать по content_offset.'),
    'get_schedule': (Read, 'Прочитать повторяющееся расписание: правило, время, пояс и следующий срок. Длинный текст читать по content_offset, как get_reminder.'),
    'list_schedules': (Page, 'Прочитать активные повторяющиеся расписания. Их будущие срабатывания создаёт планировщик; list_reminders не заменяет этот список.'),
    'update_note': (NoteUpdate, 'Исправить однозначно выбранную памятку по явной просьбе. Передай дату события или null.'),
    'delete_note': (Identity, 'Удалить однозначно выбранную памятку по явной просьбе.'),
    'create_reminder': (ReminderCreate, 'Создать разовое напоминание. По явной просьбе напомнить, независимо от точных слов, или ответу на уточнение. due_at — ISO 8601 с часовым поясом.'),
    'list_reminders': (Page, 'Прочитать активные напоминания, включая ожидающие доставки.'),
    'update_reminder': (ReminderUpdate, 'Перенести или исправить однозначно выбранное активное напоминание по просьбе пользователя.'),
    'cancel_reminder': (Identity, 'Отменить однозначно выбранное активное напоминание.'),
    'ask_clarification': (Clarification, 'Задать уточняющий вопрос и сохранить ожидаемое уточнение. Для незавершённого создания выбери note/reminder, иначе other. При неоднозначной записи передай choices с record_type/id; сервер покажет временную нумерацию.'),
}


def _without_schema_titles(schema):
    """Remove redundant UI labels without changing the tool's data contract."""
    if isinstance(schema, list):
        return [_without_schema_titles(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key == 'title':
            continue
        if key in ('properties', '$defs', 'patternProperties', 'dependentSchemas'):
            result[key] = {name: _without_schema_titles(child) for name, child in value.items()}
        elif key in ('default', 'const', 'enum', 'examples'):
            result[key] = value
        else:
            result[key] = _without_schema_titles(value)
    return result


def definitions():
    return [{'type': 'function', 'function': {'name': name, 'description': desc,
            'parameters': _without_schema_titles(model.model_json_schema())}} for name, (model, desc) in SPECS.items()]


def failure(code, message, details=None):
    result = {'code': code, 'message': message}
    if details is not None:
        result['details'] = details
    return {'error': result}


class ToolError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details


class ToolRunner:
    def __init__(self, store):
        self.store = store
        self.requests = Requests(store, {**{name: SPECS[name][0] for name in ACTIONS if name in SPECS}, **SCHEDULE_MODELS, **EVENT_MODELS}, self._apply)

    def run(self, call, job, now):
        key = f"{job['id']}:{call['id']}"
        cached = self.store.db.execute('SELECT result FROM tool_results WHERE key=?', (key,)).fetchone()
        if cached:
            return json.loads(cached[0])
        job = {**job, 'tool_call_id': call['id']}
        name = call['function']['name']
        try:
            if name not in SPECS:
                raise ToolError('unknown_tool', 'Неизвестный инструмент: ' + name)
            args = SPECS[name][0].model_validate_json(call['function']['arguments'])
            selection = (job.get('pending') or {}).get('selection')
            current_text = json.loads(job['payload']).get('message', {}).get('text', '')
            numbered_answer = re.fullmatch(r'(?:вариант\s*)?\d+[.)]?', current_text.strip(), re.I)
            if name in ('create_note', 'create_reminder', 'create_event') and not args.continue_request and not numbered_answer:
                # An explicit new creation request may supersede a selected record's
                # unfinished clarification. A bare choice number never authorizes it.
                selection = None
            typed = {'get_note': 'note', 'update_note': 'note', 'delete_note': 'note',
                     'get_reminder': 'reminder', 'update_reminder': 'reminder', 'cancel_reminder': 'reminder',
                     'get_schedule': 'schedule', 'get_event': 'event', 'change_event': 'event', 'set_event_rule': 'event'}
            if selection and name in typed and (typed[name], args.id) != (selection['record_type'], selection['id']):
                raise ToolError('selection_mismatch', 'Используй внутреннюю запись из pending.selection, а не временный номер варианта.', selection)
            if selection and name in ('create_note', 'create_reminder', 'create_event'):
                raise ToolError('selection_mismatch', 'Пользователь выбрал существующую запись; это не просьба создать новую.')
            if name != 'prepare_request' and getattr(args, 'continue_request', False) and not job.get('pending'):
                raise ToolError('no_pending_request', 'Нет ожидающего уточнения. Для новой просьбы используй continue_request=false.')
            if name == 'update_reminder' and args.due_at.timestamp() <= now:
                raise ToolError('past_due', 'Указанный срок уже прошёл. Уточни новое время у пользователя.',
                                {'requested_due_at': args.due_at.isoformat(), 'now': local_time(now, timezone_name(self.store))})
            with self.store.atomic():
                result = self._execute(name, args, job, now)
                self._record_outcome(name, result, job, now)
                self.store.db.execute('INSERT INTO tool_results VALUES(?,?)', (key, dump(result)))
            return result
        except ValidationError as exc:
            fields = []
            for issue in exc.errors(include_input=False, include_url=False):
                field = {'field': '.'.join(map(str, issue['loc'])), 'type': issue['type'], 'message': issue['msg']}
                # Persist only schema limits, never arbitrary validator context/input.
                limits = {key: value for key, value in (issue.get('ctx') or {}).items()
                          if key in ('min_length', 'max_length', 'gt', 'ge', 'lt', 'le')
                          and type(value) in (int, float)}
                if limits:
                    field['constraints'] = limits
                fields.append(field)
            invalid_json = any(e['type'] == 'json_invalid' for e in fields)
            result = failure('invalid_json' if invalid_json else 'invalid_arguments',
                             'Аргументы не являются корректным JSON.' if invalid_json else 'Аргументы не соответствуют схеме инструмента.', fields)
        except (ToolError, RequestError, EventError) as exc:
            result = failure(exc.code, str(exc), exc.details)
        with self.store.atomic():
            self._record_outcome(name, result, job, now)
            self.store.db.execute('INSERT OR IGNORE INTO tool_results VALUES(?,?)', (key, dump(result)))
        return result

    def _record_outcome(self, name, result, job, now):
        if name not in {*ACTIONS, 'prepare_request', 'execute_request', 'cancel_request', 'update_profile'}:
            return
        failure = result.get('error') or (result.get('result') or {}).get('error')
        if failure:
            self.store.record_receipt(job, name, 'error', failure, now)
        elif name == 'update_profile':
            self.store.record_receipt(job, name, 'done', result, now)
        elif name == 'cancel_request' and result.get('status') == 'cancelled':
            self.store.record_receipt(job, name, 'done', {}, now, result.get('request_id'))

    def _execute(self, name, a, job, now):
        if name == 'get_profile':
            return {'profile': load_profile(self.store).model_dump()}
        if name == 'update_profile':
            text = json.loads(job['payload']).get('message', {}).get('text', '')
            if not a.source_text.strip() or a.source_text not in text:
                raise ToolError('profile_instruction_missing', 'Нужна явная просьба изменить постоянный профиль в текущем сообщении.')
            previous = load_profile(self.store)
            patch = a.model_dump(exclude_unset=True, exclude={'source_text'})
            updated = Profile.model_validate({**previous.model_dump(), **patch})
            active = self.requests.active()
            pending = self.store.get_meta('pending') or job.get('pending')
            if updated.timezone != previous.timezone and (
                    active or pending):
                raise ToolError('profile_timezone_busy', 'Сначала завершите или отмените незавершённое поручение; затем измените часовой пояс.')
            self.store.set_meta('profile', updated.model_dump())
            warnings = Events(self.store).profile_changed(now) if updated.notification_time != previous.notification_time else []
            if updated.timezone!=previous.timezone:
                Events(self.store).recover_digests(now)
            if updated.notification_time!=previous.notification_time or updated.timezone!=previous.timezone:
                Events(self.store).reschedule_digest(now)
            if previous.birthday_digest and not updated.birthday_digest:
                Events(self.store).disable_digest()
            Events(self.store).digest(now, immediate=updated.birthday_digest and not previous.birthday_digest)
            return {'profile': updated.model_dump(), 'changed_fields': sorted(patch), 'warnings': warnings}
        if name == 'prepare_request':
            patch = a.parameters.model_dump(mode='json', exclude_unset=True)
            return self.requests.prepare(a.action, patch, job, now, a.continue_request)
        if name == 'get_request':
            active = self.requests.active()
            return {'request': self.requests.view(active) if active else None}
        if name == 'execute_request':
            return self.requests.execute(a.request_id, job, now)
        if name == 'cancel_request':
            return self.requests.cancel(now)
        if name in ACTIONS:
            return self.requests.direct(name, a, job, now)
        return self._apply(name, a, job, now)

    def _apply(self, name, a, job, now):
        db = self.store.db
        payload = json.loads(job['payload'])
        msg = payload.get('message', {})
        if getattr(a, 'continue_request', False):
            msg = job['pending']['origin']
        if name == 'ask_clarification':
            if unchecked_scopes(a.question, job.get('memory_messages', []), msg.get('text', '')):
                raise ToolError('memory_not_checked', 'Сначала прочитай память. Уточнение не должно утверждать отсутствие непроверенных записей.')
            if len(a.choices) == 1:
                raise ToolError('invalid_choices', 'Для выбора нужны минимум два варианта; для однозначной записи выбор не нужен.')
            choices = [choice.model_dump() for choice in a.choices]
            if len({(c['record_type'], c['id']) for c in choices}) != len(choices):
                raise ToolError('invalid_choices', 'Варианты не должны повторяться.')
            labels = []
            for choice in choices:
                item = (self._note(choice['id']) if choice['record_type'] == 'note' else
                        self._schedule(choice['id']) if choice['record_type'] == 'schedule' else
                        Events(self.store).view(choice['id']) if choice['record_type'] == 'event' else self._reminder(choice['id']))
                if 'error' in item:
                    raise ToolError('invalid_choices', 'Один из вариантов больше не существует. Заново прочитай записи.')
                content = ' '.join(item['content'].split())
                content = content[:200] + ('…' if len(content) > 200 else '')
                if choice['record_type'] == 'schedule':
                    details = item['repeat'][:300] + ', в ' + item['due_time'] + ' (' + item['timezone'] + ')'
                elif choice['record_type'] == 'reminder':
                    details = datetime.fromisoformat(item['due_at']).strftime('%d.%m.%Y %H:%M')
                elif choice['record_type'] == 'event':
                    details = item['segments'][-1]['start_date'] + ' (' + item['timezone'] + ')'
                elif item['event_date']:
                    details = 'событие ' + date.fromisoformat(item['event_date']).strftime('%d.%m.%Y')
                else:
                    details = 'сохранено ' + datetime.fromisoformat(item['created_at']).strftime('%d.%m.%Y %H:%M')
                labels.append(f'{len(labels) + 1}. «{content}» — {details}')
            question = public_text(a.question)
            if labels:
                question += '\n' + '\n'.join(labels) + '\nОтветь номером варианта.'
            pending = {'kind': a.kind, 'question': question, 'origin': msg}
            if choices:
                pending['choices'] = choices
            elif a.continue_request and (job.get('pending') or {}).get('selection'):
                pending['selection'] = job['pending']['selection']
            if a.continue_request and (job.get('pending') or {}).get('request_id'):
                pending['request_id'] = job['pending']['request_id']
            self.store.set_meta('pending', pending)
            # Keep the exact mapping with the cached result so another tool in
            # this same response cannot erase the clarification before delivery.
            return {'question': question, 'awaiting_answer': True, 'pending': pending}
        events = Events(self.store)
        if name == 'create_event':
            return events.create(a.data, now, msg['message_id'])
        if name == 'change_event':
            return events.change(a.id,a.operation,now,scope=a.scope,occurrence_id=a.occurrence_id,from_position=a.from_position,data=a.data,rule_id=a.rule_id)
        if name == 'set_event_rule':
            return events.add_rule(a.id,a.rule,now,scope=a.scope,occurrence_id=a.occurrence_id,from_position=a.from_position,replace_rule_id=a.replace_rule_id)
        if name in ('get_event','get_event_occurrence'):
            return self._event_details(a,now=now,occurrence=name=='get_event_occurrence')
        if name == 'search_events':
            words=set(index_values(a.query)[1].split())
            rows=[dict(r) for r in db.execute('SELECT id,content,person,kind,timezone,birth_month,birth_day,birth_year FROM events WHERE deleted=0 ORDER BY id')
                  if words.intersection(index_values(r['content']+' '+(r['person'] or ''))[1].split())]
            return self._page([self._excerpt(row, 0, 300) for row in rows[a.offset:a.offset+a.limit]],len(rows),a.offset)
        if name == 'calendar':
            rows=events.calendar(a.start,a.end,now)
            items=[self._excerpt(r,0,300) for r in rows[a.offset:a.offset+a.limit]]
            for item in items:
                item['incomplete']=item['incomplete'] or item.get('notifications_incomplete',False)
            return self._page(items,len(rows),a.offset)
        if name == 'create_note':
            event_date = a.event_date.isoformat() if a.event_date else None
            cur = db.execute('INSERT INTO notes(content,original,event_date,created_at,updated_at,source_message_id) VALUES(?,?,?,?,?,?)',
                             (a.content, msg['text'], event_date, now, now, msg['message_id']))
            db.execute('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)', (cur.lastrowid, *index_values(a.content)))
            self.store.set_meta('pending', None)
            return self._note(cur.lastrowid)
        if name == 'get_note':
            limit = a.content_limit
            while True:
                result = self._note(a.id, a.content_offset, limit)
                budget = job.get('note_read_budget')
                if budget is None or len(dump(result).encode('utf-8')) <= budget:
                    return result
                if limit <= 1:
                    return self._note_read_limit()
                limit = max(1, limit // 2)
        if name == 'get_reminder':
            return self._reminder(a.id, a.content_offset, a.content_limit)
        if name == 'search_notes':
            query = match_query(a.query)
            if not query:
                return {'items': [], 'hint': 'Используй list_notes'}
            count_sql = 'SELECT count(*) FROM notes_fts JOIN notes n ON n.id=notes_fts.rowid WHERE notes_fts MATCH ? AND n.deleted=0'
            total = db.execute(count_sql, (query,)).fetchone()[0]
            # Decide for the whole query, not an empty page past its last offset.
            # Even an irrelevant original hit suppresses dictionary expansion.
            if total == 0:
                related = related_query(a.query)
                if related:
                    query = related
                    total = db.execute(count_sql, (query,)).fetchone()[0]
            rows = db.execute('SELECT n.id FROM notes_fts JOIN notes n ON n.id=notes_fts.rowid WHERE notes_fts MATCH ? AND n.deleted=0 ORDER BY bm25(notes_fts,4.0,1.0),n.id LIMIT ? OFFSET ?', (query, min(a.limit, 5), a.offset)).fetchall()
            return self._note_page([self._note(r[0]) for r in rows], total, a.offset, job.get('note_read_budget'))
        if name == 'list_notes':
            if job.get('defer_note_list') and a.offset == 0:
                return {'deferred': True, 'message': 'Сначала оцени search_notes; затем при необходимости вызови list_notes отдельным шагом.'}
            rows = db.execute('SELECT id FROM notes WHERE deleted=0 ORDER BY id DESC LIMIT ? OFFSET ?', (min(a.limit, 5), a.offset)).fetchall()
            return self._note_page([self._note(r[0]) for r in rows], db.execute('SELECT count(*) FROM notes WHERE deleted=0').fetchone()[0], a.offset, job.get('note_read_budget'))
        if name in ('update_note', 'delete_note'):
            if not self._note(a.id).get('id'):
                return failure('note_not_found', 'Памятка не найдена.')
            db.execute('DELETE FROM notes_fts WHERE rowid=?', (a.id,))
            if name == 'delete_note':
                db.execute('UPDATE notes SET deleted=1,updated_at=? WHERE id=?', (now, a.id))
                self.store.set_meta('pending', None)
                return {'deleted_note_id': a.id}
            db.execute('UPDATE notes SET content=?,event_date=?,updated_at=? WHERE id=?',
                       (a.content, a.event_date.isoformat() if a.event_date else None, now, a.id))
            db.execute('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)', (a.id, *index_values(a.content)))
            self.store.set_meta('pending', None)
            return self._note(a.id)
        if name == 'create_reminder':
            due_at = (local_timestamp(a.local_date,time.fromisoformat(load_profile(self.store).notification_time),ZoneInfo(a.timezone))
                      if a.time_mode=='default' and a.local_date and a.timezone else a.due_at.timestamp())
            cur = db.execute('INSERT INTO reminders(content,due_at,created_at,source_message_id) VALUES(?,?,?,?)',
                             (a.content, due_at, now, msg['message_id']))
            self.store.set_meta('pending', None)
            parent_zone=a.timezone or timezone_name(self.store)
            local=datetime.fromtimestamp(due_at,ZoneInfo(parent_zone))
            db.execute('UPDATE schedules SET timezone=?,local_date=?,due_time=? WHERE id=(SELECT standalone_id FROM reminders WHERE id=?)',
                       (parent_zone,local.date().isoformat(),local.time().isoformat(),cur.lastrowid))
            if a.time_mode=='default':
                if not a.local_date or not a.timezone:
                    raise ToolError('calendar_source_required','Нужны местная дата и закреплённый пояс.')
                db.execute("UPDATE schedules SET time_mode='default',timezone=?,local_date=? WHERE id=(SELECT standalone_id FROM reminders WHERE id=?)",
                           (a.timezone,a.local_date.isoformat(),cur.lastrowid))
            return self._reminder(cur.lastrowid)
        if name in ('create_schedule', 'update_schedule'):
            if name=='update_schedule':
                old=db.execute('SELECT * FROM schedules WHERE id=?',(a.id,)).fetchone()
                previous=Rule.model_validate_json(old['rule'])
                new_clock = time.fromisoformat(load_profile(self.store).notification_time) if a.time_mode == 'default' else a.due_time
                old_clock = time.fromisoformat(load_profile(self.store).notification_time if old['time_mode'] == 'default' else old['due_time'])
                if a.recurrence == previous and a.timezone == old['timezone'] and a.time_mode == old['time_mode'] and new_clock == old_clock:
                    rows = db.execute("SELECT id FROM reminders WHERE schedule_id=? AND is_snooze=0 AND is_manual=0 AND status IN ('pending','queued')", (a.id,)).fetchall()
                    Events(self.store)._guard([row['id'] for row in rows])
                    for row in rows:
                        db.execute('DELETE FROM outbox WHERE reminder_id=? AND sent_at IS NULL', (row['id'],))
                        db.execute("UPDATE reminders SET content=?,status='pending' WHERE id=?", (a.content, row['id']))
                    db.execute('UPDATE schedules SET content=?,updated_at=? WHERE id=?', (a.content, now, a.id))
                    self.store.set_meta('pending', None)
                    return self._schedule(a.id)
                shape=('frequency','interval','weekdays','days','dates','ordinal_weekdays')
                changed=a.recurrence.start_date!=previous.start_date or any(getattr(a.recurrence,k)!=getattr(previous,k) for k in shape)
                if previous.count and changed:
                    old_time=old['due_time'] if old['time_mode']=='explicit' else load_profile(self.store).notification_time
                    boundary=old['next_due_at'] if old['next_due_at']>now else occurrence(previous,old_time,old['timezone'],now)
                    today=datetime.fromtimestamp(now,ZoneInfo(old['timezone'])).date()
                    reusable=db.execute("""SELECT r.occurrence_at FROM reminders r WHERE schedule_id=? AND is_snooze=0 AND is_manual=0 AND status IN ('pending','queued')
                        AND NOT EXISTS(SELECT 1 FROM outbox o WHERE o.reminder_id=r.id AND (o.delivery_started=1 OR o.sent_at IS NOT NULL))""",(a.id,)).fetchall()
                    candidates=[r[0] for r in reusable if r[0] is not None and datetime.fromtimestamp(r[0],ZoneInfo(old['timezone'])).date()>=today]
                    if boundary is not None:
                        candidates.append(boundary)
                    if not candidates:
                        raise ToolError('series_exhausted','Лимит запланированных экземпляров исчерпан.')
                    boundary=min(candidates)
                    consumed=sum(local_timestamp(day,time.fromisoformat(old_time),ZoneInfo(old['timezone']))<boundary for _,day in iter_dates(previous))
                    remaining=previous.count-consumed
                    if remaining<=0:
                        raise ToolError('series_exhausted','Лимит запланированных экземпляров исчерпан.')
                    start=a.recurrence.start_date if a.recurrence.start_date!=previous.start_date else datetime.fromtimestamp(boundary,ZoneInfo(old['timezone'])).date()
                    values={**a.recurrence.model_dump(),'start_date':start}
                    if a.recurrence.count == previous.count and a.recurrence.until is None:
                        values.update(count=remaining,until=None)
                    a=a.model_copy(update={'recurrence':Rule.model_validate(values)})
            due_time = time.fromisoformat(load_profile(self.store).notification_time) if a.time_mode=='default' else a.due_time
            next_due = occurrence(a.recurrence, due_time, a.timezone, now)
            if next_due is None:
                raise ToolError('schedule_out_of_range', 'Не удалось вычислить следующий срок расписания.')
            if name == 'update_schedule':
                self._cancel_schedule_instances(a.id,preserve_manual=True)
                db.execute("UPDATE schedules SET content=?,rule=?,due_time=?,timezone=?,next_due_at=?,updated_at=? WHERE id=? AND status='active'",
                           (a.content, a.recurrence.model_dump_json(), a.due_time.isoformat(), a.timezone, next_due, now, a.id))
                identity = a.id
            else:
                cur = db.execute('INSERT INTO schedules(content,rule,due_time,timezone,next_due_at,source_message_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)',
                                 (a.content, a.recurrence.model_dump_json(), a.due_time.isoformat(), a.timezone, next_due, msg['message_id'], now, now))
                identity = cur.lastrowid
            db.execute('UPDATE schedules SET time_mode=? WHERE id=?',(a.time_mode,identity))
            self.store.set_meta('pending', None)
            return self._schedule(identity)
        if name == 'get_schedule':
            return self._schedule(a.id, a.content_offset, a.content_limit)
        if name == 'list_schedules':
            rows = db.execute("SELECT id FROM schedules WHERE kind='recurring' AND status='active' ORDER BY next_due_at,id LIMIT ? OFFSET ?", (a.limit, a.offset)).fetchall()
            total = db.execute("SELECT count(*) FROM schedules WHERE kind='recurring' AND status='active'").fetchone()[0]
            return self._page([self._schedule(r['id']) for r in rows], total, a.offset)
        if name == 'cancel_schedule':
            self._cancel_schedule_instances(a.id)
            db.execute("UPDATE schedules SET status='cancelled',updated_at=? WHERE id=?", (now, a.id))
            self.store.set_meta('pending', None)
            return {'cancelled_schedule_id': a.id}
        if name == 'skip_schedule':
            row = db.execute("SELECT * FROM schedules WHERE id=? AND status IN ('active','completed')", (a.id,)).fetchone()
            pending = db.execute("SELECT id,due_at,occurrence_at FROM reminders WHERE schedule_id=? AND status IN ('pending','queued') ORDER BY due_at LIMIT 1", (a.id,)).fetchone()
            if pending and (row['status']=='completed' or pending['due_at']<=row['next_due_at']):
                self._cancel_schedule_instances(a.id, pending['id'])
                db.execute("UPDATE reminders SET reason='user_skipped' WHERE schedule_id=? AND occurrence_at=? AND status='cancelled'",(a.id,pending['occurrence_at']))
                skipped = pending['due_at']
            else:
                skipped = row['next_due_at']
                if skipped <= now:
                    skipped = occurrence(Rule.model_validate_json(row['rule']), row['due_time'] if row['time_mode']=='explicit' else load_profile(self.store).notification_time, row['timezone'], now, previous=True)
                following = occurrence(Rule.model_validate_json(row['rule']), row['due_time'] if row['time_mode']=='explicit' else load_profile(self.store).notification_time, row['timezone'], max(skipped, now))
                db.execute("INSERT OR IGNORE INTO reminders(content,due_at,planned_due_at,created_at,source_message_id,schedule_id,occurrence_at,status,reason) VALUES(?,?,?,?,?,?,?,'cancelled','user_skipped')",(row['content'],skipped,skipped,now,msg['message_id'],a.id,skipped))
                if following is None:
                    db.execute("UPDATE schedules SET status='completed',updated_at=? WHERE id=?",(now,a.id))
                else:
                    db.execute('UPDATE schedules SET next_due_at=?,updated_at=? WHERE id=?', (following, now, a.id))
            self.store.set_meta('pending', None)
            return {**self._schedule(a.id), 'skipped_due_at': local_time(skipped, row['timezone'])}
        if name == 'list_reminders':
            rows = db.execute("SELECT id FROM reminders WHERE status IN ('pending','queued') ORDER BY due_at,id LIMIT ? OFFSET ?", (a.limit, a.offset)).fetchall()
            return self._page([self._reminder(r[0]) for r in rows], db.execute("SELECT count(*) FROM reminders WHERE status IN ('pending','queued')").fetchone()[0], a.offset)
        if name in ('update_reminder', 'cancel_reminder'):
            row = db.execute("SELECT * FROM reminders WHERE id=? AND status IN ('pending','queued')", (a.id,)).fetchone()
            if not row:
                return failure('reminder_not_active', 'Активное напоминание не найдено: оно отсутствует, отменено или уже доставлено. Для доставленного можно создать новое по явной просьбе.')
            if row['event_rule_id'] is not None:
                raise ToolError('event_rule_required','Измените правило события или используйте кнопку откладывания.')
            if db.execute('SELECT 1 FROM outbox WHERE reminder_id=? AND (delivery_started=1 OR sent_at IS NOT NULL)', (a.id,)).fetchone():
                return failure('delivery_in_progress', 'Отправка напоминания уже началась; результат доставки может быть неизвестен. Отмена или перенос не выполнены: остановить доставку гарантированно нельзя. Проверьте доставку перед новой просьбой.')
            db.execute('DELETE FROM outbox WHERE reminder_id=? AND sent_at IS NULL', (a.id,))
            if name == 'cancel_reminder':
                db.execute("UPDATE reminders SET status='cancelled',reason='user_cancelled' WHERE id=?", (a.id,))
                if row['standalone_id']:
                    db.execute("UPDATE schedules SET status='cancelled' WHERE id=?",(row['standalone_id'],))
                self.store.set_meta('pending', None)
                return {'cancelled_reminder_id': a.id}
            if a._preserve_due:
                db.execute("UPDATE reminders SET content=?,status='pending' WHERE id=?", (a.content, a.id))
                if row['standalone_id']:
                    db.execute('UPDATE schedules SET content=? WHERE id=?', (a.content, row['standalone_id']))
            else:
                due_at = a.due_at.timestamp()
                if a.time_mode == 'default':
                    if not a.local_date or not a.timezone:
                        raise ToolError('calendar_source_required','Нужны местная дата и закреплённый пояс.')
                    due_at = local_timestamp(a.local_date, time.fromisoformat(load_profile(self.store).notification_time), ZoneInfo(a.timezone))
                db.execute("UPDATE reminders SET content=?,due_at=?,planned_due_at=CASE WHEN standalone_id IS NOT NULL AND is_snooze=0 THEN ? ELSE planned_due_at END,status='pending',is_manual=? WHERE id=?",
                           (a.content, due_at, due_at, int(a.time_mode != 'default' or not row['standalone_id']), a.id))
                if row['standalone_id']:
                    parent = db.execute('SELECT timezone,local_date FROM schedules WHERE id=?', (row['standalone_id'],)).fetchone()
                    zone = parent['timezone'] if parent['local_date'] else a.timezone or timezone_name(self.store)
                    local = datetime.fromtimestamp(due_at, ZoneInfo(zone))
                    db.execute('UPDATE schedules SET content=?,absolute_due_at=?,time_mode=?,timezone=?,local_date=?,due_time=? WHERE id=?',
                               (a.content,due_at,a.time_mode,zone,local.date().isoformat(),local.time().isoformat(),row['standalone_id']))
            self.store.set_meta('pending', None)
            return self._reminder(a.id)
        raise ValueError('Неизвестный инструмент')

    def _event_details(self,a,*,now,occurrence=False):
        events=Events(self.store)
        upcoming = events.next_occurrence(a.id, now) if not occurrence else None
        limit=a.limit
        while True:
            if occurrence:
                found=self.store.db.execute('SELECT * FROM event_occurrences WHERE id=?',(a.id,)).fetchone()
                if not found:
                    raise EventError('occurrence_not_found','Экземпляр не найден.')
                event=events.event(found['event_id'])
                result={**dict(found),'timezone':event['timezone']}
                result['notifications']=events.notification_page(a.id,a.offset,limit)
                total=self.store.db.execute('SELECT count(*) FROM reminders WHERE event_occurrence_id=?',(a.id,)).fetchone()[0]
                result['notifications_total']=total
                result['next_notification_offset']=a.offset+len(result['notifications']) if a.offset+len(result['notifications'])<total else None
                incomplete=result['next_notification_offset'] is not None
            else:
                result=events.view(a.id,a.offset,limit)
                result['next_occurrence']=upcoming
                for index,segment in enumerate(result['segments']):
                    full=dict(self.store.db.execute('SELECT * FROM event_segments WHERE id=?',(segment['id'],)).fetchone())
                    result['segments'][index]=self._excerpt(full,a.content_offset,min(a.content_limit,300))
                incomplete=result['incomplete'] or any(segment['incomplete'] for segment in result['segments'])
            try:
                excerpt=self._excerpt(result,a.content_offset,a.content_limit)
                excerpt['incomplete']=excerpt['incomplete'] or incomplete
                return excerpt
            except ToolError as exc:
                if exc.code!='result_too_large' or limit<=1:
                    raise
                limit=max(1,limit//2)

    @staticmethod
    def _page(items, total, offset):
        selected = []
        for item in items:
            if len(dump(selected + [item]).encode('utf-8')) > 12000:
                break
            selected.append(item)
        next_offset = offset + len(selected)
        return {'items': selected, 'total': total,
                'incomplete': next_offset < total or any(item.get('incomplete', False) for item in selected),
                'next_offset': next_offset if next_offset < total else None}

    @staticmethod
    def _note_read_limit():
        return {'limited': True, 'message': 'Лимит чтения памяти: заверши ответ по проверенной части. Поиск неполон; отсутствие записи не подтверждено.'}

    def _note_page(self, items, total, offset, budget):
        page = self._page(items, total, offset)
        if budget is None:
            return page
        while len(dump(note_page_preview(page)).encode('utf-8')) > budget:
            if not page['items']:
                return self._note_read_limit()
            page = self._page(page['items'][:-1], total, offset)
            if not page['items'] and items:
                return self._note_read_limit()
        return page

    @staticmethod
    def _excerpt(result, offset, limit):
        # Character counts alone do not bound JSON: control characters expand to
        # six bytes. Leave room for the page envelope and always return progress.
        texts = {key: result[key] for key in ('content', 'original') if key in result}
        lengths = {key: len(text) for key, text in texts.items()}
        while True:
            end = offset + limit
            excerpt = {**result, **{key: text[offset:end] for key, text in texts.items()}}
            excerpt.update(content_offset=offset, text_lengths=lengths,
                           incomplete=any(n > end for n in lengths.values()),
                           next_content_offset=end if any(n > end for n in lengths.values()) else None)
            if len(dump(excerpt).encode('utf-8')) <= 10000:
                return excerpt
            if limit <= 1:
                raise ToolError('result_too_large', 'Метаданные записи превышают допустимый размер выдачи.')
            limit = max(1, limit // 2)

    def _note(self, note_id, offset=0, limit=1000):
        row = self.store.db.execute('SELECT * FROM notes WHERE id=? AND deleted=0', (note_id,)).fetchone()
        if not row:
            return failure('note_not_found', 'Памятка не найдена.')
        result = dict(row)
        for key in ('created_at', 'updated_at'):
            result[key] = local_time(result[key], timezone_name(self.store))
        del result['deleted']
        return self._excerpt(result, offset, limit)

    def _reminder(self, reminder_id, offset=0, limit=1000):
        row = self.store.db.execute('SELECT * FROM reminders WHERE id=?', (reminder_id,)).fetchone()
        if row is None:
            return failure('reminder_not_found', 'Напоминание не найдено.')
        result = dict(row)
        schedule = self.store.db.execute('SELECT timezone FROM schedules WHERE id=?', (row['schedule_id'],)).fetchone() if row['schedule_id'] else None
        event=self.store.db.execute('SELECT e.timezone FROM events e JOIN event_occurrences o ON o.event_id=e.id WHERE o.id=?',(row['event_occurrence_id'],)).fetchone() if row['event_occurrence_id'] else None
        standalone=self.store.db.execute("SELECT timezone FROM schedules WHERE id=? AND local_date IS NOT NULL",(row['standalone_id'],)).fetchone() if row['standalone_id'] else None
        zone = event['timezone'] if event else standalone['timezone'] if standalone else schedule['timezone'] if schedule else timezone_name(self.store)
        for key in ('created_at', 'due_at', 'occurrence_at'):
            if result[key] is not None:
                result[key] = local_time(result[key], zone)
        if schedule:
            result['timezone'] = zone
        return self._excerpt(result, offset, limit)

    def _schedule(self, identity, offset=0, limit=1000):
        row = self.store.db.execute("SELECT * FROM schedules WHERE id=? AND kind='recurring'", (identity,)).fetchone()
        if row is None:
            return failure('schedule_not_found', 'Повторяющееся напоминание не найдено.')
        result = dict(row)
        result['recurrence'] = json.loads(result.pop('rule'))
        result['repeat'] = rule_text(result['recurrence'])
        if row['time_mode']=='default':
            result['due_time']=load_profile(self.store).notification_time
        for key in ('next_due_at', 'created_at', 'updated_at'):
            result[key] = local_time(result[key], result['timezone'])
        return self._excerpt(result, offset, limit)

    def _cancel_schedule_instances(self, identity, reminder_id=None, *, preserve_manual=False):
        occurrence_at = None
        if reminder_id is not None:
            occurrence_at = self.store.db.execute('SELECT occurrence_at FROM reminders WHERE id=?', (reminder_id,)).fetchone()[0]
        rows = self.store.db.execute("SELECT id FROM reminders WHERE schedule_id=? AND status IN ('pending','queued')" +
                                     (' AND occurrence_at=?' if reminder_id is not None else '') + (' AND is_snooze=0 AND is_manual=0' if preserve_manual else ''),
                                     (identity, occurrence_at) if reminder_id is not None else (identity,)).fetchall()
        ids = [r['id'] for r in rows]
        for rid in ids:
            if self.store.db.execute('SELECT 1 FROM outbox WHERE reminder_id=? AND (delivery_started=1 OR sent_at IS NOT NULL)', (rid,)).fetchone():
                raise ToolError('delivery_in_progress', 'Отправка срабатывания уже началась; результат может быть неизвестен. Расписание не изменено и не отменено.')
        for rid in ids:
            self.store.db.execute('DELETE FROM outbox WHERE reminder_id=? AND sent_at IS NULL', (rid,))
            self.store.db.execute("UPDATE reminders SET status='cancelled',reason=? WHERE id=?", ('schedule_changed' if preserve_manual else 'user_cancelled',rid))
