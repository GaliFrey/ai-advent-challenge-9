"""Action confirmations are rendered from committed server receipts, never model prose."""
import json
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from .presentation import public_text
from .requests import ACTIONS
from .profile import profile_text, timezone_name, zone_label
from .recurrence import rule_text

TZ = ZoneInfo('Europe/Kirov')

RESULT_PROTOCOL = '''Финальный ответ без tool_calls всегда возвращай JSON-объектом одного из трёх видов:
{"kind":"answer","text":"Обычный ответ на вопрос или беседу"}
{"kind":"actions","tool_call_ids":["ID выполненного вызова"]}
{"kind":"mixed","tool_call_ids":["ID выполненного вызова"],"text":"Ответ на отдельный вопрос пользователя"}
kind=actions используется для сообщения результатов создания, изменения, удаления и отмены, а также отказов таких операций. В нём запрещён свободный text:
подтверждения, даты и причины ошибок формирует сервер по сохранённым результатам. Указывай ID вызовов выполнения действий (включая update_profile), а не prepare_request или инструментов чтения. Не придумывай ID вызовов.
kind=answer предназначен только для беседы и ответов чтения/поиска, не для заявлений о создании, изменении или удалении записей.
Если пользователь просит выполнить действие и ответить на вопрос, используй kind=mixed: text содержит только объяснение или ответ на вопрос, без подтверждений действий, сроков созданных напоминаний и заявлений об успехе. Подтверждения и ошибки добавляет сервер по квитанциям. Не теряй вопрос из составной просьбы.
Если действие не выполнялось, не заявляй о его успехе. Неполный черновик не является выполненным действием.
Служебный JSON и tool_call_ids пользователю не показываются.'''


class ChatAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['answer']
    text: str = Field(min_length=1, max_length=20000)

    @field_validator('text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Ответ не должен быть пустым')
        return value


class ActionsAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['actions']
    tool_call_ids: list[str] = Field(min_length=1, max_length=100)


class MixedAnswer(ChatAnswer):
    kind: Literal['mixed']
    tool_call_ids: list[str] = Field(min_length=1, max_length=100)


def parse_answer(content):
    try:
        data = json.loads(content)
        model = {'answer': ChatAnswer, 'actions': ActionsAnswer, 'mixed': MixedAnswer}.get(data.get('kind')) if isinstance(data, dict) else None
        if model is None:
            return None
        return model.model_validate(data)
    except (ValueError, TypeError, ValidationError):
        return None


def quote(content):
    text = str(content)
    visible = text[:300]
    suffix = ' … (текст сокращён)' if len(text) > 300 else ''
    # JSON quoting keeps user newlines/quotes from becoming server status lines.
    return json.dumps(visible, ensure_ascii=False) + suffix


def time_text(value, date_only=False, timezone='Europe/Kirov'):
    zone = ZoneInfo(timezone)
    if isinstance(value, (int, float)):
        when = datetime.fromtimestamp(value, zone)
    else:
        when = datetime.fromisoformat(value)
        if when.tzinfo is not None:
            when = when.astimezone(zone)
    if date_only:
        return when.strftime('%d.%m.%Y')
    result = when.strftime('%d.%m.%Y %H:%M:%S' if when.second or when.microsecond else '%d.%m.%Y %H:%M')
    return result + (('.' + f'{when.microsecond:06d}'.rstrip('0')) if when.microsecond else '')


def success_text(row, timezone='Europe/Kirov'):
    data = json.loads(row['payload'])
    action = row['action']
    content = quote(data.get('content', ''))
    if action == 'create_note':
        text = 'Памятка сохранена: ' + content + '.'
    elif action == 'update_note':
        text = 'Памятка исправлена: ' + content + '.'
    elif action == 'delete_note':
        text = 'Памятка удалена: ' + content + '.'
    elif action in ('create_reminder', 'update_reminder'):
        verb = 'Напоминание создано' if action == 'create_reminder' else 'Напоминание изменено'
        zone = data.get('timezone') or timezone
        text = verb + ': ' + content + '. Срок: ' + time_text(data['due_at'], timezone=zone) + ' (' + zone_label(zone) + ').'
    elif action == 'cancel_reminder':
        text = 'Напоминание отменено: ' + content + '.'
    elif action in ('create_schedule', 'update_schedule', 'skip_schedule', 'cancel_schedule'):
        zone = data['timezone']
        if action == 'cancel_schedule':
            text = 'Повторяющееся напоминание отменено: ' + content + '.'
        else:
            verb = {'create_schedule': 'Повторяющееся напоминание создано',
                    'update_schedule': 'Повторяющееся напоминание изменено',
                    'skip_schedule': 'Срабатывание пропущено'}[action]
            text = verb + ': ' + content + '.'
            if action == 'skip_schedule':
                text += ' Пропущенный срок: ' + time_text(data['skipped_due_at'], timezone=zone) + '.'
            text += (' Повтор: ' + rule_text(json.loads(data['rule'])) + ', в ' + data['due_time'] +
                     ' (' + zone + '). '+('Серия завершена.' if data.get('status')=='completed' else 'Следующий срок: '+time_text(data['next_due_at'], timezone=zone)+'.'))
    elif action in ('create_event','change_event','set_event_rule'):
        event = data.get('event',data)
        title = quote((data.get('occurrence') or event).get('content',''))
        if action=='create_event':
            if event.get('kind') == 'birthday':
                birthday = f"{event['birth_day']:02d}.{event['birth_month']:02d}"
                if event.get('birth_year') is not None:
                    birthday += f".{event['birth_year']:04d}"
                text = 'День рождения сохранён: ' + quote(event['person']) + '. Дата рождения: ' + birthday + '. Повтор: ежегодно.'
            else:
                text='Событие сохранено: '+title+'.'
        elif action=='set_event_rule':
            text='Правило напоминания сохранено: '+title+'.'
        elif data.get('operation') == 'disable_rule':
            rule = data['disabled_rule']
            condition = ('в день события' if rule['offset'] == 0 else f"за {rule['offset']} дн.") if rule['kind'] == 'calendar' else (f"за {rule['offset']} с" if rule['kind'] == 'interval' else 'на конкретную дату')
            text = 'Правило напоминания отключено: ' + title + '. Условие: ' + condition.rstrip('.') + '.'
        else:
            text={'move':'Событие перенесено', 'cancel':'Событие отменено', 'delete':'Событие удалено', 'disable':'Уведомления отключены'}[data['operation']]+': '+title+'.'
        if event.get('segments') and data.get('operation') != 'disable_rule' and not (action == 'create_event' and event.get('kind') == 'birthday'):
            seg=data.get('occurrence') or event['segments'][-1]
            text+=' Дата: '+seg['start_date']+(' '+seg['start_time'] if seg['start_time'] else ' (весь день)')+'; пояс '+event['timezone']+'.'
        if data.get('warnings'):
            text+=' '+' '.join(data['warnings'])
    elif action == 'update_profile':
        text = 'Профиль обновлён.\n' + profile_text(data['profile'], data['changed_fields'])
        if data.get('warnings'):
            text+='\n'+' '.join(data['warnings'])
    elif action == 'cancel_request':
        text = 'Поручение отменено.'
    else:
        raise ValueError('Неизвестное подтверждённое действие')
    if action in ('create_note', 'update_note') and data.get('event_date'):
        text += ' Дата события: ' + time_text(data['event_date'], date_only=True) + '.'
    if row['replayed']:
        text = 'Ранее выполнено: ' + text
    return text


def validation_reason(item, timezone='Europe/Kirov'):
    kind = item.get('type')
    limits = item.get('constraints') or {}
    field = str(item.get('field', '')).split('.')[-1]
    if kind == 'value_error' and field == 'timezone':
        return 'нужен известный часовой пояс IANA'
    if kind == 'value_error' and field == 'address':
        return 'нужна одна строка без управляющих символов'
    if kind == 'literal_error' and field == 'language':
        return 'допустимы ru (русский) и en (английский)'
    if kind == 'literal_error' and field == 'style':
        return 'допустимы concise, neutral или detailed'
    if kind == 'string_too_long':
        maximum = limits.get('max_length')
        return f'нужно не более {maximum} символов' if maximum is not None else 'текст слишком длинный'
    if kind == 'string_too_short':
        minimum = limits.get('min_length')
        if minimum == 1:
            return 'текст не должен быть пустым'
        return f'минимальная длина: {minimum}' if minimum is not None else 'текст слишком короткий'
    if kind == 'value_error' and str(item.get('field', '')).split('.')[-1] == 'due_time':
        return 'нужно местное время ' + ('Кирова' if timezone == 'Europe/Kirov' else 'профиля') + ' без смещения'
    return {
        'missing': 'обязательное поле не указано',
        'timezone_aware': 'не указан часовой пояс',
        'datetime_from_date_parsing': 'неверный формат даты и времени',
        'datetime_parsing': 'неверный формат даты и времени',
        'date_from_datetime_parsing': 'неверный формат даты',
        'date_parsing': 'неверный формат даты',
        'time_parsing': 'неверный формат времени',
        'extra_forbidden': 'лишнее поле',
        'string_type': 'ожидается текст',
        'int_parsing': 'ожидается целое число',
        'int_type': 'ожидается целое число',
        'greater_than': 'значение меньше или равно допустимому минимуму',
    }.get(kind, 'значение не соответствует требованиям поля')


def error_text(row, timezone='Europe/Kirov'):
    error = json.loads(row['payload'])
    code = error.get('code')
    if code == 'invalid_arguments':
        names = {'due_at': 'дата и время', 'due_date': 'дата', 'due_time': 'время',
                 'content': 'текст', 'id': 'выбранная запись', 'request_id': 'поручение',
                 'timezone': 'часовой пояс', 'language': 'язык', 'style': 'стиль',
                 'address': 'обращение', 'source_text': 'исходная просьба',
                 'recurrence': 'повтор', 'frequency': 'частота', 'weekdays': 'дни недели',
                 'days': 'числа месяца', 'dates': 'даты года', 'interval': 'интервал', 'start_date': 'дата начала'}
        fields = [names.get(str(item.get('field', '')).split('.')[-1], 'параметр') + ': ' + validation_reason(item, timezone)
                  for item in error.get('details', []) if isinstance(item, dict)]
        message = 'Некорректно указаны параметры' + (': ' + ', '.join(dict.fromkeys(fields)) if fields else '') + '.'
    elif code == 'conflicting_schedule':
        message = 'Срок задан противоречиво: одновременно указаны полная дата со временем и отдельные дата или время. Нужно указать один вариант.'
    elif code == 'selection_mismatch':
        message = 'Выбранная запись не соответствует текущему уточнению.'
    elif code == 'no_pending_request':
        message = 'Нет ожидающего уточнения; нужна новая просьба.'
    elif code == 'unknown_tool':
        message = 'Неизвестный инструмент.'
    else:
        message = public_text(error.get('message') or 'Причина не указана.')
    return 'Попытка отклонена: ' + message


def confirmed_text(store, job_id, extra=None):
    rows = store.receipts(job_id)
    if not rows:
        return None
    lines, seen = [], set()
    for row in rows:
        # Re-execution of one completed request must not look like multiple actions.
        key = (row['request_id'], row['action']) if row['request_id'] is not None and row['outcome'] == 'done' else ('call', row['call_id'])
        if key in seen:
            continue
        seen.add(key)
        lines.append(success_text(row, timezone_name(store)) if row['outcome'] == 'done' else error_text(row, timezone_name(store)))
    if extra and extra not in lines:
        lines.append(extra)
    return '\n'.join(lines)


def final_text(store, job_id, content, draft_text=None, *, messages=()):
    parsed = parse_answer(content)
    rows = store.receipts(job_id)
    # Preparation/reads have no action receipt. Unknown IDs cannot establish
    # that another action occurred; they must not discredit committed results.
    # Only actual execution calls in this job's saved chain can be unconfirmed.
    execution_tools = {*ACTIONS, 'execute_request', 'cancel_request', 'update_profile'}
    action_calls = {c['id'] for m in messages if m['role'] == 'assistant'
                    for c in m.get('tool_calls', []) if c['function']['name'] in execution_tools}
    referenced = set(parsed.tool_call_ids) if isinstance(parsed, (ActionsAnswer, MixedAnswer)) else set()
    returned = {m['tool_call_id'] for m in messages if m['role'] == 'tool'}
    # An omitted reference must not hide a returned execution without a receipt.
    unconfirmed_actions = bool(((referenced | returned) & action_calls) - {r['call_id'] for r in rows})
    confirmed = confirmed_text(store, job_id)
    if confirmed:
        if draft_text is not None:
            confirmed += '\nНезавершённое поручение:\n' + draft_text
        if unconfirmed_actions:
            confirmed += '\nЧасть заявленных результатов не подтверждена.'
        elif isinstance(parsed, MixedAnswer) and referenced <= {r['call_id'] for r in rows}:
            confirmed += '\n\n' + public_text(parsed.text)
        return confirmed
    if unconfirmed_actions:
        # A missing receipt is just as important when there are no other
        # receipts. Neither free model prose nor a draft can prove execution.
        reply = ('Выполнение не подтверждено сохранёнными результатами. '
                 'Проверьте записи перед повторной просьбой.')
        if draft_text is not None:
            reply += '\nНезавершённое поручение:\n' + draft_text
        return reply
    if draft_text is not None:
        return draft_text
    if isinstance(parsed, MixedAnswer):
        return 'Выполнение не подтверждено сохранёнными результатами. Новое действие автоматически не выполнялось.'
    if isinstance(parsed, ChatAnswer):
        return public_text(parsed.text)
    if isinstance(parsed, ActionsAnswer):
        return 'Выполнение не подтверждено сохранёнными результатами. Новое действие автоматически не выполнялось.'
    return 'Не удалось проверить формат ответа. Подтверждённого результата действия нет.'
