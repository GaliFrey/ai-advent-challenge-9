"""Bounded note previews and evidence checks for common memory-absence claims."""
import json
import re


READ_SCOPES = {
    'search_notes': 'notes', 'list_notes': 'notes', 'get_note': 'notes',
    'search_events': 'events', 'calendar': 'events', 'get_event': 'events',
    'list_reminders': 'reminders', 'get_reminder': 'reminders',
    'list_schedules': 'reminders', 'get_schedule': 'reminders',
}
_QUOTED = re.compile(r'«[^»]*»|"(?:\\.|[^"\\])*"', re.S)
_NEGATIVE = re.compile(
    r'\b(?:нет|ничего|не\s+(?:наш[её]л[аои]?|найден\w*|обнаруж\w*|вижу|знаю|помню|сохран\w*|запис\w*|было|храню)|отсутству\w*)\b'
    r'|\b(?:no|nothing|not\s+(?:found|saved|recorded)|don[’\x27]t\s+(?:know|have|remember)|couldn[’\x27]t\s+find)\b', re.I)
_SCOPES = {'notes': r'памят(?:к|ок)\w*|замет(?:к|ок)\w*|notes?',
           'events': r'событи\w*|календар\w*|events?|calendar',
           'reminders': r'напоминан\w*|расписан\w*|reminders?|schedules?'}
_GENERIC = re.compile(r'\b(?:запис\w*|памят[ьи]|переписк\w*|сведени\w*|records?|memory|conversation)\b', re.I)
NOTE_DATE_MEANINGS = {'event_date': 'описанное действие, не сохранение',
                      'note_saved_at': 'сохранение текста памятки',
                      'note_edited_at': 'последняя правка памятки'}


def ambiguous_note_wording(text):
    """Recognize the known ambiguous phrase in prose, preserving source quotes."""
    return bool(re.search(r'\bзапись\s+(?:была\s+)?сделана\b', _QUOTED.sub('', text), re.I))


def absence_scopes(text):
    """Only examine prose, not quoted source data or a user's example."""
    text = _QUOTED.sub('', text)
    scopes = set()
    for sentence in re.split(r'[.!?\n]+', text):
        if not _NEGATIVE.search(sentence):
            continue
        named = {key for key, pattern in _SCOPES.items() if re.search(r'\b(?:' + pattern + r')\b', sentence, re.I)}
        scopes.update(named or ({'any'} if _GENERIC.search(sentence) else set()))
    return scopes


def checked_scopes(messages):
    names = {call['id']: call['function']['name'] for m in messages if m['role'] == 'assistant'
             for call in m.get('tool_calls', [])}
    checked = set()
    for m in messages:
        scope = READ_SCOPES.get(names.get(m.get('tool_call_id')))
        if m['role'] != 'tool' or scope is None:
            continue
        try:
            value = json.loads(m['content'])
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict) or value.get('error') or value.get('deferred'):
            continue
        if (isinstance(value.get('items'), list) and type(value.get('total')) is int) or value.get('id') or value.get('event'):
            checked.add(scope)
    return checked


def unchecked_scopes(text, messages, user_text=''):
    requested, checked = absence_scopes(text), checked_scopes(messages)
    prose = _QUOTED.sub('', text)
    if not requested and _NEGATIVE.search(prose):
        requested = {key for key, pattern in _SCOPES.items() if re.search(r'\b(?:' + pattern + r')\b', user_text, re.I)}
        if not requested and (_GENERIC.search(user_text) or re.search(r'\bне помню\b|\bdon[’\x27]t remember\b', prose, re.I)):
            requested = {'any'}
    return {scope for scope in requested if not checked or (scope != 'any' and scope not in checked)}


def note_dates_for_model(note):
    """Disambiguate timestamps only in API data, never rewrite stored source text."""
    if not isinstance(note, dict) or 'event_date' not in note or ('created_at' not in note and 'note_saved_at' not in note):
        return note
    result = dict(note)
    for source, label in (('created_at', 'note_saved_at'), ('updated_at', 'note_edited_at')):
        if source in result:
            result[label] = result.pop(source)
    result['date_meanings'] = {key: label for key, label in NOTE_DATE_MEANINGS.items() if key in result}
    return result


def note_page_preview(result):
    """Keep originals in the durable result; expose one short copy to the model."""
    if not isinstance(result, dict) or not isinstance(result.get('items'), list):
        return result
    source_items = result['items']
    if len(source_items) > 5:
        end = result.get('next_offset')
        if end is None:
            end = result.get('total', len(source_items))
        result = {**result, 'next_offset': end - len(source_items) + 5, 'incomplete': True}
    items = []
    for note in source_items[:5]:
        dates = note_dates_for_model(note)
        item = {key: dates[key] for key in ('id', 'content', 'event_date', 'note_saved_at') if key in dates}
        content = item.get('content', '')
        # Bound UTF-8 and JSON escaping as well as ordinary character count.
        preview = content[:160]
        while len(json.dumps(preview, ensure_ascii=False).encode('utf-8')) > 480:
            preview = preview[:max(0, len(preview) - 16)]
        item['content'] = preview
        item['incomplete'] = bool(note.get('incomplete') or len(preview) < len(content))
        items.append(item)
    page = {**result, 'items': items, 'incomplete': bool(result.get('incomplete') or any(n['incomplete'] for n in items)),
            'details_tool': 'get_note'}
    if items:
        page['date_meanings'] = {key: label for key, label in NOTE_DATE_MEANINGS.items() if key != 'note_edited_at'}
    return page
