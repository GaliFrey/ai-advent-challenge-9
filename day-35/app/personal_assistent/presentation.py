"""User-facing text: database identifiers remain in tool data only."""
import re

_RECORD = r'(памятк[а-яё]*|напоминани[а-яё]*|расписани[а-яё]*|событи[а-яё]*)'


def _hide_metadata(text):
    # Keep dates, times, quantities and temporary list numbers intact.
    text = re.sub(r'\b' + _RECORD + r'\s*(?:с\s+)?(?:№|#|id\s*[:=]?)\s*\d+\b',
                  r'\1', text, flags=re.I)
    text = re.sub(r'\b(?:id|идентификатор)\s+' + _RECORD + r'\s*[:=№#]?\s*\d+\b',
                  r'\1', text, flags=re.I)
    text = re.sub(r'\s*[\[(]\s*(?:внутренн[а-яё]*\s+)?id\s*[:=#]?\s*\d+\s*[\])]', '', text, flags=re.I)
    text = re.sub(r'\b(?:внутренн[а-яё]*\s+id|note_id|reminder_id|schedule_id|event_id|event_occurrence_id|event_rule_id|rule_id|source_message_id)\s*[:=#]?\s*\d+\b', '', text, flags=re.I)
    return text


def public_text(text):
    # Quoted source material is user data: its IDs may belong to a device/order,
    # rather than this application's database. Only format surrounding prose.
    quoted = re.compile(r'«[^»]*»|"(?:\\.|[^"\\])*"', re.S)
    parts, offset = [], 0
    for match in quoted.finditer(text):
        parts.extend((_hide_metadata(text[offset:match.start()]), match[0]))
        offset = match.end()
    parts.append(_hide_metadata(text[offset:]))
    return ''.join(parts).strip()


def notification_text(text, first_part=True):
    # A reminder body is entirely user data, including unquoted technical labels.
    if not first_part:
        return text
    header, newline, body = text.partition('\n')
    return public_text(header) + newline + body
