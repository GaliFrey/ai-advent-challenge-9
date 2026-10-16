"""Строгая серверная проверка диагностического снимка версии 1.

Контракт описан в ``docs/support-context.md``, раздел 5.3. Проверка не ослабляет
контракт: обязательные поля и контейнеры должны присутствовать, типы и
допустимые значения соблюдаться, а числа — быть конечными. Любое нарушение даёт
понятную ошибку; частичной записи не происходит, потому что проверка выполняется
до транзакции.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .errors import SupportError

SCHEMA_VERSION = 1

_COLLECTED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

_STATUSES = {"idle", "requesting", "initializing", "capturing", "error"}
_ERROR_NAMES = {
    "NotAllowedError",
    "NotFoundError",
    "NotReadableError",
    "AbortError",
    "InvalidStateError",
    "SecurityError",
    "TypeError",
    "OverconstrainedError",
    "other",
}
_TRACK_STATES = {"live", "ended"}
_CONTEXT_STATES = {"suspended", "running", "closed", "interrupted"}
_STRING_NAMES = {"E2", "A2", "D3", "G3", "B3", "E4"}
_DIRECTIONS = {"Ослабить", "Подтянуть"}


def _fail(message: str) -> None:
    raise SupportError("invalid_snapshot", f"Некорректный диагностический снимок: {message}")


def _is_number(value: Any) -> bool:
    # bool — подкласс int, но числом для контракта не является.
    return type(value) in (int, float)


def _finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value)


def _require_object(container: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in container:
        _fail(f"отсутствует поле {key}.")
    value = container[key]
    if not isinstance(value, dict):
        _fail(f"поле {key} должно быть объектом.")
    return value


def _require_key(container: dict[str, Any], key: str, where: str) -> Any:
    if key not in container:
        _fail(f"отсутствует поле {where}.{key}.")
    return container[key]


def _nullable_enum(container: dict[str, Any], key: str, allowed: set[str], where: str) -> None:
    value = _require_key(container, key, where)
    if value is None:
        return
    if not isinstance(value, str) or value not in allowed:
        _fail(f"поле {where}.{key} имеет недопустимое значение.")


def _nullable_bool(container: dict[str, Any], key: str, where: str) -> None:
    value = _require_key(container, key, where)
    if value is None:
        return
    if type(value) is not bool:
        _fail(f"поле {where}.{key} должно быть логическим или null.")


def _nullable_number(
    container: dict[str, Any],
    key: str,
    where: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = _require_key(container, key, where)
    if value is None:
        return None
    if not _finite_number(value):
        _fail(f"поле {where}.{key} должно быть конечным числом или null.")
    if minimum is not None and value < minimum:
        _fail(f"поле {where}.{key} меньше допустимого минимума.")
    if maximum is not None and value > maximum:
        _fail(f"поле {where}.{key} больше допустимого максимума.")
    return value


def _nullable_string(
    container: dict[str, Any], key: str, allowed: set[str], where: str
) -> str | None:
    value = _require_key(container, key, where)
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        _fail(f"поле {where}.{key} имеет недопустимое значение.")
    return value


def validate_snapshot(value: Any) -> dict[str, Any]:
    """Проверяет снимок версии 1 и возвращает его. При нарушении — SupportError."""

    if not isinstance(value, dict):
        _fail("снимок должен быть JSON-объектом.")

    allowed = {
        'schemaVersion', 'collectedAt', 'capture', 'audio', 'device',
        'measurement', 'display', 'commit',
    }
    if set(value) != allowed:
        _fail('набор полей должен соответствовать версии 1.')
    sections = {
        'capture': {'status', 'lastError'},
        'audio': {'trackState', 'trackEnabled', 'trackMuted', 'contextState'},
        'device': {'actualLabel'},
        'measurement': {'rms', 'frequencyHz', 'matchedString', 'ageMs'},
        'display': {'string', 'cents', 'direction', 'tuned'},
        'commit': {'ageMs'},
    }
    for section, keys in sections.items():
        if not isinstance(value.get(section), dict) or set(value[section]) != keys:
            _fail('набор полей раздела должен соответствовать версии 1.')

    schema_version = value.get("schemaVersion", "__missing__")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        _fail("schemaVersion должен быть целым числом 1.")

    collected_at = value.get("collectedAt", "__missing__")
    if not isinstance(collected_at, str) or not _COLLECTED_AT_RE.match(collected_at):
        _fail("collectedAt должен быть строкой UTC ISO 8601 с миллисекундами.")

    capture = _require_object(value, "capture")
    _nullable_enum(capture, "status", _STATUSES, "capture")
    _nullable_enum(capture, "lastError", _ERROR_NAMES, "capture")

    audio = _require_object(value, "audio")
    _nullable_enum(audio, "trackState", _TRACK_STATES, "audio")
    _nullable_bool(audio, "trackEnabled", "audio")
    _nullable_bool(audio, "trackMuted", "audio")
    _nullable_enum(audio, "contextState", _CONTEXT_STATES, "audio")

    device = _require_object(value, "device")
    actual_label = _require_key(device, "actualLabel", "device")
    if actual_label is not None:
        _fail("device.actualLabel в версии 1 всегда null.")

    measurement = _require_object(value, "measurement")
    rms = _nullable_number(measurement, "rms", "measurement", minimum=0)
    frequency = _nullable_number(
        measurement, "frequencyHz", "measurement", minimum=65, maximum=450
    )
    matched_string = _nullable_string(
        measurement, "matchedString", _STRING_NAMES, "measurement"
    )
    measurement_age = _nullable_number(measurement, "ageMs", "measurement", minimum=0)

    # Согласованность: без измерения три значения тоже null; струна не может
    # быть распознана без оценки частоты.
    if measurement_age is None and (rms is not None or frequency is not None or matched_string is not None):
        _fail("без measurement.ageMs значения измерения должны быть null.")
    if matched_string is not None and frequency is None:
        _fail("measurement.matchedString не может быть задан без frequencyHz.")

    display = _require_object(value, "display")
    display_string = _nullable_string(display, "string", _STRING_NAMES, "display")
    cents = _nullable_number(display, "cents", "display", minimum=-100, maximum=100)
    if cents is not None and not isinstance(cents, int):
        _fail("display.cents должен быть целым числом или null.")
    direction = _nullable_string(display, "direction", _DIRECTIONS, "display")
    tuned = _require_key(display, "tuned", "display")
    if type(tuned) is not bool:
        _fail("display.tuned должен быть логическим.")

    commit = _require_object(value, "commit")
    commit_age = _nullable_number(commit, "ageMs", "commit", minimum=0)

    if display_string is None:
        # Пустой показ: ни числа, ни направления, ни метки, ни возраста.
        if cents is not None or direction is not None or tuned is not False or commit_age is not None:
            _fail("при пустом показе display и commit должны быть пустыми.")
    else:
        if tuned:
            # Метка «настроено» скрывает число и направление.
            if cents is not None or direction is not None:
                _fail("при display.tuned=true число и направление должны быть null.")
        else:
            if cents is not None:
                # Показанное число по контракту не попадает в зону ±5 центов.
                if abs(cents) < 6:
                    _fail("показанное display.cents должно быть не меньше 6 по модулю.")
                if direction is None:
                    _fail("при показанном числе display.direction обязателен.")
            elif direction is not None:
                _fail("display.direction не может быть задан без числа.")

    if commit_age is not None and display_string is None:
        _fail("commit.ageMs не может быть задан без текущего показания.")

    return value
