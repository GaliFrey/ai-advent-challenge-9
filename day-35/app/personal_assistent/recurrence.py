"""Calendar recurrence rules. Dates are computed by code, independently of delivery."""
import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MonthDay(BaseModel):
    model_config = ConfigDict(extra='forbid')
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)

    @model_validator(mode='after')
    def valid_date(self):
        date(2000,self.month,self.day)
        return self


class OrdinalWeekday(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ordinal: Literal[-1, 1, 2, 3, 4, 5]
    weekday: int = Field(ge=0, le=6)


class RuleDraft(BaseModel):
    model_config = ConfigDict(extra='forbid')
    frequency: Literal['daily', 'weekly', 'monthly', 'yearly'] | None = None
    interval: int = Field(default=1, ge=1, le=100)
    weekdays: list[int] = Field(default_factory=list, max_length=7, description='0=понедельник, 6=воскресенье')
    days: list[int] = Field(default_factory=list, max_length=31, description='Числа месяца, 1–31; в коротком месяце — последний день')
    dates: list[MonthDay] = Field(default_factory=list, max_length=100, description='Даты ежегодного повтора: месяц и число')
    start_date: date | None = None
    version: Literal[1] = 1
    ordinal_weekdays: list[OrdinalWeekday] = Field(default_factory=list, max_length=35)
    until: date | None = None
    count: int | None = Field(default=None, ge=1, le=100000)

    @field_validator('weekdays', 'days')
    @classmethod
    def valid_numbers(cls, value, info):
        low, high = (0, 6) if info.field_name == 'weekdays' else (1, 31)
        if any(not low <= n <= high for n in value) or len(value) != len(set(value)):
            raise ValueError('Недопустимые или повторяющиеся дни')
        return sorted(value)

    @model_validator(mode='after')
    def consistent(self):
        if len({(d.month, d.day) for d in self.dates}) != len(self.dates):
            raise ValueError('Даты не должны повторяться')
        allowed = {'daily': (), 'weekly': ('weekdays',), 'monthly': ('days', 'ordinal_weekdays'), 'yearly': ('dates',)}
        if self.frequency:
            for key in ('weekdays', 'days', 'dates', 'ordinal_weekdays'):
                if getattr(self, key) and key not in allowed[self.frequency]:
                    raise ValueError('Дни не соответствуют виду повтора')
        if self.until and self.start_date and self.until < self.start_date:
            raise ValueError('Окончание раньше начала')
        if self.until and self.count:
            raise ValueError('Укажите дату окончания либо число экземпляров')
        if len({(d.ordinal, d.weekday) for d in self.ordinal_weekdays}) != len(self.ordinal_weekdays):
            raise ValueError('Порядковые дни не должны повторяться')
        return self


class Rule(RuleDraft):
    frequency: Literal['daily', 'weekly', 'monthly', 'yearly']
    start_date: date

    @model_validator(mode='after')
    def complete(self):
        key = {'weekly': 'weekdays', 'monthly': 'days', 'yearly': 'dates'}.get(self.frequency)
        if key and not getattr(self, key) and not (self.frequency == 'monthly' and self.ordinal_weekdays):
            raise ValueError('Укажите конкретные дни или даты повторения')
        return self


def local_timestamp(day, at, zone):
    """On a clock change use the first occurrence, or shift forward through a gap."""
    naive = datetime.combine(day, at)
    candidates = [naive.replace(tzinfo=zone, fold=fold) for fold in (0, 1)]
    valid = [v for v in candidates if v.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == naive]
    if valid:
        return min(v.timestamp() for v in valid)
    return max(v.timestamp() for v in candidates)


def _period(rule, day):
    start = rule.start_date
    if rule.frequency == 'daily':
        distance = (day - start).days
    elif rule.frequency == 'weekly':
        distance = ((day - timedelta(days=day.weekday())) - (start - timedelta(days=start.weekday()))).days // 7
    elif rule.frequency == 'monthly':
        distance = (day.year - start.year) * 12 + day.month - start.month
    else:
        distance = day.year - start.year
    return distance // rule.interval


def _days(rule, period):
    start, step = rule.start_date, period * rule.interval
    if rule.frequency == 'daily':
        return [start + timedelta(days=step)]
    if rule.frequency == 'weekly':
        monday = start - timedelta(days=start.weekday()) + timedelta(weeks=step)
        return [monday + timedelta(days=d) for d in rule.weekdays]
    if rule.frequency == 'monthly':
        year, month = divmod(start.year * 12 + start.month - 1 + step, 12)
        month += 1
        last = calendar.monthrange(year, month)[1]
        values = {date(year, month, min(d, last)) for d in rule.days}
        for item in rule.ordinal_weekdays:
            if item.ordinal == -1:
                day = last - (date(year, month, last).weekday() - item.weekday) % 7
            else:
                day = 1 + (item.weekday - date(year, month, 1).weekday()) % 7 + 7 * (item.ordinal - 1)
            if day <= last:
                values.add(date(year, month, day))
        return sorted(values)
    year = start.year + step
    return sorted({date(year, d.month, min(d.day, calendar.monthrange(year, d.month)[1])) for d in rule.dates})


def occurrence(rule, at, timezone_name, reference, *, previous=False):
    """Next strictly after reference, or latest at/before it. Never scan missed days."""
    if rule.count:
        values = [local_timestamp(day, time.fromisoformat(at) if isinstance(at, str) else at, ZoneInfo(timezone_name))
                  for _, day in iter_dates(rule)]
        matches = [v for v in values if v <= reference] if previous else [v for v in values if v > reference]
        return (max(matches) if previous else min(matches)) if matches else None
    zone = ZoneInfo(timezone_name)
    at = time.fromisoformat(at) if isinstance(at, str) else at
    period = _period(rule, datetime.fromtimestamp(reference, zone).date())
    period = period if previous else max(0, period)
    direction = -1 if previous else 1
    while period >= 0:
        try:
            values = sorted({local_timestamp(day, at, zone) for day in _days(rule, period) if day >= rule.start_date and (rule.until is None or day <= rule.until)})
        except (ValueError, OverflowError):
            return None
        matches = [v for v in values if v <= reference] if previous else [v for v in values if v > reference]
        if matches:
            return max(matches) if previous else min(matches)
        if not previous and rule.until and period > _period(rule,rule.until):
            return None
        period += direction
    return None


def rule_text(rule):
    rule = Rule.model_validate(rule) if isinstance(rule, dict) else rule
    prefix = {'daily': 'каждый день', 'weekly': 'каждую неделю', 'monthly': 'каждый месяц', 'yearly': 'каждый год'}[rule.frequency]
    if rule.interval != 1:
        forms = {'daily': ('день', 'дня', 'дней'), 'weekly': ('неделю', 'недели', 'недель'),
                 'monthly': ('месяц', 'месяца', 'месяцев'), 'yearly': ('год', 'года', 'лет')}[rule.frequency]
        n = rule.interval
        form = forms[2] if 11 <= n % 100 <= 14 else forms[0] if n % 10 == 1 else forms[1] if 2 <= n % 10 <= 4 else forms[2]
        prefix = f'раз в {n} {form}, начало {rule.start_date:%d.%m.%Y}'
    if rule.frequency == 'weekly':
        prefix += ': ' + ', '.join(('пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс')[d] for d in rule.weekdays)
    elif rule.frequency == 'monthly':
        prefix += ', числа: ' + ', '.join(map(str, rule.days))
    elif rule.frequency == 'yearly':
        prefix += ', даты: ' + ', '.join(f'{d.day:02d}.{d.month:02d}' for d in sorted(rule.dates, key=lambda d: (d.month, d.day)))
    if rule.ordinal_weekdays:
        prefix += ', дни недели: ' + ', '.join(f'{d.ordinal}-й {d.weekday+1}' for d in rule.ordinal_weekdays)
    if rule.until:
        prefix += f'; до {rule.until:%d.%m.%Y} включительно'
    if rule.count:
        prefix += f'; запланировано {rule.count}'
    if rule.frequency in ('monthly', 'yearly'):
        prefix += '; в коротком месяце — последний день (совпавшие даты дают одно напоминание)'
    return prefix


def iter_dates(rule, *, through=None):
    """Yield stable zero-based positions, counting cancelled positions as planned."""
    index = 0
    period = 0
    while True:
        try:
            days = _days(rule, period)
        except (ValueError, OverflowError):
            return
        for day in days:
            if day < rule.start_date:
                continue
            if (rule.until and day > rule.until) or (through and day > through) or (rule.count and index >= rule.count):
                return
            yield index, day
            index += 1
        # Empty ordinal months must still terminate at the requested horizon.
        period += 1
        if period > 120000:
            return
        if through and _period(rule, through) < period:
            return
