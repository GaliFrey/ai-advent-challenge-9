"""Calendar domain. Model extracts values; this module owns dates and state changes."""
import calendar
import unicodedata
from itertools import islice
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, model_validator, field_validator
from .profile import load_profile
from .recurrence import Rule, iter_dates, local_timestamp


class EventError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.details = None


def person_key(value):
    return ' '.join(unicodedata.normalize('NFC',value).casefold().split())


class EventData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    content: str = Field(min_length=1, max_length=4000)
    start_date: date
    start_time: time | None = None
    end_date: date | None = None
    end_time: time | None = None
    all_day: bool = False
    timezone: str | None = None
    recurrence: Rule | None = None
    kind: Literal['event', 'birthday'] = 'event'
    person: str | None = Field(default=None, min_length=1, max_length=200)
    birth_year: int | None = Field(default=None, ge=1, le=9999)

    @field_validator('person')
    @classmethod
    def person_name(cls,value):
        if value is not None:
            value=' '.join(value.split())
            if not value:
                raise ValueError('Нужно имя человека')
        return value

    @field_validator('timezone')
    @classmethod
    def zone(cls, value):
        if value is not None:
            try:
                ZoneInfo(value)
            except (KeyError, ValueError):
                raise ValueError('Неизвестный пояс IANA') from None
        return value

    @field_validator('start_time', 'end_time')
    @classmethod
    def local(cls, value):
        if value is not None and value.tzinfo:
            raise ValueError('Нужно местное время без смещения')
        return value

    @model_validator(mode='after')
    def valid(self):
        if self.kind == 'birthday':
            if not self.person:
                raise ValueError('Нужно имя человека')
            self.all_day = True
            if self.end_date is not None:
                if self.end_date!=self.start_date:
                    raise ValueError('День рождения занимает один день; празднование создайте отдельным событием')
                self.end_date=None
            if self.recurrence and (self.recurrence.frequency!='yearly' or len(self.recurrence.dates)!=1 or (self.recurrence.dates[0].month,self.recurrence.dates[0].day)!=(self.start_date.month,self.start_date.day)):
                raise ValueError('День рождения повторяется ежегодно в исходную дату')
        if self.all_day and (self.start_time or self.end_time):
            raise ValueError('У события на весь день нет времени')
        if not self.all_day and not self.start_time:
            raise ValueError('Уточните час события либо укажите «на весь день»')
        if self.end_time and not self.end_date:
            self.end_date = self.start_date
        if self.end_date and (self.end_date < self.start_date or
                (self.end_date == self.start_date and self.end_time and self.start_time and self.end_time <= self.start_time)):
            raise ValueError('Окончание должно быть позже начала')
        if self.recurrence and self.recurrence.start_date != self.start_date:
            raise ValueError('Начало повтора должно совпадать с датой события')
        return self


class ReminderRuleData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['calendar', 'interval', 'fixed']
    offset: int = Field(default=0, ge=0, le=315360000, description='calendar: дни, interval: секунды')
    due_time: time | None = None
    fixed_due_at: AwareDatetime | None = None

    @field_validator('due_time')
    @classmethod
    def local(cls, value):
        if value is not None and value.tzinfo:
            raise ValueError('Нужно местное время без смещения')
        return value

    @model_validator(mode='after')
    def valid(self):
        if (self.kind == 'fixed') != (self.fixed_due_at is not None):
            raise ValueError('Конкретный срок нужен только для fixed')
        if self.kind != 'calendar' and self.due_time is not None:
            raise ValueError('Время уведомлений относится только к календарному отступу')
        if self.kind == 'calendar' and self.offset > 3660:
            raise ValueError('Календарный отступ не больше 3660 дней')
        return self


class Events:
    def __init__(self, store):
        self.store, self.db = store, store.db

    def event(self, identity):
        row = self.db.execute('SELECT * FROM events WHERE id=? AND deleted=0', (identity,)).fetchone()
        if not row:
            raise EventError('event_not_found', 'Событие не найдено или удалено.')
        return dict(row)

    def view(self, identity, offset=0, limit=10):
        result=self.event(identity)
        segments=self.db.execute('SELECT * FROM event_segments WHERE event_id=? ORDER BY from_position DESC,id DESC LIMIT ? OFFSET ?',(identity,limit,offset)).fetchall()
        result['segments']=[dict(r) for r in reversed(segments)]
        # A closed empty interval is an archived version, not a usable rule.
        # Keep its database row and receipts, but do not repeat it in every read.
        visible_rules = ' AND active=1 AND (to_position IS NULL OR to_position>from_position)'
        result['rules']=[dict(r) for r in self.db.execute('SELECT * FROM event_reminder_rules WHERE event_id=?'+visible_rules+' ORDER BY id DESC LIMIT ? OFFSET ?',(identity,limit,offset))]
        for name,table,extra in (('segments','event_segments',''),('rules','event_reminder_rules',visible_rules)):
            total=self.db.execute('SELECT count(*) FROM '+table+' WHERE event_id=?'+extra,(identity,)).fetchone()[0]
            result[name+'_total']=total
            result['next_'+name+'_offset']=offset+len(result[name]) if offset+len(result[name])<total else None
        result['incomplete']=any(result['next_'+name+'_offset'] is not None for name in ('segments','rules'))
        for segment in result['segments']:
            if len(segment['content'])>300:
                segment['content_length']=len(segment['content'])
                segment['content']=segment['content'][:300]
                segment['content_incomplete']=True
        return result

    def notification_page(self, occurrence_id, offset=0, limit=20):
        return [dict(r) for r in self.db.execute('SELECT id,due_at,planned_due_at,status,reason,event_rule_id,is_snooze,parent_reminder_id FROM reminders WHERE event_occurrence_id=? ORDER BY due_at,id LIMIT ? OFFSET ?',(occurrence_id,limit,offset))]

    def create(self, data, now, source):
        zone = data.timezone or load_profile(self.store).timezone
        if data.kind == 'birthday':
            matches = [r for r in self.db.execute("SELECT * FROM events WHERE kind='birthday' AND deleted=0") if person_key(r['person'])==person_key(data.person)]
            exact = [r for r in matches if (r['birth_month'],r['birth_day']) == (data.start_date.month,data.start_date.day)
                     and (data.birth_year is None or r['birth_year'] == data.birth_year)]
            if len(matches) == 1 and exact:
                return {**self.view(exact[0]['id']), 'warnings': ['Такой день рождения уже сохранён.']}
            if matches:
                raise EventError('birthday_ambiguous', 'Имя уже встречается. Уточните: исправить дату или добавить другого человека с различимым именем.')
            # Birth year is metadata. Start generation in this calendar year.
            today = datetime.fromtimestamp(now, ZoneInfo(zone)).date()
            month, day = data.start_date.month, data.start_date.day
            start = date(today.year, month, min(day, calendar.monthrange(today.year, month)[1]))
            rule = Rule(frequency='yearly', start_date=start, dates=[{'month':month,'day':day}],count=data.recurrence.count if data.recurrence else None,until=data.recurrence.until if data.recurrence else None)
        else:
            start, rule = data.start_date, data.recurrence
        identity = self.db.execute('''INSERT INTO events(kind,content,timezone,source_message_id,created_at,person,birth_month,birth_day,birth_year)
                VALUES(?,?,?,?,?,?,?,?,?)''', (data.kind,data.content,zone,source,now,data.person,
                data.start_date.month if data.kind=='birthday' else None, data.start_date.day if data.kind=='birthday' else None,data.birth_year)).lastrowid
        self._segment(identity, 0, start, data, rule)
        self.materialize(identity, now)
        warnings=self.refresh_digest(now) if data.kind=='birthday' else []
        return {**self.view(identity),**({'warnings':warnings} if warnings else {})}

    def _segment(self, identity, position, start, data, rule):
        return self.db.execute('''INSERT INTO event_segments(event_id,from_position,start_date,start_time,end_date,end_time,all_day,rule,content)
            VALUES(?,?,?,?,?,?,?,?,?)''', (identity,position,start.isoformat(),data.start_time.isoformat() if data.start_time else None,
            data.end_date.isoformat() if data.end_date else None,data.end_time.isoformat() if data.end_time else None,
            int(data.all_day),rule.model_dump_json() if rule else None,data.content)).lastrowid

    def materialize(self, identity, now, through=None):
        event = self.event(identity)
        zone = ZoneInfo(event['timezone'])
        horizon = through or (datetime.fromtimestamp(now, zone).date() + timedelta(days=400))
        for seg in self.db.execute('SELECT * FROM event_segments WHERE event_id=? ORDER BY from_position', (identity,)).fetchall():
            start = date.fromisoformat(seg['start_date'])
            rule = Rule.model_validate_json(seg['rule']) if seg['rule'] else None
            dates = iter_dates(rule, through=horizon) if rule else [(0,start)]
            # Global positions continue across segments; count is the original total budget.
            for index, day in dates:
                position = seg['from_position'] + index
                if seg['to_position'] is not None and position >= seg['to_position']:
                    break
                starts, expires, end_day = self._moments(day, seg, zone)
                self.db.execute('''INSERT OR IGNORE INTO event_occurrences(event_id,segment_id,position,original_date,
                    start_date,start_time,end_date,end_time,starts_at,expires_at,content) VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                    (identity,seg['id'],position,day.isoformat(),day.isoformat(),seg['start_time'],end_day,seg['end_time'],starts,expires,seg['content']))

    @staticmethod
    def _moments(day, seg, zone):
        end_day = day + (date.fromisoformat(seg['end_date']) - date.fromisoformat(seg['start_date'])) if seg['end_date'] else day
        starts = local_timestamp(day, time.fromisoformat(seg['start_time']) if seg['start_time'] else time(), zone)
        expires = local_timestamp(end_day+timedelta(days=1),time(),zone) if seg['all_day'] else starts
        return starts, expires, end_day.isoformat() if seg['end_date'] else None

    def occurrence(self, identity, occurrence_id):
        self.event(identity)
        row = self.db.execute('SELECT * FROM event_occurrences WHERE id=? AND event_id=?', (occurrence_id,identity)).fetchone()
        if not row:
            raise EventError('occurrence_not_found', 'Экземпляр события не найден. Сначала прочитайте календарь.')
        return dict(row)

    def _future_position(self, identity, minimum, now, *, regular_only=False, active_only=False):
        """Find the next planned position, independent of the lookahead cache."""
        event = self.event(identity)
        zone = ZoneInfo(event['timezone'])
        cached = {r['position']: r for r in self.db.execute(
            'SELECT * FROM event_occurrences WHERE event_id=? AND position>=?', (identity, minimum))}
        best = min((p for p, row in cached.items() if row['expires_at'] > now
                    and not (regular_only and row['manual']) and not (active_only and row['status'] != 'active')), default=None)
        materialize_through = None
        for seg in self.db.execute('SELECT * FROM event_segments WHERE event_id=? ORDER BY from_position', (identity,)).fetchall():
            if seg['to_position'] is not None and seg['to_position'] <= minimum:
                continue
            dates = iter_dates(Rule.model_validate_json(seg['rule'])) if seg['rule'] else [(0, date.fromisoformat(seg['start_date']))]
            for index, day in dates:
                position = seg['from_position'] + index
                if (seg['to_position'] is not None and position >= seg['to_position']) or (best is not None and position >= best):
                    break
                if position < minimum:
                    continue
                row = cached.get(position)
                if row and ((regular_only and row['manual']) or (active_only and row['status'] != 'active')):
                    continue
                expires = row['expires_at'] if row else self._moments(day, seg, zone)[1]
                if expires > now:
                    best = position
                    materialize_through = day if row is None else None
                    break
        if materialize_through is not None:
            self.materialize(identity, now, through=materialize_through)
        return best

    def next_occurrence(self, identity, now):
        # An early position may be manually moved far past the next regular date.
        self._future_position(identity, 0, now, regular_only=True, active_only=True)
        self.generate(now, identity)
        row = self.db.execute("""SELECT id,position,start_date,start_time,end_date,end_time,status
            FROM event_occurrences WHERE event_id=? AND status='active' AND expires_at>?
            ORDER BY starts_at,id LIMIT 1""", (identity, now)).fetchone()
        return dict(row) if row else None

    def _has_all_day(self, identity, start):
        cached = {r['position']: r for r in self.db.execute(
            'SELECT * FROM event_occurrences WHERE event_id=? AND position>=?', (identity, start))}
        if any(r['status'] == 'active' and r['start_time'] is None for r in cached.values()):
            return True
        for seg in self.db.execute('SELECT * FROM event_segments WHERE event_id=? AND all_day=1', (identity,)):
            if seg['to_position'] is not None and seg['to_position'] <= start:
                continue
            dates = iter_dates(Rule.model_validate_json(seg['rule'])) if seg['rule'] else [(0, date.fromisoformat(seg['start_date']))]
            for index, day in dates:
                position = seg['from_position'] + index
                if seg['to_position'] is not None and position >= seg['to_position']:
                    break
                if position >= start and position not in cached:
                    return True
        return False

    def _interval_on_plan(self, identity, start, recurrence, day, affected):
        rules = self.db.execute("""SELECT r.*,o.position AS target_position FROM event_reminder_rules r
            LEFT JOIN event_occurrences o ON o.id=r.occurrence_id
            WHERE r.event_id=? AND r.active=1 AND r.kind='interval'""", (identity,)).fetchall()
        if not rules:
            return False
        cached = {r['position']: r for r in self.db.execute('SELECT * FROM event_occurrences WHERE event_id=?', (identity,))}
        affected_ids = {r['id'] for r in affected}
        dates = iter_dates(recurrence) if recurrence else [(0, day)]
        for index, _ in dates:
            position = start + index
            possible = [r for r in rules if (r['to_position'] is None or position < r['to_position'])
                        and (r['target_position'] is None or position <= r['target_position'])]
            if not possible:
                break
            occ = cached.get(position)
            if occ and (occ['manual'] or (occ['status'] != 'active' and occ['cancel_reason'] != 'plan_changed') or occ['id'] not in affected_ids):
                continue
            projected = occ if occ else {'id': None, 'position': position}
            if any(self.rule_applies(r, projected) for r in possible):
                return True
        return False

    def _guard(self, ids):
        for identity in ids:
            # Includes sent chunks of a partially delivered multi-part message.
            group = self.db.execute('SELECT group_id FROM notification_members WHERE reminder_id=?',(identity,)).fetchone()
            members = [r[0] for r in self.db.execute('SELECT reminder_id FROM notification_members WHERE group_id=?',(group[0],))] if group else [identity]
            for member in members:
                if self.db.execute('SELECT 1 FROM outbox WHERE reminder_id=? AND (delivery_started=1 OR sent_at IS NOT NULL)',(member,)).fetchone():
                    raise EventError('delivery_in_progress','Отправка уже началась; результат может быть неизвестен. Изменение не выполнено.')

    def _invalidate(self, rows, reason, *, status='cancelled'):
        ids = [r['id'] for r in rows]
        self._guard(ids)
        groups = set()
        for identity in ids:
            group = self.db.execute('SELECT group_id FROM notification_members WHERE reminder_id=?',(identity,)).fetchone()
            if group:
                groups.add(group[0])
        # A group may contain unaffected rules: dissolve the unstarted group and requeue survivors.
        for group in groups:
            leader = self.db.execute('SELECT leader_id FROM notification_groups WHERE id=?',(group,)).fetchone()[0]
            self.db.execute('DELETE FROM outbox WHERE reminder_id=? AND sent_at IS NULL',(leader,))
            self.db.execute("UPDATE reminders SET status='pending' WHERE id IN (SELECT reminder_id FROM notification_members WHERE group_id=?) AND status='queued'",(group,))
            self.db.execute('DELETE FROM notification_members WHERE group_id=?',(group,))
            self.db.execute('DELETE FROM notification_groups WHERE id=?',(group,))
        for identity in ids:
            self.db.execute('DELETE FROM outbox WHERE reminder_id=? AND sent_at IS NULL',(identity,))
            self.db.execute('UPDATE reminders SET status=?,reason=? WHERE id=?',(status,reason,identity))

    def add_rule(self, identity, data, now, *, scope, occurrence_id=None, from_position=None, replace_rule_id=None):
        event = self.event(identity)
        occurrence = self.occurrence(identity,occurrence_id) if occurrence_id else None
        if scope == 'one' and occurrence is None:
            raise EventError('scope_required','Для изменения одного экземпляра выберите его в календаре.')
        if scope == 'future' and from_position is None:
            raise EventError('scope_required','Нужна начальная позиция будущей части серии.')
        if data.kind == 'fixed' and (scope!='one' or not occurrence):
            raise EventError('fixed_occurrence_required','Конкретная дата уведомления привязывается к одному экземпляру.')
        start = occurrence['position'] if scope=='one' else from_position
        all_day = occurrence['start_time'] is None if scope=='one' else self._has_all_day(identity, start)
        if data.kind=='interval' and all_day:
            raise EventError('all_day_reference_required','Для точного интервала у события на весь день уточните точку отсчёта; используйте конкретный срок.')
        start = occurrence['position'] if scope=='one' else from_position
        if replace_rule_id is not None:
            old = self.db.execute('SELECT * FROM event_reminder_rules WHERE id=? AND event_id=? AND active=1',(replace_rule_id,identity)).fetchone()
            if not old:
                raise EventError('rule_not_found','Правило для замены не найдено.')
            if (old['occurrence_id'] is not None and (scope!='one' or old['occurrence_id']!=occurrence_id)) or start<old['from_position'] or (old['to_position'] is not None and start>=old['to_position']):
                raise EventError('rule_scope_mismatch','Выбранное правило не действует на этот экземпляр или участок.')
            if scope == 'one' and not self.rule_applies(old, occurrence):
                raise EventError('rule_scope_mismatch','Выбранное правило уже заменено или отключено для этого экземпляра.')
            if scope=='one':
                self.db.execute('INSERT OR IGNORE INTO event_rule_exceptions VALUES(?,?)',(occurrence_id,replace_rule_id))
            else:
                self.db.execute('UPDATE event_reminder_rules SET to_position=? WHERE id=?',(start,replace_rule_id))
            rows = self.db.execute("SELECT * FROM reminders WHERE event_rule_id=? AND status IN ('pending','queued') AND event_occurrence_id IN (SELECT id FROM event_occurrences WHERE event_id=? AND position>=?)" + (' AND event_occurrence_id=?' if scope=='one' else ''),
                (replace_rule_id,identity,start,*([occurrence_id] if scope=='one' else []))).fetchall()
            self._invalidate(rows,'rule_replaced')
        rule_id = self.db.execute('''INSERT INTO event_reminder_rules(event_id,occurrence_id,from_position,kind,offset,time_mode,due_time,fixed_due_at,created_at)
            VALUES(?,?,?,?,?,?,?,?,?)''',(identity,occurrence_id if scope=='one' else None,start,data.kind,data.offset,
            'explicit' if data.due_time else 'default',data.due_time.isoformat() if data.due_time else None,
            data.fixed_due_at.timestamp() if data.fixed_due_at else None,now)).lastrowid
        self.db.execute('UPDATE events SET revision=revision+1 WHERE id=?',(identity,))
        warnings = self.generate(now, identity)
        return {'event':self.view(identity),'rule_id':rule_id,'warnings':warnings}

    def rule_applies(self,rule,occ):
        return (rule['active'] and occ['position']>=rule['from_position'] and
                (rule['to_position'] is None or occ['position']<rule['to_position']) and
                (rule['occurrence_id'] is None or rule['occurrence_id']==occ['id']) and
                not self.db.execute('SELECT 1 FROM event_rule_exceptions WHERE occurrence_id=? AND rule_id=?',(occ['id'],rule['id'])).fetchone())

    @staticmethod
    def notification_content(event,occ):
        content=occ['content']
        if event['kind']=='birthday' and event['birth_year']:
            content+=f" — {date.fromisoformat(occ['start_date']).year-event['birth_year']} лет"
        return content

    def _due(self, rule, occ, zone):
        if rule['kind']=='fixed':
            return rule['fixed_due_at']
        if rule['kind']=='interval':
            return occ['starts_at']-rule['offset']
        day = date.fromisoformat(occ['start_date'])-timedelta(days=rule['offset'])
        at = rule['due_time'] if rule['time_mode']=='explicit' else load_profile(self.store).notification_time
        return local_timestamp(day,time.fromisoformat(at),zone)

    def generate(self, now, identity=None, *, through=None):
        warnings=[]
        events = [self.event(identity)] if identity else [dict(r) for r in self.db.execute('SELECT * FROM events WHERE deleted=0')]
        for event in events:
            rules = self.db.execute('SELECT * FROM event_reminder_rules WHERE event_id=? AND active=1',(event['id'],)).fetchall()
            max_offset = max([r['offset'] if r['kind']=='calendar' else r['offset']//86400 if r['kind']=='interval' else 0 for r in rules] or [0])
            horizon=datetime.fromtimestamp(now,ZoneInfo(event['timezone'])).date()+timedelta(days=400+min(max_offset,3660))
            self.materialize(event['id'],now,max(horizon,through) if through else horizon)
            zone = ZoneInfo(event['timezone'])
            occurrences = self.db.execute("SELECT * FROM event_occurrences WHERE event_id=? AND status='active'",(event['id'],)).fetchall()
            for occ in occurrences:
                for rule in rules:
                    if occ['position']<rule['from_position'] or (rule['to_position'] is not None and occ['position']>=rule['to_position']):
                        continue
                    if rule['occurrence_id'] is not None and rule['occurrence_id']!=occ['id']:
                        continue
                    if self.db.execute('SELECT 1 FROM event_rule_exceptions WHERE occurrence_id=? AND rule_id=?',(occ['id'],rule['id'])).fetchone():
                        continue
                    if self.db.execute('SELECT 1 FROM reminders WHERE event_rule_id=? AND event_occurrence_id=? AND is_snooze=0',(rule['id'],occ['id'])).fetchone():
                        continue
                    due = self._due(rule,occ,zone)
                    # Rules created after a deadline never fire retroactively.
                    reason = 'created_after_deadline' if due < rule['created_at'] else 'event_passed' if occ['expires_at']<=now else 'after_event' if due>=occ['expires_at'] else None
                    status = 'cancelled' if reason else 'pending'
                    content = self.notification_content(event,occ)
                    self.db.execute('''INSERT INTO reminders(content,due_at,planned_due_at,created_at,source_message_id,event_rule_id,event_occurrence_id,status,reason)
                        VALUES(?,?,?,?,?,?,?,?,?)''',(content,due,due,now,event['source_message_id'],rule['id'],occ['id'],status,reason))
                    if reason=='after_event':
                        warnings.append('Срок уведомления после начала события; уведомление отменено.')
                    if reason=='created_after_deadline' and occ['expires_at']>now:
                        warnings.append('Срок уведомления уже прошёл; для этого экземпляра он пропущен. Правило сохранено.')
        return list(dict.fromkeys(warnings))

    def change(self, identity, operation, now, *, scope, occurrence_id=None, from_position=None, data=None, rule_id=None):
        event=self.event(identity)
        occurrence=self.occurrence(identity,occurrence_id) if occurrence_id else None
        if scope=='one' and occurrence is None:
            raise EventError('scope_required','Выберите один экземпляр в календаре.')
        if scope=='future' and from_position is None:
            raise EventError('scope_required','Укажите начало будущей части серии.')
        position=occurrence['position'] if scope=='one' else from_position
        if operation == 'disable_rule':
            rule = self.db.execute('SELECT * FROM event_reminder_rules WHERE id=? AND event_id=? AND active=1',
                                   (rule_id, identity)).fetchone()
            if rule is None:
                raise EventError('rule_not_found', 'Выбранное правило этого события не найдено.')
            if rule['occurrence_id'] is not None and (scope != 'one' or rule['occurrence_id'] != occurrence_id):
                raise EventError('rule_scope_mismatch', 'Правило относится к одному экземпляру; выберите этот экземпляр.')
            if scope == 'one':
                applicable = self.rule_applies(rule, occurrence)
            else:
                position = max(position, rule['from_position'])
                applicable = rule['to_position'] is None or position < rule['to_position']
            if not applicable:
                raise EventError('rule_scope_mismatch', 'Правило уже отключено или не действует на выбранном участке.')
            rows = self.db.execute('''SELECT r.* FROM reminders r JOIN event_occurrences o ON o.id=r.event_occurrence_id
                WHERE r.event_rule_id=? AND r.status IN ('pending','queued') AND o.event_id=? AND ''' +
                ('o.id=?' if scope == 'one' else 'o.position>=?'),
                (rule_id, identity, occurrence_id if scope == 'one' else position)).fetchall()
            self._invalidate(rows, 'rule_disabled')
            if scope == 'one':
                self.db.execute('INSERT OR IGNORE INTO event_rule_exceptions VALUES(?,?)', (occurrence_id, rule_id))
            else:
                self.db.execute('UPDATE event_reminder_rules SET to_position=? WHERE id=?', (position, rule_id))
            self.db.execute('UPDATE events SET revision=revision+1 WHERE id=?', (identity,))
            saved_rule = dict(self.db.execute('SELECT * FROM event_reminder_rules WHERE id=?', (rule_id,)).fetchone())
            return {'event': self.view(identity), 'operation': operation, 'disabled_rule': saved_rule,
                    'occurrence': occurrence, 'warnings': []}
        if scope=='future' and operation in ('move','cancel'):
            self.materialize(identity,now)
            future=self._future_position(identity,position,now,regular_only=operation=='move')
            if future is not None:
                position=future
            elif operation=='move':
                raise EventError('series_exhausted','У серии нет будущих экземпляров.')
            else:
                position=max(position,1+self.db.execute('SELECT coalesce(max(position),-1) FROM event_occurrences WHERE event_id=?',(identity,)).fetchone()[0])
        condition='event_id=? AND '+('id=?' if scope=='one' else 'position>=?')
        args=(identity,occurrence_id if scope=='one' else position)
        if scope=='future' and operation in ('move','cancel'):
            condition+=' AND expires_at>?'
            args=(*args,now)
        affected=self.db.execute('SELECT * FROM event_occurrences WHERE '+condition,args).fetchall()
        ids=[r['id'] for r in affected]
        rows=[r for r in self.db.execute("SELECT * FROM reminders WHERE event_occurrence_id IS NOT NULL AND (status IN ('pending','queued') OR (? AND status='cancelled' AND is_snooze=0 AND (reason IN ('created_after_deadline','rescheduled_into_past','profile_time_in_past','event_passed','superseded','plan_changed') OR (reason='after_event' AND event_rule_id IN (SELECT id FROM event_reminder_rules WHERE kind IN ('calendar','interval'))))))",(operation=='move',)) if r['event_occurrence_id'] in ids]
        self._guard([r['id'] for r in rows])
        warnings=[]
        self.db.execute('UPDATE events SET revision=revision+1 WHERE id=?',(identity,))
        if operation in ('cancel','delete','disable'):
            self._invalidate(rows,'event_'+operation)
            if operation=='delete':
                # Deletion and disabling always cover the whole identity, including snoozes.
                remaining=self.db.execute("SELECT r.* FROM reminders r JOIN event_occurrences o ON o.id=r.event_occurrence_id WHERE o.event_id=? AND r.status IN ('pending','queued')",(identity,)).fetchall()
                self._invalidate(remaining,'event_deleted')
                self.db.execute('UPDATE events SET deleted=1 WHERE id=?',(identity,))
                self.db.execute('UPDATE event_reminder_rules SET active=0 WHERE event_id=?',(identity,))
                if event['kind']=='birthday':
                    warnings.extend(self.refresh_digest(now))
                return {'content':event['content'],'operation':operation,'warnings':warnings}
            if operation=='disable':
                if scope=='one':
                    self.db.execute('INSERT OR IGNORE INTO event_rule_exceptions SELECT ?,id FROM event_reminder_rules WHERE event_id=?',(occurrence_id,identity))
                else:
                    self.db.execute('UPDATE event_reminder_rules SET to_position=max(from_position,?) WHERE event_id=? AND (to_position IS NULL OR to_position>?)',(position,identity,position))
            else:
                self.db.execute("UPDATE event_occurrences SET status='cancelled',cancel_reason='user_cancelled' WHERE "+condition,args)
                if scope=='future':
                    self.db.execute('UPDATE event_segments SET to_position=max(from_position,?) WHERE event_id=? AND (to_position IS NULL OR to_position>?)',(position,identity,position))
            return {'event':self.view(identity),'occurrence':self.occurrence(identity,occurrence_id) if occurrence_id else None,'operation':operation,'warnings':warnings}
        if data is None:
            raise EventError('event_data_required','Нужны новые дата и время.')
        if data.timezone and data.timezone!=event['timezone']:
            raise EventError('timezone_pinned','Пояс события закреплён при создании.')
        if event['kind']=='birthday':
            if scope!='future' or data.kind!='birthday':
                raise EventError('birthday_correction_required','Для исправления даты дня рождения измените будущую серию с kind=birthday. Празднование в другой день создайте отдельным событием.')
            person=data.person or event['person']
            others=[r for r in self.db.execute("SELECT id,person FROM events WHERE kind='birthday' AND deleted=0 AND id!=?",(identity,)) if person_key(r['person'])==person_key(person)]
            if others:
                raise EventError('birthday_ambiguous','Уточните человека: это имя уже используется другим событием.')
            self.db.execute('UPDATE events SET person=?,birth_month=?,birth_day=?,birth_year=? WHERE id=?',
                (person,data.start_date.month,data.start_date.day,data.birth_year if 'birth_year' in data.model_fields_set else event['birth_year'],identity))
            today = datetime.fromtimestamp(now, ZoneInfo(event['timezone'])).date()
            anchor = self.db.execute('SELECT start_date FROM event_occurrences WHERE event_id=? AND position=?', (identity, position)).fetchone()
            year = max(today.year, date.fromisoformat(anchor['start_date']).year if anchor else today.year)
            month, day = data.start_date.month, data.start_date.day
            start = date(year, month, min(day, calendar.monthrange(year, month)[1]))
            if start < today:
                year += 1
                start = date(year, month, min(day, calendar.monthrange(year, month)[1]))
            limits = {key: getattr(data.recurrence, key) for key in ('count', 'until')
                      if data.recurrence and key in data.recurrence.model_fields_set}
            rule = Rule(frequency='yearly', start_date=start, dates=[{'month': month, 'day': day}], **limits)
            data = data.model_copy(update={'start_date': start, 'recurrence': rule})
        if scope=='one':
            if data.all_day and any(self.rule_applies(r, occurrence) for r in self.db.execute(
                    "SELECT * FROM event_reminder_rules WHERE event_id=? AND active=1 AND kind='interval'", (identity,))):
                raise EventError('all_day_reference_required','Перед переводом на весь день уточните точку отсчёта точного интервала или замените его календарным правилом.')
            seg=self.db.execute('SELECT * FROM event_segments WHERE id=?',(occurrence['segment_id'],)).fetchone()
            values={'start_date':data.start_date.isoformat(),'start_time':data.start_time.isoformat() if data.start_time else None,
                    'end_date':data.end_date.isoformat() if data.end_date else None,'end_time':data.end_time.isoformat() if data.end_time else None,'all_day':int(data.all_day)}
            starts,expires,end_day=self._moments(data.start_date,values,ZoneInfo(event['timezone']))
            self.db.execute('''UPDATE event_occurrences SET start_date=?,start_time=?,end_date=?,end_time=?,starts_at=?,expires_at=?,content=?,manual=1 WHERE id=?''',
                (data.start_date.isoformat(),values['start_time'],end_day,values['end_time'],starts,expires,data.content,occurrence_id))
        else:
            # Preserve manual exceptions and IDs; only replace generated dates at the same positions.
            old=self.db.execute('SELECT * FROM event_segments WHERE event_id=? AND from_position<=? AND (to_position IS NULL OR to_position>?) ORDER BY from_position DESC LIMIT 1',(identity,position,position)).fetchone()
            if not old:
                raise EventError('series_closed','Эта часть серии уже закрыта.')
            original=Rule.model_validate_json(old['rule']) if old['rule'] else None
            rule=data.recurrence or (original.model_copy(update={'start_date':data.start_date}) if original else None)
            if rule:
                values=rule.model_dump(mode='json')
                explicit=data.recurrence.model_fields_set if data.recurrence else set()
                if original and (not explicit.intersection({'count','until'}) or original.count is not None and rule.count==original.count):
                    values['count']=original.count-(position-old['from_position']) if original.count else None
                    values['until']=original.until
                if values.get('count') is not None and values['count']<=0 or values.get('until') and data.start_date>date.fromisoformat(str(values['until'])):
                    raise EventError('series_exhausted','Новый срок за пределами ограничения серии.')
                rule=Rule.model_validate(values)
            if data.all_day and self._interval_on_plan(identity, position, rule, data.start_date, affected):
                raise EventError('all_day_reference_required','Перед переводом на весь день уточните точку отсчёта точного интервала или замените его календарным правилом.')
            self.db.execute('UPDATE event_segments SET to_position=max(from_position,?) WHERE event_id=? AND (to_position IS NULL OR to_position>?)',(position,identity,position))
            self.db.execute('UPDATE events SET content=? WHERE id=?',(data.content,identity))
            seg_id=self._segment(identity,position,data.start_date,data,rule)
            segment=dict(self.db.execute('SELECT * FROM event_segments WHERE id=?',(seg_id,)).fetchone())
            # Reposition every already materialized ID, even if its new date is beyond the normal lookahead.
            length=max([r['position'] for r in affected]+[position])-position+1
            dates=islice(iter_dates(rule),length) if rule else [(0,data.start_date)]
            mapping={position+i:day for i,day in dates}
            for occ in affected:
                if occ['manual']:
                    continue
                day=mapping.get(occ['position'])
                if day is None:
                    self.db.execute("UPDATE event_occurrences SET cancel_reason=CASE WHEN status='active' THEN 'plan_changed' ELSE cancel_reason END,status='cancelled' WHERE id=?",(occ['id'],))
                    continue
                starts,expires,end_day=self._moments(day,segment,ZoneInfo(event['timezone']))
                self.db.execute('''UPDATE event_occurrences SET segment_id=?,start_date=?,start_time=?,end_date=?,end_time=?,starts_at=?,expires_at=?,content=?,
                    status=CASE WHEN cancel_reason='plan_changed' THEN 'active' ELSE status END,
                    cancel_reason=CASE WHEN cancel_reason='plan_changed' THEN NULL ELSE cancel_reason END WHERE id=?''',
                    (seg_id,day.isoformat(),segment['start_time'],end_day,segment['end_time'],starts,expires,data.content,occ['id']))
        event=self.event(identity)
        for row in rows:
            occ=self.db.execute('SELECT * FROM event_occurrences WHERE id=?',(row['event_occurrence_id'],)).fetchone()
            rule=self.db.execute('SELECT * FROM event_reminder_rules WHERE id=?',(row['event_rule_id'],)).fetchone()
            if not rule['active'] or (rule['to_position'] is not None and occ['position']>=rule['to_position']) or self.db.execute('SELECT 1 FROM event_rule_exceptions WHERE occurrence_id=? AND rule_id=?',(occ['id'],rule['id'])).fetchone():
                continue
            due=row['due_at'] if row['is_snooze'] or rule['kind']=='fixed' else self._due(rule,occ,ZoneInfo(event['timezone']))
            reason=('plan_changed' if occ['cancel_reason']=='plan_changed' else 'event_cancelled') if occ['status']=='cancelled' else 'after_event' if due>=occ['expires_at'] else 'rescheduled_into_past' if due<=now and not row['is_snooze'] else None
            self._invalidate([row],reason,status='cancelled' if reason else 'pending')
            self.db.execute('UPDATE reminders SET due_at=?,planned_due_at=CASE WHEN is_snooze=0 THEN ? ELSE planned_due_at END,content=? WHERE id=?',(due,due,self.notification_content(event,occ),row['id']))
            if reason:
                warnings.append('Уведомление отменено: срок после начала события или уже прошёл.')
        self.generate(now,identity)
        if event['kind']=='birthday':
            warnings.extend(self.refresh_digest(now))
        return {'event':self.view(identity),'occurrence':self.occurrence(identity,occurrence_id) if occurrence_id else None,'operation':operation,'warnings':list(dict.fromkeys(warnings))}

    def recover(self, now):
        rows=self.db.execute("SELECT r.*,o.expires_at FROM reminders r JOIN event_occurrences o ON o.id=r.event_occurrence_id WHERE r.status IN ('pending','queued') AND r.due_at<=? ORDER BY r.event_occurrence_id,r.due_at DESC,r.id DESC",(now,)).fetchall()
        latest={}
        for row in rows:
            reason='event_passed' if row['expires_at']<=now else 'superseded' if row['event_occurrence_id'] in latest and latest[row['event_occurrence_id']]!=row['due_at'] else None
            latest.setdefault(row['event_occurrence_id'],row['due_at'])
            if reason:
                try:
                    self._invalidate([row],reason)
                except EventError as exc:
                    if exc.code!='delivery_in_progress':
                        raise

    def calendar(self, start, end, now):
        if end<start or (end-start).days>366:
            raise EventError('calendar_range','Диапазон календаря должен быть от 0 до 366 дней.')
        self.generate(now,through=end)
        result=[]
        for row in self.db.execute('''SELECT o.*,e.timezone,e.kind FROM event_occurrences o JOIN events e ON e.id=o.event_id
            WHERE e.deleted=0 AND o.start_date<=? AND coalesce(o.end_date,o.start_date)>=? ORDER BY o.start_date,o.start_time,o.id''',(end.isoformat(),start.isoformat())):
            item=dict(row)
            item['notifications']=self.notification_page(row['id'])
            item['notifications_total']=self.db.execute('SELECT count(*) FROM reminders WHERE event_occurrence_id=?',(row['id'],)).fetchone()[0]
            item['notifications_incomplete']=len(item['notifications'])<item['notifications_total']
            item['next_notification_offset']=len(item['notifications']) if item['notifications_incomplete'] else None
            result.append(item)
        # Concrete notifications are authoritative for moves, snoozes, skips and history.
        # Only positions with no concrete record are projected from the current rule.
        for row in self.db.execute("SELECT * FROM schedules"):
            zone_name=row['timezone'] if row['kind']=='recurring' or row['local_date'] else load_profile(self.store).timezone
            zone=ZoneInfo(zone_name)
            concrete=self.db.execute('SELECT * FROM reminders WHERE schedule_id=? OR standalone_id=? ORDER BY due_at,id',(row['id'],row['id'])).fetchall()
            occupied=set()
            for reminder in concrete:
                if not reminder['is_snooze'] and reminder['occurrence_at'] is not None and (reminder['status']!='cancelled' or reminder['reason'] not in (None,'schedule_changed')):
                    occupied.add(datetime.fromtimestamp(reminder['occurrence_at'],zone).date())
                if start<=datetime.fromtimestamp(reminder['due_at'],zone).date()<=end:
                    result.append({'type':'standalone','schedule_id':row['id'],'reminder_id':reminder['id'],'content':reminder['content'],
                                   'due_at':reminder['due_at'],'status':reminder['status'],'reason':reminder['reason'],'is_snooze':bool(reminder['is_snooze']),'timezone':zone_name})
            if row['kind']!='recurring' or row['status']!='active':
                continue
            rule=Rule.model_validate_json(row['rule'])
            at=row['due_time'] if row['time_mode']=='explicit' else load_profile(self.store).notification_time
            for _,day in iter_dates(rule,through=end):
                due=local_timestamp(day,time.fromisoformat(at),zone)
                if day>=start and day not in occupied and due>=row['next_due_at']:
                    result.append({'type':'standalone','schedule_id':row['id'],'content':row['content'],'due_at':due,'status':'planned','timezone':zone_name})
        return result

    def profile_changed(self, now):
        warnings=[]
        rows=self.db.execute("""SELECT r.*,o.* FROM reminders r JOIN event_occurrences o ON o.id=r.event_occurrence_id
            JOIN event_reminder_rules rule ON rule.id=r.event_rule_id
            JOIN events e ON e.id=o.event_id
            WHERE e.deleted=0 AND rule.active=1 AND rule.kind='calendar' AND rule.time_mode='default' AND r.is_snooze=0 AND (r.status IN ('pending','queued') OR (r.status='cancelled' AND r.reason IN ('created_after_deadline','profile_time_in_past','after_event','rescheduled_into_past')))""").fetchall()
        for row in rows:
            rule=self.db.execute('SELECT * FROM event_reminder_rules WHERE id=?',(row['event_rule_id'],)).fetchone()
            occ=self.db.execute('SELECT * FROM event_occurrences WHERE id=?',(row['event_occurrence_id'],)).fetchone()
            event=self.event(occ['event_id'])
            if not self.rule_applies(rule,occ) or occ['status']!='active' or occ['expires_at']<=now:
                continue
            due=self._due(rule,occ,ZoneInfo(event['timezone']))
            reason='profile_time_in_past' if due<=now else 'after_event' if due>=occ['expires_at'] else None
            self._invalidate([row],reason,status='cancelled' if reason else 'pending')
            self.db.execute('UPDATE reminders SET due_at=?,planned_due_at=? WHERE id=?',(due,due,row['id']))
            if reason:
                warnings.append('Новый срок уже прошёл или оказался после события; уведомление пропущено.')
        for row in self.db.execute("SELECT * FROM schedules WHERE kind='once' AND time_mode='default' AND status='active'").fetchall():
            due=local_timestamp(date.fromisoformat(row['local_date']),time.fromisoformat(load_profile(self.store).notification_time),ZoneInfo(row['timezone']))
            pending=self.db.execute("SELECT * FROM reminders WHERE standalone_id=? AND is_snooze=0 AND is_manual=0 AND (status IN ('pending','queued') OR (status='cancelled' AND reason IN ('profile_time_in_past','after_event')))",(row['id'],)).fetchall()
            for item in pending:
                self._invalidate([item],'profile_time_in_past' if due<=now else None,status='cancelled' if due<=now else 'pending')
                self.db.execute('UPDATE reminders SET due_at=?,planned_due_at=? WHERE id=?',(due,due,item['id']))
                if due<=now:
                    warnings.append('Новый срок самостоятельного уведомления уже прошёл; он пропущен.')
            if pending:
                self.db.execute('UPDATE schedules SET absolute_due_at=? WHERE id=?',(due,row['id']))
        from .recurrence import occurrence
        for row in self.db.execute("SELECT * FROM schedules WHERE kind='recurring' AND time_mode='default' AND status IN ('active','completed')").fetchall():
            rule=Rule.model_validate_json(row['rule'])
            pending=self.db.execute("SELECT * FROM reminders WHERE schedule_id=? AND is_snooze=0 AND is_manual=0 AND (status IN ('pending','queued') OR (status='cancelled' AND reason IN ('profile_time_in_past','after_event')))",(row['id'],)).fetchall()
            for item in pending:
                day=datetime.fromtimestamp(item['occurrence_at'],ZoneInfo(row['timezone'])).date()
                due=local_timestamp(day,time.fromisoformat(load_profile(self.store).notification_time),ZoneInfo(row['timezone']))
                self._invalidate([item],'profile_time_in_past' if due<=now else None,status='cancelled' if due<=now else 'pending')
                self.db.execute('UPDATE reminders SET due_at=?,planned_due_at=? WHERE id=?',(due,due,item['id']))
                if due<=now:
                    warnings.append('Новый срок самостоятельного уведомления уже прошёл; он пропущен.')
            at=load_profile(self.store).notification_time
            next_day=datetime.fromtimestamp(row['next_due_at'],ZoneInfo(row['timezone'])).date()
            changed_due=local_timestamp(next_day,time.fromisoformat(at),ZoneInfo(row['timezone']))
            if row['next_due_at']>now and changed_due<=now:
                warnings.append('Новый срок самостоятельного уведомления уже прошёл; он пропущен.')
            next_due=occurrence(rule,at,row['timezone'],now)
            if next_due is None:
                self.db.execute("UPDATE schedules SET status='completed' WHERE id=?",(row['id'],))
            else:
                self.db.execute("UPDATE schedules SET next_due_at=?,status='active' WHERE id=?",(next_due,row['id']))
        return list(dict.fromkeys(warnings))

    def digest(self, now, *, immediate=False):
        profile=load_profile(self.store)
        if not profile.birthday_digest:
            return
        zone=ZoneInfo(profile.timezone)
        today=datetime.fromtimestamp(now,zone).date()
        due=local_timestamp(today.replace(day=1),time.fromisoformat(profile.notification_time),zone)
        if now<due:
            return
        if self.db.execute("SELECT 1 FROM digest_runs WHERE kind='birthdays' AND year=? AND month=? AND state!='cancelled'",(today.year,today.month)).fetchone():
            return
        rows=self.db.execute("SELECT * FROM events WHERE kind='birthday' AND deleted=0 AND birth_month=? ORDER BY birth_day,person,id",(today.month,)).fetchall()
        key=f'digest:birthdays:{today.year}:{today.month}'
        if rows:
            lines=[f'Дни рождения: {today.month:02d}.{today.year}']
            for row in rows:
                day=min(row['birth_day'],calendar.monthrange(today.year,today.month)[1])
                age=f" — {today.year-row['birth_year']} лет" if row['birth_year'] else ''
                lines.append(f"{day:02d}.{today.month:02d}: {row['person']}{age}"+(' (уже прошёл)' if day<today.day else ''))
            self.store.queue_message(key,'\n'.join(lines),now=now)
        self.db.execute("INSERT INTO digest_runs VALUES('birthdays',?,?,?,?) ON CONFLICT(kind,year,month) DO UPDATE SET state=excluded.state,outbox_key=excluded.outbox_key",(today.year,today.month,'queued' if rows else 'empty',key if rows else None))

    def disable_digest(self):
        for row in self.db.execute("SELECT * FROM digest_runs WHERE state='queued'").fetchall():
            prefix=row['outbox_key']+':%'
            if self.db.execute('SELECT 1 FROM outbox WHERE key LIKE ? AND (delivery_started=1 OR sent_at IS NOT NULL)',(prefix,)).fetchone():
                raise EventError('delivery_in_progress','Отправка сводки уже началась; гарантированно отменить её нельзя.')
            self.db.execute('DELETE FROM outbox WHERE key LIKE ?',(prefix,))
            self.db.execute("UPDATE digest_runs SET state='cancelled' WHERE kind=? AND year=? AND month=?",(row['kind'],row['year'],row['month']))

    def recover_digests(self,now):
        today=datetime.fromtimestamp(now,ZoneInfo(load_profile(self.store).timezone)).date()
        for row in self.db.execute("SELECT * FROM digest_runs WHERE state='queued' AND (year!=? OR month!=?)",(today.year,today.month)).fetchall():
            prefix=row['outbox_key']+':%'
            if self.db.execute('SELECT 1 FROM outbox WHERE key LIKE ? AND (delivery_started=1 OR sent_at IS NOT NULL)',(prefix,)).fetchone():
                continue
            self.db.execute('DELETE FROM outbox WHERE key LIKE ?',(prefix,))
            self.db.execute("UPDATE digest_runs SET state='cancelled' WHERE kind=? AND year=? AND month=?",(row['kind'],row['year'],row['month']))

    def refresh_digest(self,now):
        """Refresh an unstarted current-month snapshot after birthday changes."""
        self.recover_digests(now)
        today=datetime.fromtimestamp(now,ZoneInfo(load_profile(self.store).timezone)).date()
        row=self.db.execute("SELECT * FROM digest_runs WHERE kind='birthdays' AND year=? AND month=? AND state='queued'",(today.year,today.month)).fetchone()
        if not row:
            return []
        prefix=row['outbox_key']+':%'
        if self.db.execute('SELECT 1 FROM outbox WHERE key LIKE ? AND (delivery_started=1 OR sent_at IS NOT NULL)',(prefix,)).fetchone():
            return ['Сводка уже отправляется; гарантированно изменить её сообщения нельзя.']
        self.db.execute('DELETE FROM outbox WHERE key LIKE ?',(prefix,))
        self.db.execute("UPDATE digest_runs SET state='cancelled' WHERE kind=? AND year=? AND month=?",(row['kind'],row['year'],row['month']))
        self.digest(now,immediate=True)
        return []

    def reschedule_digest(self,now):
        profile=load_profile(self.store)
        today=datetime.fromtimestamp(now,ZoneInfo(profile.timezone)).date()
        row=self.db.execute("SELECT * FROM digest_runs WHERE kind='birthdays' AND year=? AND month=? AND state='queued'",(today.year,today.month)).fetchone()
        if not row:
            return
        prefix=row['outbox_key']+':%'
        if self.db.execute('SELECT 1 FROM outbox WHERE key LIKE ? AND (delivery_started=1 OR sent_at IS NOT NULL)',(prefix,)).fetchone():
            raise EventError('delivery_in_progress','Отправка сводки уже началась; изменить её срок гарантированно нельзя.')
        due=local_timestamp(today.replace(day=1),time.fromisoformat(profile.notification_time),ZoneInfo(profile.timezone))
        self.db.execute('UPDATE outbox SET available_at=? WHERE key LIKE ? AND sent_at IS NULL',(max(now,due),prefix))
