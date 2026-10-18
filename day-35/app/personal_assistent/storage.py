import json
import sqlite3
from contextlib import contextmanager
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from .presentation import public_text
from .profile import timezone_name
from .search import INDEX_VERSION, index_values
from .recurrence import Rule, occurrence


def utcnow():
    return datetime.now(timezone.utc).timestamp()


def dump(value):
    return json.dumps(value, ensure_ascii=False)


class Store:
    def __init__(self, path):
        if str(path) != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        try:
            self._initialize()
        except BaseException:
            self.db.close()
            raise

    def _initialize(self):
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS notes(
            id INTEGER PRIMARY KEY, content TEXT NOT NULL, original TEXT NOT NULL,
            event_date TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            source_message_id INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0);
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(content, normalized, tokenize='unicode61');
        CREATE TABLE IF NOT EXISTS reminders(
            id INTEGER PRIMARY KEY, content TEXT NOT NULL, due_at REAL NOT NULL,
            created_at REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            source_message_id INTEGER NOT NULL, snoozed_to INTEGER,
            FOREIGN KEY(snoozed_to) REFERENCES reminders(id));
        CREATE TABLE IF NOT EXISTS jobs(
            id INTEGER PRIMARY KEY, payload TEXT NOT NULL, received_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', checkpoint TEXT);
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tool_results(key TEXT PRIMARY KEY, result TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS action_receipts(
            id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL, call_id TEXT NOT NULL,
            action TEXT NOT NULL, outcome TEXT NOT NULL, payload TEXT NOT NULL,
            request_id INTEGER, replayed INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, UNIQUE(job_id,call_id));
        CREATE TABLE IF NOT EXISTS outbox(
            key TEXT PRIMARY KEY, text TEXT NOT NULL, reminder_id INTEGER,
            available_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            sent_at REAL, telegram_message_id INTEGER);
        CREATE TABLE IF NOT EXISTS llm_calls(
            id INTEGER PRIMARY KEY, created_at REAL NOT NULL, duration REAL NOT NULL,
            status TEXT NOT NULL, error_type TEXT, finish_reason TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER);
        CREATE TABLE IF NOT EXISTS requests(
            id INTEGER PRIMARY KEY, session INTEGER NOT NULL, action TEXT NOT NULL,
            parameters TEXT NOT NULL, origin TEXT NOT NULL, status TEXT NOT NULL
                CHECK(status IN ('waiting','ready','executing','done','failed','cancelled')),
            missing TEXT NOT NULL, target TEXT, result TEXT, error TEXT, provided TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL, updated_at REAL NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_request ON requests(session)
            WHERE status IN ('waiting','ready','executing');
        CREATE INDEX IF NOT EXISTS reminders_due ON reminders(status, due_at);
        CREATE TABLE IF NOT EXISTS schedules(
            id INTEGER PRIMARY KEY, content TEXT NOT NULL, rule TEXT NOT NULL,
            due_time TEXT NOT NULL, timezone TEXT NOT NULL,
            next_due_at REAL NOT NULL, status TEXT NOT NULL DEFAULT 'active',
            source_message_id INTEGER NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
        ''')
        for table, column, declaration in (
                ('jobs', 'session', 'INTEGER'),
                ('reminders', 'schedule_id', 'INTEGER REFERENCES schedules(id)'),
                ('reminders', 'occurrence_at', 'REAL'),
                ('reminders', 'is_snooze', 'INTEGER NOT NULL DEFAULT 0'),
                ('jobs', 'source_timezone', 'TEXT'),
                ('history', 'source_timezone', 'TEXT'),
                ('requests', 'provided', "TEXT NOT NULL DEFAULT '[]'"),
                ('history', 'source_time', 'REAL'),
                ('outbox', 'presentation_done', 'INTEGER NOT NULL DEFAULT 0'),
                ('outbox', 'delivery_started', 'INTEGER NOT NULL DEFAULT 0'),
                ('llm_calls', 'purpose', "TEXT NOT NULL DEFAULT 'work'")):
            if column not in {r[1] for r in self.db.execute('PRAGMA table_info(' + table + ')')}:
                self.db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {declaration}')
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS schedule_occurrence ON reminders(schedule_id,occurrence_at) WHERE schedule_id IS NOT NULL AND is_snooze=0')
        self.db.execute('CREATE INDEX IF NOT EXISTS schedules_due ON schedules(status,next_due_at)')
        from .event_schema import migrate
        migrate(self)
        try:
            with self.atomic():
                columns = {r[1] for r in self.db.execute('PRAGMA table_info(notes_fts)')}
                if 'normalized' not in columns or self.get_meta('notes_search_version') != INDEX_VERSION:
                    # Rebuild only the derived index, atomically with its version.
                    # A failed migration leaves the previous searchable index intact.
                    self.db.execute('DROP TABLE notes_fts')
                    self.db.execute("CREATE VIRTUAL TABLE notes_fts USING fts5(content, normalized, tokenize='unicode61')")
                    self.db.executemany('INSERT INTO notes_fts(rowid,content,normalized) VALUES(?,?,?)',
                        ((r['id'], *index_values(r['content'])) for r in self.db.execute(
                            'SELECT id,content FROM notes WHERE deleted=0 ORDER BY id')))
                    self.set_meta('notes_search_version', INDEX_VERSION)
        except BaseException:
            self.db.rollback()
            raise
        self.db.commit()

    @contextmanager
    def atomic(self):
        # sqlite3.Connection context managers commit outer transactions when nested.
        # Savepoints preserve the caller's transaction, including rollback on a crash.
        name = 'sp_' + uuid4().hex
        self.db.execute('SAVEPOINT ' + name)
        try:
            yield
        except BaseException:
            self.db.execute('ROLLBACK TO ' + name)
            self.db.execute('RELEASE ' + name)
            raise
        else:
            self.db.execute('RELEASE ' + name)

    def get_meta(self, key, default=None):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        self.db.execute('INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                        (key, dump(value)))

    def enqueue(self, update, now=None):
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO jobs(id,payload,received_at) VALUES(?,?,?)',
                            (update['update_id'], dump(update), utcnow() if now is None else now))
            self.set_meta('offset', max(update['update_id'] + 1, self.get_meta('offset', 0)))

    def next_job(self):
        row = self.db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
        return dict(row) if row else None

    def checkpoint(self, job_id, state):
        with self.db:
            self.db.execute('UPDATE jobs SET checkpoint=? WHERE id=?', (dump(state), job_id))

    def save_transcription(self, job, text):
        # Keep Telegram metadata, date, voice and message_id. No schema migration:
        # the persisted input becomes usable by the ordinary agent after restart.
        with self.atomic():
            row = self.db.execute("SELECT payload FROM jobs WHERE id=? AND status='pending' AND session=?",
                                  (job['id'], self.get_meta('session', 0))).fetchone()
            if row is None:
                return False
            payload = json.loads(row['payload'])
            payload['message']['text'] = text
            encoded = dump(payload)
            self.db.execute('UPDATE jobs SET payload=? WHERE id=?', (encoded, job['id']))
            job['payload'] = encoded
            return True

    def bind_job(self, job_id):
        with self.atomic():
            row = self.db.execute('SELECT status,session FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None or row['status'] != 'pending':
                return False
            session = self.get_meta('session', 0)
            if row['session'] is not None and row['session'] != session:
                self.db.execute("UPDATE jobs SET status='superseded' WHERE id=?", (job_id,))
                return False
            self.db.execute('UPDATE jobs SET session=?,source_timezone=coalesce(source_timezone,?) WHERE id=?', (session, timezone_name(self), job_id))
            return True

    def history_rows(self):
        boundary = max(self.get_meta('session_start', 0), self.get_meta('summary', {}).get('through', 0))
        return [dict(r) for r in self.db.execute(
            'SELECT id,role,content,source_time,source_timezone FROM history WHERE id>? ORDER BY id', (boundary,))]

    def history(self):
        return [{k: row[k] for k in ('role', 'content')} for row in self.history_rows()]

    def new_command(self, job_id, command, owner, now):
        with self.atomic():
            session = self.get_meta('session', 0)
            if command == '/new':
                self.set_meta('new_confirmation', None)
                self.db.execute("UPDATE requests SET status='cancelled',updated_at=?,error=? WHERE session=? AND status IN ('waiting','ready','executing')", (now, dump({'code': 'new_session'}), session))
                self.set_meta('session', session + 1)
                self.set_meta('session_start', self.db.execute('SELECT coalesce(max(id),0) FROM history').fetchone()[0])
                self.set_meta('summary', {})
                self.set_meta('pending', None)
                # Unstarted queued messages bind when processed, after this command.
                self.db.execute("UPDATE jobs SET status='superseded' WHERE status='pending' AND id!=? AND (session=? OR checkpoint IS NOT NULL)", (job_id, session))
                reply = 'Новая беседа начата. Памятки и напоминания сохранены.'
            else:
                reply = 'Используйте /new без аргументов — новая беседа начнётся сразу.'
            self.complete_command(job_id, reply, now)

    def record_receipt(self, job, action, outcome, payload, now, request_id=None, replayed=False):
        # The action caller owns the surrounding transaction: do not commit here.
        if not job.get('tool_call_id'):
            return
        self.db.execute('INSERT OR IGNORE INTO action_receipts(job_id,call_id,action,outcome,payload,request_id,replayed,created_at) VALUES(?,?,?,?,?,?,?,?)',
            (job['id'], job['tool_call_id'], action, outcome, dump(payload), request_id, int(replayed), now))

    def receipts(self, job_id):
        return [dict(row) for row in self.db.execute('SELECT * FROM action_receipts WHERE job_id=? ORDER BY id', (job_id,))]

    def finish(self, job_id, user_text, reply, now=None, *, presented=False):
        reply = reply if presented else public_text(reply)
        with self.atomic():
            row = self.db.execute('SELECT payload,source_timezone FROM jobs WHERE id=?', (job_id,)).fetchone()
            source_time = json.loads(row[0]).get('message', {}).get('date') if row else None
            self.db.execute('INSERT INTO history(role,content,source_time,source_timezone) VALUES(?,?,?,?)', ('user', user_text, source_time, (row['source_timezone'] if row else None) or timezone_name(self)))
            self.db.execute('INSERT INTO history(role,content,source_time,source_timezone) VALUES(?,?,?,?)',
                            ('assistant', reply, utcnow() if now is None else now, timezone_name(self)))
            self.db.execute('UPDATE jobs SET status=? WHERE id=?', ('done', job_id))
            self.queue_message(f'job:{job_id}', reply, now=now)

    def complete_command(self, job_id, reply, now):
        with self.atomic():
            self.db.execute("UPDATE jobs SET status='done' WHERE id=?", (job_id,))
            self.queue_message(f'job:{job_id}', reply, now=now)

    def record_llm_call(self, record):
        with self.db:
            self.db.execute('INSERT INTO llm_calls(created_at,duration,status,error_type,finish_reason,prompt_tokens,completion_tokens,cached_tokens,purpose) VALUES(?,?,?,?,?,?,?,?,?)',
                tuple(record[k] for k in ('created_at', 'duration', 'status', 'error_type', 'finish_reason',
                                         'prompt_tokens', 'completion_tokens', 'cached_tokens')) + (record.get('purpose', 'work'),))

    def stats_text(self):
        row = self.db.execute("""
            SELECT count(*) calls, sum(status='ok') successes,
                sum(status='failed') failures, sum(status='cancelled') cancelled,
                avg(duration) duration,
                sum(prompt_tokens) prompt_tokens, count(prompt_tokens) prompt_count,
                sum(completion_tokens) completion_tokens, count(completion_tokens) completion_count,
                sum(cached_tokens) cached_tokens, count(cached_tokens) cached_count
            FROM llm_calls""").fetchone()
        estimate = self.get_meta('context_estimate')
        context = (f"\nЛокальная оценка входа: {estimate['input_estimate']}; бюджет {estimate['budget']}, резерв {estimate['reserve']} ({estimate['purpose']})."
                   if estimate else '')
        if not row['calls']:
            return 'Статистики API пока нет. Учёт начинается после обновления бота.' + context
        def tokens(column, count):
            if row[count] == 0:
                return 'нет данных'
            suffix = f" (данные {row[count]}/{row['calls']} вызовов)" if row[count] != row['calls'] else ''
            return str(row[column]) + suffix
        last = self.db.execute('SELECT * FROM llm_calls ORDER BY id DESC LIMIT 1').fetchone()
        lines = [f"Вызовов API: {row['calls']} — успешно {row['successes']}, ошибки {row['failures']}, отменено {row['cancelled']}.",
                 'Входные токены: ' + tokens('prompt_tokens', 'prompt_count') + '.',
                 'Выходные токены: ' + tokens('completion_tokens', 'completion_count') + '.',
                 'Из кеша: ' + tokens('cached_tokens', 'cached_count') + '.',
                 f"Среднее время API: {row['duration']:.2f} с. Последний: {last['duration']:.2f} с, {last['status']}."]
        if last['finish_reason']:
            lines.append('Завершение последнего вызова: ' + last['finish_reason'] + '.')
        purposes = self.db.execute("""SELECT purpose,count(*) calls,
            sum(prompt_tokens) input, count(prompt_tokens) known_input,
            sum(completion_tokens) output, count(completion_tokens) known_output
            FROM llm_calls GROUP BY purpose""").fetchall()
        for r in purposes:
            def subtotal(column, known):
                return f"{r[column]} ({r[known]}/{r['calls']})" if r[known] else 'нет данных'
            lines.append(f"{r['purpose']}: {r['calls']} вызовов; вход {subtotal('input', 'known_input')}, выход {subtotal('output', 'known_output')}.")
        return '\n'.join(lines) + context

    def queue_message(self, key, text, reminder_id=None, now=None):
        parts = [text[i:i + 1800] for i in range(0, len(text), 1800)] or ['…']
        for i, part in enumerate(parts):
            suffix = 'last' if i == len(parts) - 1 else 'part'
            self.db.execute('INSERT OR IGNORE INTO outbox(key,text,reminder_id,available_at,presentation_done) VALUES(?,?,?,?,1)',
                            (f'{key}:{i:04d}:{suffix}', part, reminder_id, utcnow() if now is None else now))

    def due_reminders(self, now):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM reminders WHERE status='pending' AND due_at<=? ORDER BY due_at,id", (now,))]

    def prepare_reminders(self, now, render):
        with self.atomic():
            from .events import Events
            events = Events(self)
            events.generate(now)
            events.recover(now)
            events.recover_digests(now)
            events.digest(now)
            self.advance_schedules(now)
            due = self.due_reminders(now)
            groups = {}
            for r in due:
                if r['event_occurrence_id'] is not None:
                    groups.setdefault((r['event_occurrence_id'], r['due_at']), []).append(r)
            grouped = {r['id'] for members in groups.values() for r in members}
            for (occurrence_id, due_at), members in groups.items():
                leader = members[0]
                existing=self.db.execute("""SELECT g.id,r.status FROM notification_groups g JOIN reminders r ON r.id=g.leader_id
                    WHERE g.occurrence_id=? AND g.due_at=? AND r.status IN ('queued','sent') ORDER BY g.id LIMIT 1""",(occurrence_id,due_at)).fetchone()
                if existing:
                    group,status=existing['id'],existing['status']
                else:
                    group = self.db.execute('INSERT INTO notification_groups(occurrence_id,due_at,leader_id) VALUES(?,?,?)',
                                            (occurrence_id,due_at,leader['id'])).lastrowid
                    status='queued'
                    self.queue_message(f"reminder:{leader['id']}:{uuid4().hex}", render(leader), leader['id'], now)
                self.db.executemany('INSERT INTO notification_members VALUES(?,?)', ((group,r['id']) for r in members))
                self.db.executemany("UPDATE reminders SET status=? WHERE id=?", ((status,r['id']) for r in members))
            for r in due:
                if r['id'] in grouped:
                    continue
                self.queue_message(f"reminder:{r['id']}:{uuid4().hex}", render(r), r['id'], now)
                self.db.execute("UPDATE reminders SET status='queued' WHERE id=?", (r['id'],))

    def advance_schedules(self, now):
        """Collapse missed calendar occurrences; creation and cursor advance are atomic."""
        with self.atomic():
            for row in self.db.execute("SELECT * FROM schedules WHERE kind='recurring' AND status='active' AND next_due_at<=?", (now,)).fetchall():
                rule = Rule.model_validate_json(row['rule'])
                from .profile import load_profile
                from zoneinfo import ZoneInfo
                at = row['due_time'] if row['time_mode']=='explicit' else load_profile(self).notification_time
                latest = occurrence(rule, at, row['timezone'], now, previous=True)
                following = occurrence(rule, at, row['timezone'], now)
                if latest is None:
                    self.db.execute("UPDATE schedules SET status='completed' WHERE id=?", (row['id'],))
                    continue
                # An unstarted old occurrence can be superseded. Never discard a
                # partial or uncertain HTTP delivery, whose outcome is unknown.
                stale = self.db.execute('''SELECT id FROM reminders r
                    WHERE schedule_id=? AND is_snooze=0 AND occurrence_at<? AND is_manual=0 AND status IN ('pending','queued')
                    AND NOT EXISTS (SELECT 1 FROM outbox o WHERE o.reminder_id=r.id
                        AND (o.sent_at IS NOT NULL OR o.delivery_started=1))''', (row['id'], latest)).fetchall()
                for item in stale:
                    self.db.execute('DELETE FROM outbox WHERE reminder_id=? AND sent_at IS NULL', (item['id'],))
                    self.db.execute("UPDATE reminders SET status='cancelled' WHERE id=?", (item['id'],))
                exists = any(datetime.fromtimestamp(r[0],ZoneInfo(row['timezone'])).date()==datetime.fromtimestamp(latest,ZoneInfo(row['timezone'])).date()
                             for r in self.db.execute("SELECT occurrence_at FROM reminders WHERE schedule_id=? AND is_snooze=0 AND (status!='cancelled' OR (reason IS NOT NULL AND reason!='schedule_changed'))", (row['id'],)) if r[0] is not None)
                if latest >= row['next_due_at'] and not exists:
                    self.db.execute('INSERT OR IGNORE INTO reminders(content,due_at,created_at,source_message_id,schedule_id,occurrence_at,planned_due_at) VALUES(?,?,?,?,?,?,?)',
                        (row['content'],latest,now,row['source_message_id'],row['id'],latest,latest))
                if following is None:
                    self.db.execute("UPDATE schedules SET status='completed' WHERE id=?", (row['id'],))
                else:
                    self.db.execute('UPDATE schedules SET next_due_at=? WHERE id=?', (following, row['id']))

    def next_outbox(self, now):
        # queue_message appends a fixed-width :0000:part / :0001:last suffix.
        # A delayed chunk blocks only later chunks of its own message.
        row = self.db.execute('''
            SELECT o.* FROM outbox o
            WHERE o.sent_at IS NULL AND o.available_at<=?
              AND NOT EXISTS (
                SELECT 1 FROM outbox earlier
                WHERE earlier.sent_at IS NULL AND earlier.key<o.key
                  AND substr(earlier.key,1,length(earlier.key)-10)
                      = substr(o.key,1,length(o.key)-10))
            ORDER BY o.available_at,o.key LIMIT 1
            ''', (now,)).fetchone()
        return dict(row) if row else None

    def sent(self, key, telegram_message_id, now):
        with self.db:
            row = self.db.execute('SELECT reminder_id FROM outbox WHERE key=?', (key,)).fetchone()
            self.db.execute('UPDATE outbox SET sent_at=?,telegram_message_id=? WHERE key=?',
                            (now, telegram_message_id, key))
            if row and row[0] is not None and not self.db.execute(
                    'SELECT 1 FROM outbox WHERE reminder_id=? AND sent_at IS NULL', (row[0],)).fetchone():
                self.db.execute("UPDATE reminders SET status='sent' WHERE id=? AND status='queued'", (row[0],))
                self.db.execute("""UPDATE reminders SET status='sent' WHERE status='queued' AND id IN
                    (SELECT reminder_id FROM notification_members WHERE group_id IN
                     (SELECT id FROM notification_groups WHERE leader_id=?))""",(row[0],))
            self.db.execute("""UPDATE digest_runs SET state='sent' WHERE state='queued' AND NOT EXISTS
                (SELECT 1 FROM outbox WHERE key LIKE digest_runs.outbox_key || ':%' AND sent_at IS NULL)""")

    def start_delivery(self, key):
        # Commit before the network wait. After a crash the outcome is unknown.
        with self.db:
            return bool(self.db.execute('UPDATE outbox SET delivery_started=1 WHERE key=? AND sent_at IS NULL', (key,)).rowcount)

    def retry(self, key, now, retry_after=None, *, rejected=False):
        with self.db:
            row = self.db.execute('SELECT attempts FROM outbox WHERE key=?', (key,)).fetchone()
            if row is None:
                return
            delay = max(min(300, 2 ** min(row[0] + 1, 9)), retry_after or 0)
            self.db.execute('UPDATE outbox SET attempts=attempts+1,available_at=?,delivery_started=CASE WHEN ? THEN 0 ELSE delivery_started END WHERE key=?',
                            (now + delay, rejected, key))

    def snooze(self, reminder_id, now):
        with self.atomic():
            row = self.db.execute('SELECT * FROM reminders WHERE id=?', (reminder_id,)).fetchone()
            if not row or row['status'] != 'sent':
                return 'Это напоминание ещё не доставлено или уже отменено.'
            if row['snoozed_to']:
                return 'Это напоминание уже отложено на 10 минут.'
            if row['event_rule_id'] is not None:
                event = self.db.execute("SELECT e.deleted,o.status,o.expires_at FROM events e JOIN event_occurrences o ON o.event_id=e.id WHERE o.id=?",(row['event_occurrence_id'],)).fetchone()
                if not event or event['deleted'] or event['status']!='active':
                    return 'Уведомления события отключены; отложить его нельзя.'
                if now+600>=event['expires_at']:
                    return 'Срок откладывания после начала события; уведомление не отложено.'
            if row['schedule_id'] is not None:
                schedule = self.db.execute('SELECT status FROM schedules WHERE id=?', (row['schedule_id'],)).fetchone()
                if not schedule or schedule['status'] not in ('active','completed'):
                    return 'Повторяющееся напоминание отменено; отложить его нельзя.'
            members = [row]
            group = self.db.execute('SELECT id FROM notification_groups WHERE leader_id=?',(reminder_id,)).fetchone()
            if group:
                members = self.db.execute('SELECT r.* FROM reminders r JOIN notification_members m ON m.reminder_id=r.id WHERE m.group_id=? ORDER BY r.id',(group[0],)).fetchall()
            created=0
            already=False
            for member in members:
                if member['snoozed_to']:
                    already=True
                    continue
                if member['event_rule_id']:
                    active=self.db.execute('SELECT * FROM event_reminder_rules WHERE id=?',(member['event_rule_id'],)).fetchone()
                    from .events import Events
                    occ=self.db.execute('SELECT * FROM event_occurrences WHERE id=?',(member['event_occurrence_id'],)).fetchone()
                    if not Events(self).rule_applies(active,occ):
                        continue
                cur = self.db.execute('INSERT INTO reminders(content,due_at,created_at,source_message_id,schedule_id,occurrence_at,is_snooze,event_rule_id,event_occurrence_id,standalone_id,planned_due_at,parent_reminder_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (member['content'],now+600,now,member['source_message_id'],member['schedule_id'],member['occurrence_at'],1,member['event_rule_id'],member['event_occurrence_id'],member['standalone_id'],member['planned_due_at'],member['id']))
                self.db.execute('UPDATE reminders SET snoozed_to=? WHERE id=?',(cur.lastrowid,member['id']))
                created+=1
            return 'Отложено на 10 минут.' if created else 'Это напоминание уже отложено на 10 минут.' if already else 'Уведомления этого экземпляра отключены; отложить его нельзя.'
