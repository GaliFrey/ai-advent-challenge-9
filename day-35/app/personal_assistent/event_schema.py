"""Additive event migration. Executed in one savepoint; never rewrites legacy IDs."""
import json

VERSION = 3


def migrate(store):
    db = store.db
    with store.atomic():
        version=store.get_meta('events_schema_version')
        if version is not None and version>VERSION:
            raise ValueError('Версия схемы новее поддерживаемой; миграция не выполнена')
        if store.get_meta('events_schema_version') == VERSION:
            return
        statements = [
            '''CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('event','birthday')),
                content TEXT NOT NULL, timezone TEXT NOT NULL, source_message_id INTEGER NOT NULL,
                created_at REAL NOT NULL, revision INTEGER NOT NULL DEFAULT 0, deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0,1)),
                person TEXT, birth_month INTEGER CHECK(birth_month BETWEEN 1 AND 12),
                birth_day INTEGER CHECK(birth_day BETWEEN 1 AND 31), birth_year INTEGER,
                CHECK(kind!='birthday' OR (person IS NOT NULL AND birth_month IS NOT NULL AND birth_day IS NOT NULL)))''',
            '''CREATE TABLE IF NOT EXISTS event_segments(
                id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL REFERENCES events(id),
                from_position INTEGER NOT NULL CHECK(from_position>=0), to_position INTEGER, content TEXT NOT NULL,
                start_date TEXT NOT NULL, start_time TEXT, end_date TEXT, end_time TEXT,
                all_day INTEGER NOT NULL CHECK(all_day IN (0,1)), rule TEXT,
                CHECK(to_position IS NULL OR to_position>=from_position),
                CHECK((all_day=1 AND start_time IS NULL AND end_time IS NULL) OR (all_day=0 AND start_time IS NOT NULL)),
                CHECK(from_position>=0))''',
            '''CREATE TABLE IF NOT EXISTS event_occurrences(
                id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL REFERENCES events(id),
                segment_id INTEGER NOT NULL REFERENCES event_segments(id), position INTEGER NOT NULL,
                original_date TEXT NOT NULL, content TEXT NOT NULL, start_date TEXT NOT NULL, start_time TEXT,
                end_date TEXT, end_time TEXT, starts_at REAL NOT NULL, expires_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','cancelled')),
                manual INTEGER NOT NULL DEFAULT 0 CHECK(manual IN (0,1)), UNIQUE(event_id,position))''',
            '''CREATE TABLE IF NOT EXISTS event_reminder_rules(
                id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL REFERENCES events(id),
                occurrence_id INTEGER REFERENCES event_occurrences(id),
                from_position INTEGER NOT NULL DEFAULT 0, to_position INTEGER,
                kind TEXT NOT NULL CHECK(kind IN ('calendar','interval','fixed')),
                offset INTEGER NOT NULL DEFAULT 0 CHECK(offset>=0 AND offset<=315360000),
                time_mode TEXT NOT NULL DEFAULT 'default' CHECK(time_mode IN ('default','explicit')),
                due_time TEXT, fixed_due_at REAL, active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
                created_at REAL NOT NULL,
                CHECK((time_mode='explicit' AND due_time IS NOT NULL) OR time_mode='default'),
                CHECK(kind!='fixed' OR (fixed_due_at IS NOT NULL AND occurrence_id IS NOT NULL)))''',
            '''CREATE TABLE IF NOT EXISTS event_rule_exceptions(
                occurrence_id INTEGER NOT NULL REFERENCES event_occurrences(id),
                rule_id INTEGER NOT NULL REFERENCES event_reminder_rules(id), PRIMARY KEY(occurrence_id,rule_id))''',
            '''CREATE TABLE IF NOT EXISTS notification_groups(
                id INTEGER PRIMARY KEY, occurrence_id INTEGER NOT NULL REFERENCES event_occurrences(id),
                due_at REAL NOT NULL, leader_id INTEGER NOT NULL REFERENCES reminders(id))''',
            '''CREATE TABLE IF NOT EXISTS notification_members(
                group_id INTEGER NOT NULL REFERENCES notification_groups(id),
                reminder_id INTEGER NOT NULL UNIQUE REFERENCES reminders(id), PRIMARY KEY(group_id,reminder_id))''',
            '''CREATE TABLE IF NOT EXISTS digest_runs(
                kind TEXT NOT NULL CHECK(kind='birthdays'), year INTEGER NOT NULL, month INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('empty','queued','sent','cancelled')), outbox_key TEXT,
                PRIMARY KEY(kind,year,month))''',
        ]
        for statement in statements:
            db.execute(statement)
        columns = (
            ('event_occurrences', 'cancel_reason', "TEXT CHECK(cancel_reason IS NULL OR cancel_reason IN ('user_cancelled','plan_changed'))"),
            ('schedules', 'kind', "TEXT NOT NULL DEFAULT 'recurring' CHECK(kind IN ('once','recurring'))"),
            ('schedules', 'time_mode', "TEXT NOT NULL DEFAULT 'explicit' CHECK(time_mode IN ('default','explicit'))"),
            ('schedules', 'absolute_due_at', 'REAL'),
            ('schedules', 'local_date', 'TEXT'),
            ('reminders', 'standalone_id', 'INTEGER REFERENCES schedules(id)'),
            ('reminders', 'event_rule_id', 'INTEGER REFERENCES event_reminder_rules(id)'),
            ('reminders', 'event_occurrence_id', 'INTEGER REFERENCES event_occurrences(id)'),
            ('reminders', 'planned_due_at', 'REAL'),
            ('reminders', 'reason', 'TEXT'),
            ('reminders', 'parent_reminder_id', 'INTEGER REFERENCES reminders(id)'),
            ('reminders', 'is_manual', 'INTEGER NOT NULL DEFAULT 0 CHECK(is_manual IN (0,1))'),
        )
        for table, column, declaration in columns:
            if column not in {r[1] for r in db.execute('PRAGMA table_info(' + table + ')')}:
                db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {declaration}')
        db.execute('UPDATE reminders SET planned_due_at=due_at WHERE planned_due_at IS NULL')
        # Legacy explicit schedules expose manual moves through their immutable occurrence_at.
        db.execute("""UPDATE reminders SET is_manual=1 WHERE schedule_id IS NOT NULL AND is_snooze=0
            AND (due_at!=planned_due_at OR (due_at!=occurrence_at AND reason IS NULL
            AND EXISTS(SELECT 1 FROM schedules s WHERE s.id=reminders.schedule_id AND s.time_mode='explicit')))""")
        # Keep schedule_id NULL on old one-offs. Their absolute UTC deadline is authoritative;
        # profile timezone is only a display fallback, never a recovered original timezone.
        for row in db.execute('SELECT * FROM reminders WHERE schedule_id IS NULL AND event_rule_id IS NULL AND standalone_id IS NULL').fetchall():
            parent = db.execute('''INSERT INTO schedules(content,rule,due_time,timezone,next_due_at,status,
                source_message_id,created_at,updated_at,kind,absolute_due_at)
                VALUES(?,'null','00:00','UTC',?,?,?,?,?,'once',?)''',
                (row['content'], row['due_at'], 'active' if row['status'] in ('pending','queued','sent') else 'cancelled',
                 row['source_message_id'], row['created_at'], row['created_at'], row['due_at'])).lastrowid
            db.execute('UPDATE reminders SET standalone_id=? WHERE id=?', (parent, row['id']))
        db.execute('''UPDATE reminders SET parent_reminder_id=(SELECT p.id FROM reminders p WHERE p.snoozed_to=reminders.id)
                      WHERE parent_reminder_id IS NULL''')
        db.execute('''CREATE UNIQUE INDEX IF NOT EXISTS event_notification ON reminders(event_rule_id,event_occurrence_id)
                      WHERE event_rule_id IS NOT NULL AND is_snooze=0''')
        for operation in ('INSERT', 'UPDATE'):
            zero_allowed = '0,1' if operation=='INSERT' else '1'
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS reminder_source_{operation.lower()} BEFORE {operation} ON reminders
                WHEN NEW.status NOT IN ('pending','queued','sent','cancelled') OR
                  ((NEW.event_rule_id IS NOT NULL)+(NEW.schedule_id IS NOT NULL)+(NEW.standalone_id IS NOT NULL)) NOT IN ({zero_allowed})
                  OR (NEW.event_rule_id IS NULL)!=(NEW.event_occurrence_id IS NULL)
                  OR (NEW.event_rule_id IS NOT NULL AND NOT EXISTS
                    (SELECT 1 FROM event_reminder_rules r JOIN event_occurrences o ON o.event_id=r.event_id
                     WHERE r.id=NEW.event_rule_id AND o.id=NEW.event_occurrence_id))
                BEGIN SELECT RAISE(ABORT,'invalid reminder source'); END""")
        db.execute("""CREATE TRIGGER IF NOT EXISTS reminder_defaults AFTER INSERT ON reminders
            BEGIN
              INSERT INTO schedules(content,rule,due_time,timezone,next_due_at,status,source_message_id,created_at,updated_at,kind,absolute_due_at)
              SELECT NEW.content,'null','00:00','UTC',NEW.due_at,'active',NEW.source_message_id,NEW.created_at,NEW.created_at,'once',NEW.due_at
              WHERE NEW.schedule_id IS NULL AND NEW.event_rule_id IS NULL AND NEW.standalone_id IS NULL;
              UPDATE reminders SET standalone_id=CASE WHEN NEW.schedule_id IS NULL AND NEW.event_rule_id IS NULL AND NEW.standalone_id IS NULL
                THEN last_insert_rowid() ELSE NEW.standalone_id END,planned_due_at=coalesce(NEW.planned_due_at,NEW.due_at) WHERE id=NEW.id;
            END""")
        for operation in ('INSERT','UPDATE'):
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS occurrence_segment_{operation.lower()} BEFORE {operation} ON event_occurrences
                WHEN NOT EXISTS(SELECT 1 FROM event_segments s WHERE s.id=NEW.segment_id AND s.event_id=NEW.event_id)
                BEGIN SELECT RAISE(ABORT,'invalid occurrence segment'); END""")
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS rule_occurrence_{operation.lower()} BEFORE {operation} ON event_reminder_rules
                WHEN NEW.occurrence_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM event_occurrences o WHERE o.id=NEW.occurrence_id AND o.event_id=NEW.event_id)
                BEGIN SELECT RAISE(ABORT,'invalid rule occurrence'); END""")
        db.execute("""CREATE TRIGGER IF NOT EXISTS exception_source BEFORE INSERT ON event_rule_exceptions
            WHEN NOT EXISTS(SELECT 1 FROM event_reminder_rules r JOIN event_occurrences o ON o.event_id=r.event_id
                            WHERE r.id=NEW.rule_id AND o.id=NEW.occurrence_id)
            BEGIN SELECT RAISE(ABORT,'invalid rule exception'); END""")
        db.execute("""CREATE TRIGGER IF NOT EXISTS group_leader BEFORE INSERT ON notification_groups
            WHEN NOT EXISTS(SELECT 1 FROM reminders r WHERE r.id=NEW.leader_id AND r.event_occurrence_id=NEW.occurrence_id AND r.due_at=NEW.due_at)
            BEGIN SELECT RAISE(ABORT,'invalid notification group leader'); END""")
        db.execute("""CREATE TRIGGER IF NOT EXISTS group_member BEFORE INSERT ON notification_members
            WHEN NOT EXISTS(SELECT 1 FROM notification_groups g JOIN reminders r ON r.event_occurrence_id=g.occurrence_id AND r.due_at=g.due_at
                            WHERE g.id=NEW.group_id AND r.id=NEW.reminder_id)
            BEGIN SELECT RAISE(ABORT,'invalid notification group member'); END""")
        store.set_meta('events_schema_version', VERSION)
