-- Synthetic schema before events, including the uncommitted recurrence extension.
CREATE TABLE action_receipts(
            id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL, call_id TEXT NOT NULL,
            action TEXT NOT NULL, outcome TEXT NOT NULL, payload TEXT NOT NULL,
            request_id INTEGER, replayed INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, UNIQUE(job_id,call_id));
CREATE TABLE history(
            id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL, source_timezone TEXT, source_time REAL);
CREATE TABLE jobs(
            id INTEGER PRIMARY KEY, payload TEXT NOT NULL, received_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', checkpoint TEXT, session INTEGER, source_timezone TEXT);
CREATE TABLE llm_calls(
            id INTEGER PRIMARY KEY, created_at REAL NOT NULL, duration REAL NOT NULL,
            status TEXT NOT NULL, error_type TEXT, finish_reason TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER, purpose TEXT NOT NULL DEFAULT 'work');
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE notes(
            id INTEGER PRIMARY KEY, content TEXT NOT NULL, original TEXT NOT NULL,
            event_date TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            source_message_id INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0);
CREATE VIRTUAL TABLE notes_fts USING fts5(content, normalized, tokenize='unicode61');
CREATE TABLE outbox(
            key TEXT PRIMARY KEY, text TEXT NOT NULL, reminder_id INTEGER,
            available_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            sent_at REAL, telegram_message_id INTEGER, presentation_done INTEGER NOT NULL DEFAULT 0, delivery_started INTEGER NOT NULL DEFAULT 0);
CREATE TABLE reminders(
            id INTEGER PRIMARY KEY, content TEXT NOT NULL, due_at REAL NOT NULL,
            created_at REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            source_message_id INTEGER NOT NULL, snoozed_to INTEGER, schedule_id INTEGER REFERENCES schedules(id), occurrence_at REAL, is_snooze INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(snoozed_to) REFERENCES reminders(id));
CREATE TABLE requests(
            id INTEGER PRIMARY KEY, session INTEGER NOT NULL, action TEXT NOT NULL,
            parameters TEXT NOT NULL, origin TEXT NOT NULL, status TEXT NOT NULL
                CHECK(status IN ('waiting','ready','executing','done','failed','cancelled')),
            missing TEXT NOT NULL, target TEXT, result TEXT, error TEXT, provided TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE schedules(id INTEGER PRIMARY KEY,content TEXT NOT NULL,rule TEXT NOT NULL,due_time TEXT NOT NULL,timezone TEXT NOT NULL,next_due_at REAL NOT NULL,status TEXT NOT NULL DEFAULT 'active',source_message_id INTEGER NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL);
CREATE TABLE tool_results(key TEXT PRIMARY KEY, result TEXT NOT NULL);
CREATE UNIQUE INDEX one_active_request ON requests(session)
            WHERE status IN ('waiting','ready','executing');
CREATE INDEX reminders_due ON reminders(status, due_at);
