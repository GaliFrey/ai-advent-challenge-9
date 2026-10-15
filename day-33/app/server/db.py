"""Слой хранения на стандартном модуле sqlite3 (добавочная схема, без ORM).

Модель данных:

* ``conversations`` — обращение; принадлежит серверной браузерной сессии,
  время создания и обновления берётся по серверным часам;
* ``messages`` — сообщения ``user``/``assistant`` с явным состоянием
  ``processing``/``answered``/``error``; у ответа есть ссылка на сообщение
  пользователя, а у сообщения пользователя — ключ идемпотентности и хэш
  содержимого;
* ``snapshots`` — диагностический снимок версии 1, привязанный к сообщению
  пользователя (message_id — первичный ключ).

Сообщение пользователя и его снимок записываются одной транзакцией
(``BEGIN IMMEDIATE``). Ответ и завершение попытки пишутся одной транзакцией и переводят
сообщение пользователя в состояние ``answered``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .errors import SupportError

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_session
    ON conversations (session_id, id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations (id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('processing', 'answered', 'error')),
    idempotency_key TEXT,
    content_hash TEXT,
    reply_message_id INTEGER REFERENCES messages (id),
    error_code TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, created_at, id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_idempotency
    ON messages (conversation_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS retrieval_results (
    message_id INTEGER PRIMARY KEY REFERENCES messages (id),
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages (id),
    session_id TEXT NOT NULL,
    state TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER,
    requested_model TEXT NOT NULL,
    returned_model TEXT,
    external_started_at REAL,
    external_outcome_unknown INTEGER NOT NULL DEFAULT 0,
    context_binding TEXT,
    index_version TEXT,
    retrieval TEXT,
    passed_ids TEXT,
    selected_ids TEXT,
    usage TEXT,
    messages_bytes INTEGER,
    estimated_tokens INTEGER,
    error_code TEXT,
    assistant_message_id INTEGER REFERENCES messages (id)
);
CREATE INDEX IF NOT EXISTS idx_generation_session_window
    ON generation_attempts (session_id, external_started_at);

CREATE TABLE IF NOT EXISTS snapshots (
    message_id INTEGER PRIMARY KEY REFERENCES messages (id),
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    """Серверное время в UTC ISO 8601 с миллисекундами: источник истины порядка."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def open_readonly(path: str | Path) -> sqlite3.Connection:
    """Открывает существующую БД в режиме только для чтения (``mode=ro``).

    Используется дочерним MCP-процессом: он не пишет и не копирует БД ради
    каждого запроса. Любая попытка записи через такое соединение завершается
    ошибкой SQLite, а не изменением данных.
    """

    uri = f"file:{Path(path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        parent = self.path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # --- восстановление после перезапуска ---------------------------------

    def recover_processing(self) -> int:
        """Незавершённые сообщения после перезапуска получают явную ошибку."""

        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE messages SET state = 'error', error_code = 'interrupted' "
                "WHERE state = 'processing'"
            )
            conn.execute(
                "UPDATE generation_attempts SET state='interrupted', error_code='interrupted', "
                "finished_at=?, duration_ms=MAX(0, CAST((julianday(?) - julianday(started_at))*86400000 AS INTEGER)), "
                "external_outcome_unknown=(external_started_at IS NOT NULL) "
                "WHERE state='processing'", (now_iso(), now_iso())
            )
            return cursor.rowcount

    # --- обращения ---------------------------------------------------------

    def create_conversation(self, session_id: str) -> sqlite3.Row:
        timestamp = now_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (session_id, created_at, updated_at) "
                "VALUES (?, ?, ?)",
                (session_id, timestamp, timestamp),
            )
            conversation_id = cursor.lastrowid
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row

    def get_conversation(self, conversation_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()

    def latest_conversation(self, session_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM conversations WHERE session_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()

    def conversation_for_session(
        self, conversation_id: int, session_id: str
    ) -> sqlite3.Row | None:
        """Обращение доступно, только если принадлежит этой сессии."""

        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND session_id = ?",
                (conversation_id, session_id),
            ).fetchone()

    # --- сообщения ---------------------------------------------------------

    def messages(self, conversation_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY created_at ASC, id ASC",
                (conversation_id,),
            ).fetchall()

    def message_count(self, conversation_id: int) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return row["total"]

    def get_message(self, message_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()

    def find_by_idempotency(
        self, conversation_id: int, idempotency_key: str
    ) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? "
                "AND idempotency_key = ? AND role = 'user'",
                (conversation_id, idempotency_key),
            ).fetchone()

    def reply_for(self, user_message_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM messages WHERE reply_message_id = ? "
                "ORDER BY id ASC LIMIT 1",
                (user_message_id,),
            ).fetchone()

    def retrieval_for(self, message_id: int):
        with self.connect() as conn:
            row = conn.execute("SELECT payload FROM retrieval_results WHERE message_id = ?", (message_id,)).fetchone()
            return json.loads(row["payload"]) if row else None

    def snapshot_for(self, message_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM snapshots WHERE message_id = ?", (message_id,)
            ).fetchone()

    def create_user_message(
        self,
        *,
        conversation_id: int,
        text: str,
        idempotency_key: str,
        content_hash: str,
        snapshot_payload: str,
        max_history: int,
    ) -> sqlite3.Row:
        """Атомарно записывает сообщение пользователя и его снимок."""

        timestamp = now_iso()
        with self.transaction() as conn:
            processing = conn.execute(
                "SELECT id FROM messages WHERE conversation_id = ? "
                "AND state = 'processing' LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if processing is not None:
                raise SupportError(
                    "already_processing",
                    "Предыдущее сообщение ещё обрабатывается.",
                    409,
                )
            total = conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()["total"]
            if total >= max_history:
                raise SupportError(
                    "history_limit",
                    "Достигнут предел числа сообщений в обращении.",
                    409,
                )
            cursor = conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, text, state, idempotency_key, content_hash, created_at) "
                "VALUES (?, 'user', ?, 'processing', ?, ?, ?)",
                (conversation_id, text, idempotency_key, content_hash, timestamp),
            )
            message_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO snapshots (message_id, payload, created_at) VALUES (?, ?, ?)",
                (message_id, snapshot_payload, timestamp),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
            return conn.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()

    def mark_message_error(self, message_id: int, error_code: str) -> None:
        """Переводит сообщение в явное состояние ошибки обработки."""

        with self.transaction() as conn:
            conn.execute(
                "UPDATE messages SET state = 'error', error_code = ? WHERE id = ?",
                (error_code, message_id),
            )

    def mark_message_processing(self, message_id: int) -> None:
        """Возвращает сохранённое сообщение в обработку для явного повтора."""

        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE messages SET state = 'processing', error_code = NULL "
                "WHERE id = ? AND state = 'error'",
                (message_id,),
            )
            if cursor.rowcount != 1:
                raise SupportError("already_processing", "Сообщение уже обрабатывается или завершено.", 409)

    def create_assistant_reply(
        self, *, conversation_id: int, user_message_id: int, text: str, retrieval: dict | None = None,
        attempt_id: int | None = None, selected_ids: list[str] | None = None, duration_ms: int | None = None
    ) -> sqlite3.Row:
        """Атомарно сохраняет ответ, retrieval, выбранные ID и успех попытки."""

        timestamp = now_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, text, state, reply_message_id, created_at) "
                "VALUES (?, 'assistant', ?, 'answered', ?, ?)",
                (conversation_id, text, user_message_id, timestamp),
            )
            reply_id = cursor.lastrowid
            if retrieval is not None:
                conn.execute("INSERT INTO retrieval_results VALUES (?, ?)",
                             (reply_id, json.dumps(retrieval, ensure_ascii=False, allow_nan=False)))
            if attempt_id is not None:
                updated = conn.execute(
                    "UPDATE generation_attempts SET state='succeeded', finished_at=?, duration_ms=?, "
                    "selected_ids=?, assistant_message_id=?, external_outcome_unknown=0 WHERE id=? AND state='processing'",
                    (timestamp, duration_ms, json.dumps(selected_ids), reply_id, attempt_id)
                )
                if updated.rowcount != 1:
                    raise RuntimeError('Attempt no longer processing')
            conn.execute(
                "UPDATE messages SET state = 'answered', error_code = NULL WHERE id = ?",
                (user_message_id,),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
            return conn.execute(
                "SELECT * FROM messages WHERE id = ?", (reply_id,)
            ).fetchone()


    def start_attempt(self, message_id: int, requested_model: str) -> int:
        with self.transaction() as conn:
            session = conn.execute(
                "SELECT c.session_id FROM conversations c JOIN messages m ON m.conversation_id=c.id WHERE m.id=?",
                (message_id,)
            ).fetchone()['session_id']
            return conn.execute(
                "INSERT INTO generation_attempts (message_id, session_id, state, started_at, requested_model) "
                "VALUES (?, ?, 'processing', ?, ?)", (message_id, session, now_iso(), requested_model)
            ).lastrowid

    def update_attempt(self, attempt_id: int, **values) -> None:
        allowed = {'returned_model', 'context_binding', 'index_version', 'retrieval', 'passed_ids',
                   'usage', 'messages_bytes', 'estimated_tokens'}
        if not values or not set(values) <= allowed:
            raise ValueError('Invalid attempt fields')
        json_fields = {'context_binding', 'retrieval', 'passed_ids', 'usage'}
        pairs = [(key, json.dumps(value, ensure_ascii=False, allow_nan=False) if key in json_fields else value)
                 for key, value in values.items()]
        with self.transaction() as conn:
            conn.execute('UPDATE generation_attempts SET ' + ', '.join(key + '=?' for key, _ in pairs) + ' WHERE id=?',
                         [value for _, value in pairs] + [attempt_id])

    def begin_external(self, attempt_id: int, limit: int, window_seconds: int, *, passed_ids=None, timestamp=None) -> None:
        now = time.time() if timestamp is None else timestamp
        with self.transaction() as conn:
            attempt = conn.execute('SELECT * FROM generation_attempts WHERE id=?', (attempt_id,)).fetchone()
            count = conn.execute(
                'SELECT COUNT(*) FROM generation_attempts WHERE session_id=? AND external_started_at > ?',
                (attempt['session_id'], now - window_seconds)
            ).fetchone()[0]
            if count >= limit:
                raise SupportError('session_limit', 'Достигнут лимит внешних попыток сессии. Попробуйте после окончания скользящего окна.', 429)
            conn.execute('UPDATE generation_attempts SET external_started_at=?, external_outcome_unknown=1, passed_ids=? WHERE id=?',
                         (now, json.dumps(passed_ids) if passed_ids is not None else None, attempt_id))

    def fail_attempt(self, attempt_id: int, code: str, duration_ms: int, *, unknown=False) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE generation_attempts SET state='failed', error_code=?, finished_at=?, duration_ms=?, "
                "external_outcome_unknown=(external_started_at IS NOT NULL AND ?) WHERE id=?",
                (code, now_iso(), duration_ms, int(unknown), attempt_id)
            )
            conn.execute("UPDATE messages SET state='error', error_code=? WHERE id=(SELECT message_id FROM generation_attempts WHERE id=?)",
                         (code, attempt_id))

    def selected_sources_for(self, assistant_message_id: int):
        """None distinguishes legacy retrieval candidates from a model selection []."""
        with self.connect() as conn:
            row = conn.execute('SELECT selected_ids FROM generation_attempts WHERE assistant_message_id=? AND state=\'succeeded\'',
                               (assistant_message_id,)).fetchone()
            return json.loads(row['selected_ids']) if row else None

    def attempts_for(self, message_id: int):
        with self.connect() as conn:
            return conn.execute('SELECT * FROM generation_attempts WHERE message_id=? ORDER BY id', (message_id,)).fetchall()
