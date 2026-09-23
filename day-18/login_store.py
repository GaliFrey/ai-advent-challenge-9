"""SQLite storage and bounded queries for successful SSH authentications."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path


def database_path() -> Path:
    configured = os.environ.get("DAY18_DB_PATH")
    return Path(configured).expanduser() if configured else Path.home() / ".local/state/ai-advent-day18/logins.sqlite3"


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        created = not path.exists()
        connection = sqlite3.connect(path)
        if created:
            path.chmod(0o600)
    connection.row_factory = sqlite3.Row
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS logins (
            cursor TEXT PRIMARY KEY,
            occurred_at TEXT NOT NULL,
            username TEXT NOT NULL,
            ip TEXT NOT NULL,
            method TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS logins_occurred_at ON logins(occurred_at);
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)


def metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def window(hours: int) -> tuple[str, str]:
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 168:
        raise ValueError("hours must be an integer from 1 to 168")
    end = datetime.now(timezone.utc)
    return (end - timedelta(hours=hours)).isoformat(), end.isoformat()


def recent_logins(path: Path, hours: int = 1, limit: int = 50) -> dict:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    since, until = window(hours)
    with closing(connect(path, readonly=True)) as db:
        total = db.execute(
            "SELECT count(*) FROM logins WHERE occurred_at >= ? AND occurred_at <= ?", (since, until)
        ).fetchone()[0]
        rows = db.execute(
            "SELECT occurred_at, username, ip, method FROM logins "
            "WHERE occurred_at >= ? AND occurred_at <= ? ORDER BY occurred_at DESC LIMIT ?",
            (since, until, limit),
        ).fetchall()
        last_collect = metadata(db, "last_success")
    return {
        "period_hours": hours, "timezone": "UTC", "since": since, "until": until,
        "total": total, "truncated": total > limit, "logins": [dict(row) for row in rows],
        "last_collection": last_collect,
    }


def login_summary(path: Path, hours: int = 24) -> dict:
    since, until = window(hours)
    with closing(connect(path, readonly=True)) as db:
        total, unique_ips = db.execute(
            "SELECT count(*), count(DISTINCT ip) FROM logins WHERE occurred_at >= ? AND occurred_at <= ?",
            (since, until),
        ).fetchone()
        users = db.execute(
            "SELECT username, count(*) AS count FROM logins WHERE occurred_at >= ? AND occurred_at <= ? "
            "GROUP BY username ORDER BY count DESC, username", (since, until)
        ).fetchall()
        hours_utc = db.execute(
            "SELECT substr(occurred_at, 1, 13) || ':00:00Z' AS hour, count(*) AS count "
            "FROM logins WHERE occurred_at >= ? AND occurred_at <= ? "
            "GROUP BY hour ORDER BY hour", (since, until)
        ).fetchall()
        last_collect = metadata(db, "last_success")
    return {
        "period_hours": hours, "timezone": "UTC", "since": since, "until": until,
        "total": total, "unique_ips": unique_ips, "by_user": [dict(row) for row in users],
        "by_hour": [dict(row) for row in hours_utc], "last_collection": last_collect,
    }


def collector_status(path: Path) -> dict:
    with closing(connect(path, readonly=True)) as db:
        count = db.execute("SELECT count(*) FROM logins").fetchone()[0]
        first = db.execute("SELECT min(occurred_at) FROM logins").fetchone()[0]
        last = db.execute("SELECT max(occurred_at) FROM logins").fetchone()[0]
        success = metadata(db, "last_success")
    return {"source": "systemd journal, ssh.service Accepted events", "timezone": "UTC", "total_saved": count,
            "first_login": first, "last_login": last, "last_collection": success}
