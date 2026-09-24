"""Three dependent MCP tools over the existing day 18 collector database."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("day-19-ssh-pipeline")
ZONE = ZoneInfo("Europe/Kirov")
snapshots: dict[str, dict] = {}
reports: dict[str, dict] = {}
MAX_EVENTS = 10000


def digest(data: object) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def local_time(value: str | None) -> str | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(ZONE).isoformat(timespec="seconds") if value else None


@server.tool()
def get_login_events(hours: int = 24) -> dict:
    """Step 1. Read SSH Accepted events for 1–168 hours. Return snapshot_id for analyze_login_events. Data stays on this MCP connection; never invent IDs."""
    if isinstance(hours, bool) or not 1 <= hours <= 168:
        raise ToolError("hours must be from 1 to 168")
    if len(snapshots) >= 20:
        raise ToolError("Snapshot limit reached; start a new connection")
    path = Path(os.environ.get("DAY19_DB_PATH", str(Path.home() / ".local/state/ai-advent-day18/logins.sqlite3")))
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            rows = db.execute(
                "SELECT occurred_at, username, ip, method FROM logins WHERE occurred_at >= ? AND occurred_at <= ? ORDER BY occurred_at, cursor LIMIT ?",
                (start.isoformat(), end.isoformat(), MAX_EVENTS + 1),
            ).fetchall()
            last = db.execute("SELECT value FROM metadata WHERE key='last_success'").fetchone()
    except (OSError, sqlite3.Error) as error:
        raise ToolError(f"Cannot read collector database: {error}") from error
    if len(rows) > MAX_EVENTS:
        raise ToolError("More than 10000 events; request a shorter period. No partial report created.")
    last_collection = last[0] if last else None
    warnings = ["Accepted означает успешную аутентификацию, а не действия внутри сеанса. Покрытие ограничено доступной историей journald."]
    if not last_collection or (end - datetime.fromisoformat(last_collection.replace("Z", "+00:00"))).total_seconds() > 180:
        warnings.append("Сборщик устарел или время последнего сбора неизвестно; данные могут быть неполными.")
    data = {"since": local_time(start.isoformat()), "until": local_time(end.isoformat()),
            "timezone": ZONE.key, "last_collection": local_time(last_collection), "warnings": warnings,
            "events": [{**dict(row), "occurred_at": local_time(row["occurred_at"])} for row in rows]}
    snapshot_id = uuid4().hex
    snapshots[snapshot_id] = data
    return {"snapshot_id": snapshot_id, "events_sha256": digest(data), "event_count": len(rows),
            **{key: value for key, value in data.items() if key != "events"}}


@server.tool()
def analyze_login_events(snapshot_id: str) -> dict:
    """Step 2. Analyze the exact snapshot_id returned by get_login_events on this connection. Return report_id for save_report; no database reread or LLM calculations."""
    if snapshot_id not in snapshots:
        raise ToolError("Unknown snapshot_id; first call get_login_events on this connection")
    data = snapshots[snapshot_id]
    report_id = snapshot_id
    if report_id not in reports:
        events = data["events"]
        reports[report_id] = {
            "report_id": report_id, "snapshot_id": snapshot_id, "events_sha256": digest(data),
            **{key: value for key, value in data.items() if key != "events"},
            "total": len(events), "unique_ips": len({event["ip"] for event in events}),
            "by_user": dict(sorted(Counter(e["username"] for e in events).items())),
            "by_ip": dict(sorted(Counter(e["ip"] for e in events).items())),
        }
    return reports[report_id].copy()


def markdown(report: dict) -> str:
    lines = ["# Отчёт об SSH-аутентификациях", "", f"Период: {report['since']} — {report['until']}",
             f"Часовой пояс: {report['timezone']}", f"Последний сбор: {report['last_collection'] or 'неизвестно'}",
             f"Всего входов: {report['total']}; уникальных IP: {report['unique_ips']}", ""]
    for title, key in (("По пользователям", "by_user"), ("По IP", "by_ip")):
        lines.extend([f"## {title}", "", "```json", json.dumps(report[key], ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Ограничения", "", *report["warnings"], "", f"Снимок: {report['snapshot_id']}", f"SHA256 снимка: {report['events_sha256']}", ""])
    return "\n".join(lines)


@server.tool()
def save_report(report_id: str, filename: str | None = None) -> dict:
    """Step 3. Save exactly the report_id from analyze_login_events as Markdown ON THE VM. Omit filename to use ssh-YYYY-MM-DD_HH-MM-SS.md in Europe/Kirov. Explicit filename: simple ASCII .md basename. Never overwrite existing files. Returns remote path and content hash."""
    if report_id not in reports:
        raise ToolError("Unknown report_id; first call analyze_login_events")
    if filename is None:
        filename = datetime.now(ZONE).strftime("ssh-%Y-%m-%d_%H-%M-%S.md")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.md", filename):
        raise ToolError("Use a simple ASCII .md filename without directories")
    root = Path(os.environ.get("DAY19_REPORT_DIR", str(Path.home() / "ai-advent-day19/reports")))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / filename
    content = markdown(reports[report_id]).encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(content)
    except OSError as error:
        raise ToolError(f"Cannot save report: {error}") from error
    return {"report_id": report_id, "snapshot_id": reports[report_id]["snapshot_id"],
            "path": str(path), "location": "SSH VM yc", "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


if __name__ == "__main__":
    server.run(transport="stdio")
