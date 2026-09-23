"""Import new successful SSH authentications from journald on a timer."""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
from contextlib import closing
from datetime import datetime, timezone

from login_store import connect, database_path, initialize, metadata, set_metadata


ACCEPTED = re.compile(r"^Accepted (?P<method>\S+) for (?P<username>\S+) from (?P<ip>\S+) port \d+ ssh2(?:[:\s]|$)")


def parse_entry(line: str) -> tuple[str, str, str, str, str] | None:
    record = json.loads(line)
    message = record.get("MESSAGE", "")
    match = ACCEPTED.match(message) if isinstance(message, str) else None
    if not match:
        return None
    try:
        ip = str(ipaddress.ip_address(match["ip"]))
    except ValueError:
        return None
    cursor = record["__CURSOR"]
    timestamp = datetime.fromtimestamp(int(record["__REALTIME_TIMESTAMP"]) / 1_000_000, timezone.utc).isoformat()
    return cursor, timestamp, match["username"], ip, match["method"]


def collect() -> int:
    path = database_path()
    with closing(connect(path)) as db:
        initialize(db)
        cursor = metadata(db, "journal_cursor")
        command = ["journalctl", "-u", "ssh.service", "-o", "json", "--no-pager", "-q"]
        command += ["--after-cursor", cursor] if cursor else ["--since", "24 hours ago"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        inserted = 0
        latest_cursor = cursor
        for line in result.stdout.splitlines():
            record = json.loads(line)
            latest_cursor = record["__CURSOR"]
            parsed = parse_entry(line)
            if parsed is not None:
                inserted += db.execute(
                    "INSERT OR IGNORE INTO logins(cursor, occurred_at, username, ip, method) VALUES (?, ?, ?, ?, ?)",
                    parsed,
                ).rowcount
        if latest_cursor:
            set_metadata(db, "journal_cursor", latest_cursor)
        set_metadata(db, "last_success", datetime.now(timezone.utc).isoformat())
        db.commit()
    return inserted


if __name__ == "__main__":
    print(f"Saved {collect()} new SSH logins")
