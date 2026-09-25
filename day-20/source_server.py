"""Read a bounded, real SSH authentication snapshot from journald."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


server = MCPServer("day-20-source")
ZONE = ZoneInfo("Europe/Kirov")
ACCEPTED = re.compile(r"^Accepted (publickey|password) for ([A-Za-z0-9_.@-]+) from ([0-9a-fA-F:.]+) port \d+ ssh2")
MAX_EVENTS = 1000


@server.tool()
def read_ssh_logins(hours: int = 24) -> dict:
    """Read real Accepted SSH logins from this VM for 1–168 hours. Returns the full bounded snapshot for the client to forward unchanged to the analysis server."""
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 168:
        raise ToolError("hours must be an integer from 1 to 168")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    command = ["journalctl", "-q", "-u", "ssh.service", "--since", start.isoformat(),
               "--until", end.isoformat(), "--no-pager", "-o", "json"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    except (subprocess.SubprocessError, OSError) as error:
        raise ToolError(f"Cannot read SSH journal: {error}") from error
    events = []
    for line in result.stdout.splitlines():
        try:
            entry = json.loads(line)
            match = ACCEPTED.match(entry.get("MESSAGE", ""))
            if match is None:
                continue
            occurred = datetime.fromtimestamp(int(entry["__REALTIME_TIMESTAMP"]) / 1_000_000, timezone.utc)
            events.append({"occurred_at": occurred.astimezone(ZONE).isoformat(timespec="seconds"),
                           "username": match.group(2), "ip": match.group(3), "method": match.group(1)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ToolError(f"Invalid SSH journal entry: {error}") from error
        if len(events) > MAX_EVENTS:
            raise ToolError("More than 1000 SSH logins; use a shorter period. No partial snapshot returned.")
    snapshot = {"since": start.astimezone(ZONE).isoformat(timespec="seconds"),
                "until": end.astimezone(ZONE).isoformat(timespec="seconds"),
                "timezone": ZONE.key, "events": events,
                "warnings": ["Accepted означает успешную аутентификацию, но не действия внутри сеанса.",
                             "Доступная история journald может не покрывать весь запрошенный период."]}
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {"snapshot": snapshot, "snapshot_sha256": hashlib.sha256(canonical).hexdigest(),
            "event_count": len(events)}


if __name__ == "__main__":
    server.run(transport="stdio")
