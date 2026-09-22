"""Append-only, local JSONL log for one TUI session."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SessionLog:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8] + ".jsonl"
        self.path = directory / name
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self.sequence = 0

    def append(self, label: str, data: object, server: str) -> None:
        event = {
            "sequence": self.sequence + 1,
            "time": datetime.now(timezone.utc).isoformat(),
            "server": server,
            "event": label,
            "data": data,
        }
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line)
        self.sequence += 1
