"""The SSH MCP server available in the day 19 TUI."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Server:
    name: str
    address: str
    command: tuple[str, ...]


SERVERS = (
    Server(
        "Yandex VM · SSH-входы", "yc · SSH/STDIO",
        (
            "ssh", "-T", "-F", str(Path.home() / ".ssh/config"), "-o", "BatchMode=yes", "yc",
            "/home/heimdall/.local/bin/uv", "run", "--directory", "/home/heimdall/ai-advent-day19",
            "--locked", "--no-sync", "python", "server.py",
        ),
    ),
)
