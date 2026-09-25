"""Fixed, explicit routing table for the three SSH/STDIO MCP servers."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Server:
    role: str
    host: str
    tool: str

    @property
    def command(self) -> tuple[str, ...]:
        return (
            "ssh", "-T", "-F", str(Path.home() / ".ssh/config"),
            "-o", "BatchMode=yes", self.host,
            "/home/heimdall/.local/bin/uv", "run", "--directory", "/home/heimdall/ai-advent-day20",
            "--locked", "--no-sync", "python", f"{self.role}_server.py",
        )


SERVERS = (
    Server("source", "ai-advent-20-mcp-source", "read_ssh_logins"),
    Server("analyze", "ai-advent-20-mcp-analyze", "analyze_logins"),
    Server("report", "ai-advent-20-mcp-report", "save_report"),
)
BY_ALIAS = {f"{server.role}__{server.tool}": server for server in SERVERS}
