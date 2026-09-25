"""Save the exact analysis as a private Markdown file on the third VM."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from uuid import uuid4

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


server = MCPServer("day-20-report")
REPORT_DIR = Path.home() / "ai-advent-day20/reports"


@server.tool()
def save_report(analysis: dict, analysis_sha256: str) -> dict:
    """Save the verified analysis as Markdown on this VM. The client must download and verify the returned file before reporting success."""
    canonical = json.dumps(analysis, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != analysis_sha256:
        raise ToolError("Analysis SHA256 mismatch")
    if not isinstance(analysis.get("total"), int) or not isinstance(analysis.get("by_user"), dict):
        raise ToolError("Invalid analysis")
    lines = ["# Отчёт об SSH-аутентификациях", "", f"Период: {analysis['since']} — {analysis['until']}",
             f"Часовой пояс: {analysis['timezone']}",
             f"Всего входов: {analysis['total']}", f"Уникальных IP: {analysis['unique_ips']}", "",
             "## По пользователям", "", "```json", json.dumps(analysis["by_user"], ensure_ascii=False, indent=2), "```", "",
             "## По IP", "", "```json", json.dumps(analysis["by_ip"], ensure_ascii=False, indent=2), "```", "",
             "## Ограничения", "", *analysis["warnings"], "",
             f"SHA256 исходного снимка: {analysis['snapshot_sha256']}",
             f"SHA256 анализа: {analysis_sha256}", ""]
    content = "\n".join(lines).encode()
    REPORT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = datetime.now(ZoneInfo("Europe/Kirov")).strftime("ssh-%Y-%m-%d_%H-%M-%S") + f"-{uuid4().hex[:6]}.md"
    path = REPORT_DIR / name
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
    except OSError as error:
        raise ToolError(f"Cannot save report: {error}") from error
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "analysis_sha256": analysis_sha256}


if __name__ == "__main__":
    server.run(transport="stdio")
