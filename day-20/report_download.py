"""Fetch the report from the report VM and verify it before announcing success."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path, PurePosixPath

from servers import SERVERS


REPORT_DIR = Path(__file__).resolve().parent / "reports"
REMOTE_DIR = PurePosixPath("/home/heimdall/ai-advent-day20/reports")


async def read_remote(path: str) -> bytes:
    process = await asyncio.create_subprocess_exec(
        "ssh", "-T", "-F", str(Path.home() / ".ssh/config"), "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", SERVERS[2].host, "cat", "--", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace").strip() or "SSH download failed")
    return stdout


async def download_report(saved: dict) -> dict:
    remote = PurePosixPath(saved["path"])
    if remote.parent != REMOTE_DIR or not re.fullmatch(r"ssh-\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d-[0-9a-f]{6}\.md", remote.name):
        raise ValueError("Сервер вернул недопустимый путь отчёта")
    content = await read_remote(str(remote))
    if len(content) != saved["bytes"] or hashlib.sha256(content).hexdigest() != saved["sha256"]:
        raise ValueError("Размер или SHA256 скачанного отчёта не совпадает с результатом MCP")
    REPORT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = REPORT_DIR / remote.name
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"local_path": str(destination), "bytes": len(content), "sha256": saved["sha256"]}
