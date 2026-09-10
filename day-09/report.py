"""Атомарное сохранение JSON-отчётов демо-прогона."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results"


class ReportError(RuntimeError):
    """Отчёт не удалось записать полностью и атомарно."""


def save_demo_report(
    payload: Mapping[str, Any],
    directory: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Добавляет время создания и атомарно записывает новый отчёт."""

    created_at = datetime.now(timezone.utc)
    filename = created_at.strftime("demo-%Y%m%dT%H%M%S-%fZ.json")
    path = directory / filename
    temporary_path: Path | None = None
    document = {"created_at": created_at.isoformat(), **payload}
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            json.dump(document, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ReportError("Не удалось сохранить JSON-отчёт демо") from error
    return path
