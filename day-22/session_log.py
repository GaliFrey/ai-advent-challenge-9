"""Local JSONL log of run stages. Never writes keys, prompts or answers."""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

FIELDS = frozenset({"question_id", "mode", "completed", "total", "report_file",
                    "error_type", "error_code", "frames", "tokens", "chunks", "seconds"})


def error_code(error: Exception, stage: str) -> str:
    message = str(error).lower()
    if stage == "model":
        if "name resolution" in message or "connection" in message or "huggingface" in message:
            return "MODEL_NETWORK"
        if "cache" in message or "not found" in message:
            return "MODEL_CACHE"
        if "индекс" in message or "corpus" in message:
            return "INDEX_INVALID"
        return "MODEL_LOAD"
    if stage == "request":
        if "http " in message:
            return "API_HTTP"
        if "сетев" in message or "network" in message:
            return "API_NETWORK"
        return "REQUEST_ERROR"
    return "LOCAL_ERROR"


def error_details(error: Exception, stage: str) -> dict:
    # Frame locations locate application errors without copying values held
    # in local variables or a provider's response body into the journal.
    frames = [f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
              for frame in traceback.extract_tb(error.__traceback__)[-8:]]
    return {"error_type": type(error).__name__, "error_code": error_code(error, stage),
            "frames": frames}


def user_error(error: Exception, stage: str) -> str:
    code = error_code(error, stage)
    if code == "MODEL_NETWORK":
        return "Модель эмбеддингов недоступна: проверьте локальный кэш или сеть."
    if code == "MODEL_CACHE":
        return "Нет файлов модели эмбеддингов в кэше; проверьте day-21/indexes/hf-cache."
    if code == "INDEX_INVALID":
        return "Индекс дня 21 отсутствует или устарел; выполните indexer.py build."
    if stage == "request" and isinstance(error, RuntimeError) and str(error).startswith((
            "DeepSeek HTTP ", "Сетевая ошибка DeepSeek", "DeepSeek вернул",
            "Ответ не завершён:")):
        # Only fixed messages from our API adapter are safe for the report/UI.
        return str(error)
    return f"{type(error).__name__} на этапе {stage}; подробности в журнале."


class SessionLog:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = directory / f"tui-{stamp}-{uuid4().hex[:8]}.jsonl"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)

    def append(self, event: str, **fields) -> None:
        if not set(fields) <= FIELDS:
            raise ValueError("Unsupported log field")
        record = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
