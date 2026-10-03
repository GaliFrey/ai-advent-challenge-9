"""Diagnostic JSONL events without requests, responses or credentials."""
from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

LOG_DIRECTORY = Path(__file__).resolve().parent / "logs"
STAGES = frozenset({"configuration", "index", "embedding_model", "reranker_model", "rewrite", "plan",
                    "search", "filter", "rerank", "answer", "validate", "judge", "save"})
MODES = frozenset({"baseline", "rewrite", "plan", "rewrite_filter", "rewrite_rerank"})
METRICS = frozenset({"candidates", "selected", "discarded", "tokens",
                     "top_k_before", "top_k_after", "threshold"})


class SessionLog:
    def __init__(self, directory: Path = LOG_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid4().hex
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = directory / f"rag-{stamp}-{self.session_id[:8]}.jsonl"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self._write("session_started")

    def _write(self, event: str, **fields):
        record = {"at": datetime.now(timezone.utc).isoformat(),
                  "session_id": self.session_id, "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    @contextmanager
    def stage(self, stage: str, *, mode: str | None = None,
              question_number: int | None = None, **metrics):
        if stage not in STAGES or (mode is not None and mode not in MODES):
            raise ValueError("Unknown diagnostic stage or mode")
        if question_number is not None and (type(question_number) is not int or question_number < 1):
            raise ValueError("Question number must be a positive integer")
        if not set(metrics) <= METRICS or any(type(value) not in (int, float) for value in metrics.values()):
            raise ValueError("Only supported numeric metrics may be logged")
        fields = {"stage": stage, "mode": mode, "question_number": question_number,
                  **metrics}
        self._write("stage_started", **fields)
        started = time.perf_counter()
        try:
            yield
        except BaseException as error:
            frames = [{"file": Path(frame.filename).name, "line": frame.lineno,
                       "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)[-12:]]
            # Exception messages, source lines and locals may contain secrets.
            self._write("stage_failed", **fields,
                        seconds=round(time.perf_counter() - started, 6),
                        error_type=type(error).__name__, frames=frames)
            raise
        else:
            self._write("stage_finished", **fields,
                        seconds=round(time.perf_counter() - started, 6))
