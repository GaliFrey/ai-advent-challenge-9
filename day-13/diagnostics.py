"""Append-only журнал полного прогона задачи и агрегаты для анализа."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class TaskSnapshot(Protocol):
    task_id: str
    profile_id: str
    phase: str
    status: str
    current_step: int
    expected_action: str


class DiagnosticError(RuntimeError):
    """Ошибка диагностического журнала."""


REQUIRED_KEYS = {
    "sequence",
    "timestamp_utc",
    "event",
    "task_id",
    "profile_id",
    "phase_before",
    "phase_after",
    "status_before",
    "status_after",
    "step_before",
    "step_after",
    "expected_action_after",
    "transition",
    "model",
    "request_messages",
    "response",
    "usage",
    "elapsed_seconds",
    "error",
    "details",
}


class RunLog:
    def __init__(self, root: Path) -> None:
        self.root = root / "runs"

    def path(self, task_id: str) -> Path:
        if not task_id or "/" in task_id or "\\" in task_id or task_id in {".", ".."}:
            raise DiagnosticError("Недопустимый ID диагностического журнала")
        return self.root / task_id / "events.jsonl"

    def read(self, task_id: str) -> list[dict[str, Any]]:
        path = self.path(task_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict) or set(event) != REQUIRED_KEYS:
                        raise ValueError(f"строка {line_number} имеет неверную структуру")
                    if event["sequence"] != len(events) + 1:
                        raise ValueError(f"нарушена последовательность в строке {line_number}")
                    events.append(event)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise DiagnosticError(f"Повреждён журнал {path}") from error
        return events

    def _append(self, task_id: str, event: dict[str, Any]) -> None:
        path = self.path(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            raise DiagnosticError(f"Не удалось записать журнал {path}") from error

    def append(
        self,
        task: TaskSnapshot,
        event_name: str,
        *,
        phase_before: str | None = None,
        status_before: str | None = None,
        step_before: int | None = None,
        model: str = "",
        request_messages: list[dict[str, str]] | None = None,
        response: str = "",
        usage: dict[str, int] | None = None,
        elapsed_seconds: float | None = None,
        error: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous_phase = task.phase if phase_before is None else phase_before
        previous_status = task.status if status_before is None else status_before
        previous_step = task.current_step if step_before is None else step_before
        sequence = len(self.read(task.task_id)) + 1
        event = {
            "sequence": sequence,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event_name,
            "task_id": task.task_id,
            "profile_id": task.profile_id,
            "phase_before": previous_phase,
            "phase_after": task.phase,
            "status_before": previous_status,
            "status_after": task.status,
            "step_before": previous_step,
            "step_after": task.current_step,
            "expected_action_after": task.expected_action,
            "transition": f"{previous_phase}/{previous_status} → {task.phase}/{task.status}",
            "model": model,
            "request_messages": request_messages or [],
            "response": response,
            "usage": usage,
            "elapsed_seconds": elapsed_seconds,
            "error": error,
            "details": details or {},
        }
        self._append(task.task_id, event)
        return event

    def summary(self, task_id: str) -> dict[str, Any]:
        events = self.read(task_id)
        calls = [item for item in events if item["event"] in {"stage_completed", "stage_failed"}]
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        by_phase: dict[str, dict[str, int | float]] = {}
        for event in calls:
            usage = event["usage"] or {}
            for key in total_usage:
                total_usage[key] += int(usage.get(key, 0))
            phase = event["phase_before"]
            bucket = by_phase.setdefault(phase, {"calls": 0, "failures": 0, "total_tokens": 0, "elapsed_seconds": 0.0})
            bucket["calls"] += 1
            bucket["failures"] += int(event["event"] == "stage_failed")
            bucket["total_tokens"] += int(usage.get("total_tokens", 0))
            bucket["elapsed_seconds"] += float(event["elapsed_seconds"] or 0.0)
        return {
            "events": len(events),
            "calls": len(calls),
            "failures": sum(item["event"] == "stage_failed" for item in calls),
            "usage": total_usage,
            "elapsed_seconds": sum(float(item["elapsed_seconds"] or 0.0) for item in calls),
            "by_phase": by_phase,
        }
