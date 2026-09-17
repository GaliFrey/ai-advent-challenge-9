#!/usr/bin/env python3
"""Текстовый отчёт по токенам, переходам и поведению одного прогона."""

from __future__ import annotations

import argparse
from pathlib import Path

from diagnostics import DiagnosticError, RunLog
from main import DEFAULT_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Диагностика сохранённого прогона дня 13.")
    parser.add_argument("task_id", help="ID задачи, например task-01")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Каталог данных приложения")
    return parser.parse_args()


def render(task_id: str, log: RunLog) -> str:
    events = log.read(task_id)
    summary = log.summary(task_id)
    lines = [f"Задача: {task_id}", "", "События:"]
    for event in events:
        usage = event["usage"] or {}
        tokens = usage.get("total_tokens", 0)
        elapsed = event["elapsed_seconds"]
        elapsed_text = "—" if elapsed is None else f"{elapsed:.2f}s"
        lines.append(
            f"{event['sequence']:02d}  {event['event']:<16}  "
            f"{event['transition']:<39}  {tokens:>5} tok  {elapsed_text}"
        )
    usage = summary["usage"]
    lines.extend(
        [
            "",
            "Итого:",
            f"  API-вызовов: {summary['calls']}",
            f"  Ошибок стадий: {summary['failures']}",
            f"  Токены: input {usage['input_tokens']} + output {usage['output_tokens']} = {usage['total_tokens']}",
            f"  Время API: {summary['elapsed_seconds']:.2f}s",
            "",
            "По стадиям:",
        ]
    )
    for phase, values in summary["by_phase"].items():
        lines.append(
            f"  {phase:<10} calls={values['calls']} failures={values['failures']} "
            f"tokens={values['total_tokens']} time={values['elapsed_seconds']:.2f}s"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        print(render(args.task_id, RunLog(args.data_dir)))
    except DiagnosticError as error:
        print(f"Ошибка: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
