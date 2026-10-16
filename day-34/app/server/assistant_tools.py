"""Инструменты серверного агентного цикла веб-ассистента.

Инструменты — обычные Python-функции без оболочки: чтение, подготовка
изменений в staging, фиксированная проверка, атомарное применение и завершение.
Инструментов commit/push/install/config нет. Лимиты шагов, чтения, размера
изменения и времени соблюдаются здесь.
"""

from __future__ import annotations

import json
from typing import Any

from . import assistant_scope as scope
from .assistant_staging import Staging
from .config import AssistantSettings
from .errors import SupportError


def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _schema(
        "list_files",
        "Список разрешённых frontend-файлов проекта.",
        {},
    ),
    _schema(
        "read_file",
        "Читает разрешённый frontend-файл целиком.",
        {"path": {"type": "string", "description": "Путь вида web/src/app.js"}},
        ["path"],
    ),
    _schema(
        "search",
        "Ищет подстроку в разрешённых frontend-файлах и возвращает совпадения по строкам.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "propose_edit",
        "Готовит новое содержимое разрешённого файла в staging, не меняя рабочий frontend.",
        {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Полное новое содержимое файла"},
        },
        ["path", "content"],
    ),
    _schema(
        "run_checks",
        "Запускает фиксированную серверную проверку frontend по staging-копии.",
        {},
    ),
    _schema(
        "apply",
        "Применяет подготовленные изменения к рабочему frontend после успешной проверки.",
        {},
    ),
    _schema(
        "finish",
        "Завершает операцию и возвращает пользователю краткий понятный итог.",
        {"summary": {"type": "string"}},
        ["summary"],
    ),
]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class AssistantTools:
    """Состояние одной операции редактирования: staging, лимиты и результат."""

    def __init__(
        self,
        staging: Staging,
        *,
        settings: AssistantSettings,
        project_root=None,
    ) -> None:
        self.staging = staging
        self.settings = settings
        self.project_root = project_root
        self.steps = 0
        self.read_chars = 0
        self.edits_since_check = False
        self.last_check: dict | None = None
        self.applied = False
        self.finished: dict | None = None

    # --- запуск инструментов ----------------------------------------------

    def execute(self, name: str, arguments: Any) -> str:
        self.steps += 1
        if self.steps > self.settings.max_steps:
            raise SupportError(
                "assistant_steps", "Превышен предел шагов инструментов.", 429
            )
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return _json({"error": {"code": "tool_unknown", "message": "Неизвестный инструмент."}})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _json({"error": {"code": "tool_args", "message": "Аргументы должны быть объектом."}})
        try:
            return _json(handler(**arguments))
        except SupportError as exc:
            return _json({"error": {"code": exc.code, "message": exc.message}})
        except TypeError:
            return _json({"error": {"code": "tool_args", "message": "Некорректные аргументы инструмента."}})

    # --- инструменты -------------------------------------------------------

    def _tool_list_files(self) -> dict:
        return {"files": scope.existing_allowed(self.staging.root)}

    def _tool_read_file(self, path: str) -> dict:
        if not isinstance(path, str):
            raise SupportError("tool_args", "Аргумент path должен быть строкой.", 400)
        text = self.staging.read(path)
        if len(text) > self.settings.max_read_chars:
            raise SupportError(
                "assistant_limit", "Файл превышает предел объёма чтения.", 413
            )
        self.read_chars += len(text)
        if self.read_chars > self.settings.max_total_read_chars:
            raise SupportError(
                "assistant_limit", "Превышен суммарный предел объёма чтения.", 413
            )
        return {"path": path, "content": text}

    def _tool_search(self, query: str) -> dict:
        if not isinstance(query, str) or not query:
            raise SupportError("tool_args", "Аргумент query должен быть непустой строкой.", 400)
        needle = query.lower()
        results: list[dict] = []
        for rel in scope.existing_allowed(self.staging.root):
            try:
                text = self.staging.read(rel)
            except SupportError:
                continue
            if len(text) > self.settings.max_read_chars:
                continue
            snippets = []
            for number, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    snippets.append({"line": number, "text": line[:200]})
                    if len(snippets) >= 5:
                        break
            if snippets:
                results.append({"path": rel, "matches": snippets})
                if len(results) >= self.settings.max_search_results:
                    break
        return {"query": query, "results": results}

    def _tool_propose_edit(self, path: str, content: str) -> dict:
        if not isinstance(path, str) or not isinstance(content, str):
            raise SupportError("tool_args", "path и content должны быть строками.", 400)
        if len(content) > self.settings.max_edit_chars:
            raise SupportError(
                "assistant_limit", "Размер изменения превышает предел.", 413
            )
        self.staging.write(path, content)
        self.edits_since_check = True
        self.last_check = None
        self.applied = False
        return {"path": path, "bytes": len(content.encode("utf-8"))}

    def _tool_run_checks(self) -> dict:
        result = self.staging.run_checks(timeout_seconds=self.settings.check_timeout_seconds)
        self.last_check = result
        self.edits_since_check = False
        output = result.get("output", "")
        if len(output) > self.settings.max_tool_output_chars:
            result = {**result, "output": output[-self.settings.max_tool_output_chars:]}
        return result

    def _tool_apply(self) -> dict:
        if not self.staging.changed:
            return {"applied": False, "changedPaths": [], "diff": "", "message": "Изменений нет."}
        if self.edits_since_check or not self.last_check or not self.last_check.get("passed"):
            raise SupportError(
                "apply_requires_check",
                "Применение возможно только после успешной проверки текущих изменений.",
                409,
            )
        changed = self.staging.apply(self.project_root)
        self.applied = True
        return {"applied": True, "changedPaths": changed, "diff": self.staging.diff()}

    def _tool_finish(self, summary: str) -> dict:
        if not isinstance(summary, str):
            summary = ""
        self.finished = self.result(summary.strip() or "Операция завершена.")
        return self.finished

    # --- итог --------------------------------------------------------------

    def result(self, summary: str) -> dict:
        applied = bool(self.applied)
        if applied:
            status = "applied"
        elif self.last_check is not None and not self.last_check.get("passed"):
            status = "check_failed"
        else:
            status = "no_changes"
        return {
            "status": status,
            "summary": summary,
            "changedPaths": sorted(self.staging.changed) if applied else [],
            "diff": self.staging.diff() if applied else "",
            "checks": self.last_check,
            "applied": applied,
        }
