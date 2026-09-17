"""Доменная модель и атомарное хранение состояния задачи."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from diagnostics import DiagnosticError, RunLog
from workflow_profile import get_profile


SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
PHASES = ("planning", "execution", "validation", "revision", "done")
STATUSES = ("ready", "paused", "failed", "done", "done_with_issues")
STEP_STATUSES = ("pending", "done")


class TaskError(RuntimeError):
    """Ошибка состояния, безопасная для показа в интерфейсе."""


@dataclass
class TaskStep:
    title: str
    instruction: str
    status: str = "pending"
    result: str = ""

    def checked(self) -> TaskStep:
        title, instruction, result = self.title.strip(), self.instruction.strip(), self.result.strip()
        if not title or not instruction or self.status not in STEP_STATUSES:
            raise ValueError("Шаг задачи имеет неверную структуру")
        if (self.status == "done") != bool(result):
            raise ValueError("Результат шага не соответствует его статусу")
        return TaskStep(title, instruction, self.status, result)


@dataclass
class TaskState:
    task_id: str
    goal: str
    profile_id: str
    phase: str = "planning"
    status: str = "ready"
    current_step: int = 0
    expected_action: str = ""
    steps: list[TaskStep] = field(default_factory=list)
    validation: str = ""
    final_result: str = ""
    validation_passed: bool | None = None
    validation_issues: list[str] = field(default_factory=list)
    validation_improvements: list[str] = field(default_factory=list)
    revision_instruction: str = ""
    revision_result: str = ""
    revision_count: int = 0
    last_error: str = ""
    last_request_messages: list[dict[str, str]] = field(default_factory=list)
    version: int = 1

    def checked(self) -> TaskState:
        if SAFE_ID.fullmatch(self.task_id) is None:
            raise ValueError("ID задачи содержит недопустимые символы")
        goal = self.goal.strip()
        if not goal:
            raise ValueError("Цель задачи не должна быть пустой")
        get_profile(self.profile_id)
        if self.phase not in PHASES or self.status not in STATUSES or self.version != 1:
            raise ValueError("Неизвестная фаза, статус или версия задачи")
        steps = [step.checked() if isinstance(step, TaskStep) else TaskStep(**step).checked() for step in self.steps]
        if self.phase == "planning" and steps:
            raise ValueError("До завершения planning план должен быть пуст")
        if self.phase in {"execution", "validation", "revision", "done"} and not steps:
            raise ValueError("После planning задача должна содержать шаги")
        if self.current_step < 0 or self.current_step > len(steps):
            raise ValueError("Текущий шаг находится вне плана")
        completed = sum(step.status == "done" for step in steps)
        if completed != self.current_step:
            raise ValueError("Текущий шаг не совпадает с завершёнными шагами")
        if self.phase == "execution" and self.current_step >= len(steps):
            raise ValueError("Execution не может находиться после последнего шага")
        if self.phase in {"validation", "revision", "done"} and self.current_step != len(steps):
            raise ValueError("Validation требует завершения всех шагов")
        if self.revision_count not in {0, 1, 2}:
            raise ValueError("Допустимо не более двух доработок")
        if self.validation_passed is not None and type(self.validation_passed) is not bool:
            raise ValueError("Результат validation должен быть логическим")
        if not isinstance(self.validation_issues, list) or any(
            not isinstance(item, str) or not item.strip() for item in self.validation_issues
        ):
            raise ValueError("Замечания validation имеют неверную структуру")
        issues = [item.strip() for item in self.validation_issues]
        if not isinstance(self.validation_improvements, list) or any(
            not isinstance(item, str) or not item.strip() for item in self.validation_improvements
        ):
            raise ValueError("Улучшения validation имеют неверную структуру")
        improvements = [item.strip() for item in self.validation_improvements]
        revision_instruction = self.revision_instruction.strip()
        revision_result = self.revision_result.strip()
        if self.phase == "revision" and (
            self.validation_passed is not False
            or not issues
            or not revision_instruction
            or self.revision_count >= 2
        ):
            raise ValueError("Revision требует неуспешной validation и доступной попытки")
        if self.revision_count >= 1 and not revision_result:
            raise ValueError("После revision должен быть сохранён исправленный результат")
        if self.phase == "done" and (
            self.status not in {"done", "done_with_issues"} or not self.final_result.strip()
        ):
            raise ValueError("Завершённая задача не содержит результата")
        if self.status in {"done", "done_with_issues"} and self.phase != "done":
            raise ValueError("Терминальный статус допустим только в фазе done")
        if self.phase == "done":
            if self.status == "done" and self.validation_passed is not True:
                raise ValueError("Успешное завершение требует passed=true")
            if self.status == "done_with_issues" and (
                self.validation_passed is not False or self.revision_count != 2 or not issues
            ):
                raise ValueError("Завершение с замечаниями требует провала после двух доработок")
        messages = [_message(item) for item in self.last_request_messages]
        candidate = TaskState(
            task_id=self.task_id,
            goal=goal,
            profile_id=self.profile_id,
            phase=self.phase,
            status=self.status,
            current_step=self.current_step,
            steps=steps,
            validation=self.validation.strip(),
            final_result=self.final_result.strip(),
            validation_passed=self.validation_passed,
            validation_issues=issues,
            validation_improvements=improvements,
            revision_instruction=revision_instruction,
            revision_result=revision_result,
            revision_count=self.revision_count,
            last_error=self.last_error.strip(),
            last_request_messages=messages,
            version=self.version,
        )
        candidate.expected_action = candidate.derived_action()
        return candidate

    def derived_action(self) -> str:
        if self.status == "paused":
            return "продолжить с сохранённого шага"
        if self.status == "failed":
            return "повторить текущий этап"
        if self.phase == "planning":
            return "составить план"
        if self.phase == "execution":
            return f"выполнить шаг {self.current_step + 1} из {len(self.steps)}"
        if self.phase == "validation":
            return "проверить и собрать итог"
        if self.phase == "revision":
            return "доработать результат по замечаниям"
        return "нет — задача завершена"

    @property
    def current_step_label(self) -> str:
        if self.phase == "planning":
            return "Формирование плана"
        if self.phase == "execution":
            return self.steps[self.current_step].title
        if self.phase == "validation":
            return "Повторная проверка" if self.revision_count else "Проверка полного результата"
        if self.phase == "revision":
            return "Доработка по результатам проверки"
        return "Завершено"

    def pause(self) -> None:
        if self.phase == "done":
            raise TaskError("Завершённую задачу нельзя поставить на паузу")
        self.status = "paused"
        self.expected_action = self.derived_action()

    def resume(self) -> None:
        if self.status != "paused":
            raise TaskError("Продолжить можно только задачу на паузе")
        self.status = "ready"
        self.last_error = ""
        self.expected_action = self.derived_action()


def _message(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise ValueError("Неверная структура сообщения")
    role, content = value["role"], value["content"]
    if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
        raise ValueError("Неверное сообщение в фактическом prompt")
    return {"role": role, "content": content.strip()}


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class TaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "tasks"
        self.run_log = RunLog(root)
        self.diagnostic_warning = ""

    def log_event(self, task: TaskState, event_name: str, **values: Any) -> None:
        try:
            self.run_log.append(task, event_name, **values)
        except DiagnosticError as error:
            self.diagnostic_warning = str(error)

    def path(self, task_id: str) -> Path:
        if SAFE_ID.fullmatch(task_id) is None:
            raise TaskError("ID задачи содержит недопустимые символы")
        return self.root / f"{task_id}.json"

    def save(self, task: TaskState) -> None:
        try:
            checked = task.checked()
            _atomic_save(self.path(checked.task_id), asdict(checked))
        except (OSError, TypeError, ValueError) as error:
            raise TaskError("Не удалось атомарно сохранить состояние задачи") from error

    def load(self, task_id: str) -> TaskState:
        path = self.path(task_id)
        try:
            with path.open(encoding="utf-8") as source:
                raw = json.load(source)
            if not isinstance(raw, dict):
                raise ValueError("ожидался объект")
            return TaskState(**raw).checked()
        except json.JSONDecodeError as error:
            raise TaskError(f"Повреждён JSON задачи: {path.name}") from error
        except (OSError, TypeError, ValueError) as error:
            raise TaskError(f"Задача {path.name} имеет неверную структуру") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in self.root.glob("task-*.json")))

    def create(self, task_id: str, goal: str, profile_id: str) -> TaskState:
        if self.path(task_id).exists():
            raise TaskError(f"Задача {task_id} уже существует")
        task = TaskState(task_id, goal, profile_id)
        task.expected_action = task.derived_action()
        self.save(task)
        checked = task.checked()
        self.log_event(checked, "task_created", details={"goal": checked.goal})
        return checked

    def pause(self, task: TaskState) -> None:
        phase_before, status_before, step_before = task.phase, task.status, task.current_step
        task.pause()
        self.save(task)
        self.log_event(
            task,
            "paused",
            phase_before=phase_before,
            status_before=status_before,
            step_before=step_before,
        )

    def resume(self, task: TaskState) -> None:
        phase_before, status_before, step_before = task.phase, task.status, task.current_step
        task.resume()
        self.save(task)
        self.log_event(
            task,
            "resumed",
            phase_before=phase_before,
            status_before=status_before,
            step_before=step_before,
        )
