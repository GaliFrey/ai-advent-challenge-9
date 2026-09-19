"""Human-controlled task lifecycle, stage sessions and atomic persistence."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class Phase(StrEnum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    REVISION = "revision"
    DONE = "done"


class Event(StrEnum):
    PLAN_APPROVED = "plan_approved"
    RETURNED_TO_PLANNING = "returned_to_planning"
    EXECUTION_SUBMITTED = "execution_submitted"
    VALIDATION_ACCEPTED = "validation_accepted"
    VALIDATION_SENT_TO_REVISION = "validation_sent_to_revision"
    REVISION_SUBMITTED = "revision_submitted"


class Status(StrEnum):
    READY = "ready"
    PAUSED = "paused"
    FAILED = "failed"
    DONE = "done"


TRANSITIONS: dict[tuple[Phase, Event], Phase] = {
    (Phase.PLANNING, Event.PLAN_APPROVED): Phase.EXECUTION,
    (Phase.EXECUTION, Event.RETURNED_TO_PLANNING): Phase.PLANNING,
    (Phase.EXECUTION, Event.EXECUTION_SUBMITTED): Phase.VALIDATION,
    (Phase.VALIDATION, Event.VALIDATION_ACCEPTED): Phase.DONE,
    (Phase.VALIDATION, Event.VALIDATION_SENT_TO_REVISION): Phase.REVISION,
    (Phase.REVISION, Event.REVISION_SUBMITTED): Phase.VALIDATION,
}


class TransitionError(RuntimeError):
    """A safe explanation of a rejected lifecycle operation."""


@dataclass
class ChatMessage:
    role: str
    content: str

    def checked(self) -> ChatMessage:
        content = self.content.strip()
        if self.role not in {"user", "assistant"} or not content:
            raise ValueError("Сообщение стадии имеет неверную структуру")
        return ChatMessage(self.role, content)


@dataclass
class StageRun:
    phase: Phase
    visit: int
    messages: list[ChatMessage] = field(default_factory=list)
    artifact: str = ""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    last_prompt_tokens: int = 0
    last_request_messages: list[dict[str, str]] = field(default_factory=list)
    last_response_raw: str = ""
    last_finish_reason: str = ""

    def checked(self) -> StageRun:
        try:
            phase = Phase(self.phase)
        except ValueError as error:
            raise ValueError("Запуск содержит неизвестную фазу") from error
        if phase == Phase.DONE or self.visit < 1:
            raise ValueError("Запуск стадии имеет неверную фазу или номер посещения")
        messages = [item.checked() if isinstance(item, ChatMessage) else ChatMessage(**item).checked() for item in self.messages]
        if len(messages) % 2 or any(
            item.role != ("user" if index % 2 == 0 else "assistant")
            for index, item in enumerate(messages)
        ):
            raise ValueError("История стадии должна состоять из завершённых пар user/assistant")
        if self.turns != len(messages) // 2:
            raise ValueError("Счётчик ходов стадии не совпадает с историей")
        token_values = (self.input_tokens, self.output_tokens, self.total_tokens, self.last_prompt_tokens)
        if any(type(value) is not int or value < 0 for value in token_values):
            raise ValueError("Счётчики токенов запуска имеют неверную структуру")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Суммарные токены запуска не совпадают")
        request_messages = [_checked_request_message(item) for item in self.last_request_messages]
        if not isinstance(self.last_response_raw, str) or not isinstance(self.last_finish_reason, str):
            raise ValueError("Последний обмен с API имеет неверную структуру")
        return StageRun(
            phase=phase,
            visit=self.visit,
            messages=messages,
            artifact=self.artifact.strip(),
            turns=self.turns,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            last_prompt_tokens=self.last_prompt_tokens,
            last_request_messages=request_messages,
            last_response_raw=self.last_response_raw,
            last_finish_reason=self.last_finish_reason,
        )


@dataclass
class PlanStep:
    title: str
    instruction: str

    def checked(self) -> PlanStep:
        title, instruction = self.title.strip(), self.instruction.strip()
        if not title or not instruction:
            raise ValueError("Шаг плана должен содержать название и инструкцию")
        return PlanStep(title, instruction)


def _initial_runs() -> list[StageRun]:
    return [StageRun(Phase.PLANNING, 1)]


@dataclass
class TaskState:
    task_id: str
    goal: str
    phase: Phase = Phase.PLANNING
    status: Status = Status.READY
    runs: list[StageRun] = field(default_factory=_initial_runs)
    plan: list[PlanStep] = field(default_factory=list)
    execution_result: str = ""
    validation_summary: str = ""
    validation_issues: list[str] = field(default_factory=list)
    validation_passed: bool | None = None
    revision_count: int = 0
    final_result: str = ""
    expected_action: str = ""
    last_error: str = ""
    last_request_messages: list[dict[str, str]] = field(default_factory=list)
    version: int = 3

    def checked(self) -> TaskState:
        if SAFE_ID.fullmatch(self.task_id) is None:
            raise ValueError("ID задачи содержит недопустимые символы")
        goal = self.goal.strip()
        if not goal:
            raise ValueError("Цель задачи не должна быть пустой")
        try:
            phase, status = Phase(self.phase), Status(self.status)
        except ValueError as error:
            raise ValueError("Неизвестная фаза или статус") from error
        if self.version != 3:
            raise ValueError("Неподдерживаемая версия состояния")
        runs = [item.checked() if isinstance(item, StageRun) else StageRun(**item).checked() for item in self.runs]
        if not runs or runs[0].phase != Phase.PLANNING or runs[0].visit != 1:
            raise ValueError("Маршрут должен начинаться с planning #1")
        visits: dict[Phase, int] = {}
        for run in runs:
            expected_visit = visits.get(run.phase, 0) + 1
            if run.visit != expected_visit:
                raise ValueError("Номера посещений стадии должны идти последовательно")
            visits[run.phase] = run.visit
        if phase != Phase.DONE and runs[-1].phase != phase:
            raise ValueError("Последний запуск не совпадает с активной фазой")
        if phase == Phase.DONE and runs[-1].phase != Phase.VALIDATION:
            raise ValueError("Done должен следовать за запуском validation")
        plan = [item.checked() if isinstance(item, PlanStep) else PlanStep(**item).checked() for item in self.plan]
        issues = [item.strip() for item in self.validation_issues if isinstance(item, str) and item.strip()]
        if len(issues) != len(self.validation_issues):
            raise ValueError("Замечания validation имеют неверную структуру")
        if phase in {Phase.EXECUTION, Phase.VALIDATION, Phase.REVISION, Phase.DONE} and not plan:
            raise ValueError("После planning задача должна содержать утверждённый план")
        if phase in {Phase.VALIDATION, Phase.REVISION, Phase.DONE} and not self.execution_result.strip():
            raise ValueError("Validation требует результата выполнения")
        if phase == Phase.REVISION and (self.validation_passed is not False or not issues):
            raise ValueError("Revision требует неуспешной validation с замечаниями")
        if self.validation_passed is not None and type(self.validation_passed) is not bool:
            raise ValueError("Решение validation должно быть логическим")
        if self.revision_count < 0 or self.revision_count > 2:
            raise ValueError("Допустимо не более двух доработок")
        if phase == Phase.DONE:
            if status != Status.DONE or self.validation_passed is not True or issues or not self.final_result.strip():
                raise ValueError("Done требует принятой успешной validation и итогового результата")
        elif status == Status.DONE:
            raise ValueError("Статус done допустим только в терминальной фазе")
        messages = [_checked_request_message(item) for item in self.last_request_messages]
        candidate = TaskState(
            task_id=self.task_id,
            goal=goal,
            phase=phase,
            status=status,
            runs=runs,
            plan=plan,
            execution_result=self.execution_result.strip(),
            validation_summary=self.validation_summary.strip(),
            validation_issues=issues,
            validation_passed=self.validation_passed,
            revision_count=self.revision_count,
            final_result=self.final_result.strip(),
            last_error=self.last_error.strip(),
            last_request_messages=messages,
            version=self.version,
        )
        candidate.expected_action = candidate.derived_action()
        return candidate

    @property
    def active_run(self) -> StageRun:
        if self.phase == Phase.DONE:
            raise TransitionError("У завершённой задачи нет активного запуска")
        return self.runs[-1]

    def next_visit(self, phase: Phase) -> int:
        return 1 + sum(run.phase == phase for run in self.runs)

    def derived_action(self) -> str:
        if self.status == Status.PAUSED:
            return f"продолжить сессию {self.phase.value}"
        if self.status == Status.FAILED:
            return f"повторить сообщение в сессии {self.phase.value}"
        return {
            Phase.PLANNING: "обсудить план или утвердить его",
            Phase.EXECUTION: "обсудить результат или отправить его на validation",
            Phase.VALIDATION: "обсудить проверку и выбрать итог проверки",
            Phase.REVISION: "обсудить исправления или отправить их на повторную validation",
            Phase.DONE: "нет — задача завершена",
        }[self.phase]


class TransitionController:
    """The only component allowed to change ``TaskState.phase``."""

    def allowed_events(self, phase: Phase) -> tuple[Event, ...]:
        return tuple(event for source, event in TRANSITIONS if source == phase)

    def allowed_targets(self, phase: Phase) -> tuple[Phase, ...]:
        return tuple(dict.fromkeys(TRANSITIONS[(phase, event)] for event in self.allowed_events(phase)))

    def request_target(self, task: TaskState, target: Phase | str) -> Phase:
        try:
            target = Phase(target)
        except ValueError as error:
            raise TransitionError(f"Неизвестная целевая фаза: {target}") from error
        matching_events = [
            event
            for (source, event), destination in TRANSITIONS.items()
            if source == task.phase and destination == target
        ]
        if not matching_events:
            allowed = self.allowed_targets(task.phase)
            names = ", ".join(item.value for item in allowed) or "нет"
            raise TransitionError(
                f"Переход {task.phase.value} → {target.value} запрещён. "
                f"Разрешённые цели: {names}. Ожидается: {task.derived_action()}."
            )
        if len(matching_events) != 1:
            raise TransitionError("Целевой переход неоднозначен; выберите именованное действие")
        return self.apply(task, matching_events[0])

    def apply(self, task: TaskState, event: Event | str) -> Phase:
        if task.status == Status.PAUSED:
            raise TransitionError("Задача на паузе: сначала выполните resume")
        if task.phase == Phase.DONE:
            raise TransitionError("Завершённая задача не допускает новых переходов")
        try:
            event = Event(event)
        except ValueError as error:
            raise TransitionError(f"Неизвестное событие перехода: {event}") from error
        key = (task.phase, event)
        if key not in TRANSITIONS:
            allowed = ", ".join(item.value for item in self.allowed_events(task.phase)) or "нет"
            raise TransitionError(
                f"Событие {event.value} запрещено в {task.phase.value}. Разрешённые события: {allowed}."
            )
        candidate = copy.deepcopy(task)
        self._check_and_prepare(candidate, event)
        target = TRANSITIONS[key]
        candidate.phase = target
        if target != Phase.DONE:
            candidate.runs.append(StageRun(target, candidate.next_visit(target)))
        candidate.status = Status.DONE if candidate.phase == Phase.DONE else Status.READY
        candidate.last_error = ""
        checked = candidate.checked()
        _replace_state(checked, task)
        return task.phase

    @staticmethod
    def _check_and_prepare(task: TaskState, event: Event) -> None:
        if event == Event.PLAN_APPROVED:
            if not task.plan or not task.active_run.messages:
                raise TransitionError("Нельзя начать execution: сначала обсудите и сформируйте непустой план")
        elif event == Event.RETURNED_TO_PLANNING:
            task.execution_result = ""
            task.validation_summary = ""
            task.validation_issues = []
            task.validation_passed = None
        elif event == Event.EXECUTION_SUBMITTED:
            if not task.execution_result.strip():
                raise TransitionError("Нельзя начать validation без результата execution")
            task.validation_summary = ""
            task.validation_issues = []
            task.validation_passed = None
        elif event == Event.VALIDATION_ACCEPTED:
            if task.validation_passed is not True or task.validation_issues:
                raise TransitionError("Финал разрешён только после успешной validation без замечаний")
            task.final_result = task.execution_result
        elif event == Event.VALIDATION_SENT_TO_REVISION:
            if task.validation_passed is not False or not task.validation_issues:
                raise TransitionError("Revision разрешена только после validation с замечаниями")
            if task.revision_count >= 2:
                raise TransitionError("Лимит автоматических доработок исчерпан")
        elif event == Event.REVISION_SUBMITTED:
            if not task.execution_result.strip() or not task.active_run.messages:
                raise TransitionError("Повторная validation требует сохранённой доработки")
            task.revision_count += 1
            task.validation_summary = ""
            task.validation_issues = []
            task.validation_passed = None


def _checked_request_message(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise ValueError("Неверная структура сообщения фактического prompt")
    role, content = value["role"], value["content"]
    if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
        raise ValueError("Неверное сообщение фактического prompt")
    return {"role": role, "content": content.strip()}


def _replace_state(source: TaskState, target: TaskState) -> None:
    for item in fields(TaskState):
        setattr(target, item.name, copy.deepcopy(getattr(source, item.name)))


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
        self.root = root

    def path(self, task_id: str) -> Path:
        if SAFE_ID.fullmatch(task_id) is None:
            raise TransitionError("Некорректный ID задачи")
        return self.root / "tasks" / f"{task_id}.json"

    def save(self, task: TaskState) -> None:
        checked = task.checked()
        _atomic_save(self.path(task.task_id), asdict(checked))

    def load(self, task_id: str) -> TaskState:
        try:
            raw = json.loads(self.path(task_id).read_text(encoding="utf-8"))
            return TaskState(**raw).checked()
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise TransitionError(f"Не удалось восстановить {task_id}: состояние повреждено") from error

    def create(self, task_id: str, goal: str) -> TaskState:
        if self.path(task_id).exists():
            raise TransitionError(f"Задача {task_id} уже существует")
        task = TaskState(task_id, goal).checked()
        self.save(task)
        self.append_event(task, "task_created")
        return task

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in (self.root / "tasks").glob("task-*.json")))

    def pause(self, task: TaskState) -> None:
        if task.phase == Phase.DONE:
            raise TransitionError("Завершённую задачу нельзя поставить на паузу")
        if task.status == Status.PAUSED:
            raise TransitionError("Задача уже находится на паузе")
        candidate = copy.deepcopy(task)
        candidate.status = Status.PAUSED
        candidate.expected_action = candidate.derived_action()
        self.save(candidate)
        _replace_state(candidate, task)
        self.append_event(task, "paused")

    def resume(self, task: TaskState) -> None:
        if task.status != Status.PAUSED:
            raise TransitionError("Продолжить можно только задачу на паузе")
        candidate = copy.deepcopy(task)
        candidate.status = Status.READY
        candidate.last_error = ""
        candidate.expected_action = candidate.derived_action()
        self.save(candidate)
        _replace_state(candidate, task)
        self.append_event(task, "resumed")

    def append_event(self, task: TaskState, event: str, **details: Any) -> None:
        path = self.root / "runs" / f"{task.task_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "phase": task.phase.value,
            "status": task.status.value,
            **details,
        }
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def events(self, task_id: str) -> list[dict[str, Any]]:
        path = self.root / "runs" / f"{task_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
