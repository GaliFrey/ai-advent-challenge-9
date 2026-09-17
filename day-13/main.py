#!/usr/bin/env python3
"""Textual-интерфейс агента с персистентной Task State Machine."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

import httpx
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, Label, RichLog, Select, Static

from agent import AgentConfig, AgentError, SUPPORTED_MODELS, WorkflowAgent, load_config
from diagnostics import DiagnosticError
from task_state import TaskError, TaskState, TaskStore
from workflow_profile import WORKFLOW_PROFILES, get_profile


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEMO_GOAL = (
    "Подготовь мини-гайд для новичка по безопасному началу работы с asyncio: "
    "три практических шага, короткий пример кода и чек-лист проверки."
)
STAGES = (
    ("planning", "1  ПЛАН"),
    ("execution", "2  ВЫПОЛНЕНИЕ"),
    ("validation", "3  ПРОВЕРКА"),
    ("revision", "4  ДОРАБОТКА"),
    ("done", "5  ЗАВЕРШЕНО"),
)


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{index:02d} {item['role'].upper()}\n{item['content']}"
        for index, item in enumerate(messages, start=1)
    )


class TaskMachineApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 13 — Task State Machine"
    BINDINGS = [("ctrl+q", "quit", "Выйти"), ("ctrl+d", "start_demo", "Автодемо")]

    def __init__(self, config: AgentConfig, *, transport=None, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        super().__init__()
        self.base_config = config
        self.transport = transport
        self.store = TaskStore(data_dir)
        self.client: httpx.AsyncClient | None = None
        self.task_state: TaskState | None = None
        self.agent: WorkflowAgent | None = None
        self.busy = False
        self.total_tokens = 0
        self.selected_phase = "planning"
        self.inflight_phase: str | None = None
        self.inflight_messages: list[dict[str, str]] = []
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        models = tuple(dict.fromkeys((self.base_config.model, *SUPPORTED_MODELS)))
        with Horizontal(id="brand"):
            yield Label("TASK MACHINE", id="title")
            yield Label("ДЕНЬ 13  /  СОСТОЯНИЕ И ВОЗОБНОВЛЕНИЕ", id="subtitle")
        with Horizontal(id="configuration"):
            yield Label("Основная модель", classes="config-label")
            yield Select(
                [(item, item) for item in models],
                value=self.base_config.model,
                id="model",
                allow_blank=False,
                compact=True,
            )
            yield Label("Workflow-профиль", classes="config-label spaced")
            yield Select(
                [(item.name, item.profile_id) for item in WORKFLOW_PROFILES],
                value=WORKFLOW_PROFILES[0].profile_id,
                id="profile",
                allow_blank=False,
                compact=True,
            )
            yield Label("Задача", classes="config-label spaced")
            yield Select(
                [("task-01", "task-01")],
                value="task-01",
                id="task",
                allow_blank=False,
                compact=True,
            )
            yield Button("Новая", id="new-task")
            yield Button("Автодемо · пауза", id="demo", variant="primary")
        yield Static("", id="scope")
        yield Static("", id="model-map")
        with Horizontal(id="goal-bar"):
            yield Label("ЦЕЛЬ", id="goal-label")
            yield Input(DEMO_GOAL, id="goal")
            yield Button("Запустить / повторить", id="run", variant="primary")
            yield Button("Пауза", id="pause")
            yield Button("Продолжить", id="resume")
        with Horizontal(id="stage-chain"):
            for index, (phase, label) in enumerate(STAGES):
                yield Button(label, id=f"stage-{phase}", classes="stage-node")
                if index < len(STAGES) - 1:
                    yield Label("──▶", classes="stage-arrow")
        with Horizontal(id="workspace"):
            with Vertical(id="chat-panel"):
                yield Label("ЧАТ ЭТАПА", id="chat-title", classes="panel-title")
                yield RichLog(id="stage-chat", classes="content", wrap=True, markup=False)
            with Vertical(id="tech-panel"):
                yield Label("ТЕХНИЧЕСКИЕ ДАННЫЕ", classes="panel-title")
                yield Static("", id="state")
                yield Label("PROMPT", id="prompt-title", classes="panel-title")
                yield RichLog(id="prompt", classes="content", wrap=True, markup=False)
        yield Static("Готово", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=20), transport=self.transport)
        try:
            ids = self.store.ids()
            self.task_state = self.store.load(ids[-1]) if ids else self.store.create("task-01", DEMO_GOAL, WORKFLOW_PROFILES[0].profile_id)
        except TaskError as error:
            self._status(str(error))
            return
        self._update_task_options()
        self._sync_controls_from_task()
        self.selected_phase = self._active_display_phase()
        self._create_agent()
        self._restore_token_total()
        self.refresh_views()

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def _create_agent(self) -> None:
        if self.client is None or self.task_state is None:
            return
        config = replace(self.base_config, model=str(self.query_one("#model", Select).value))
        self.agent = WorkflowAgent(config, self.client, self.task_state, self.store)

    def _next_task_id(self) -> str:
        numbers = [int(match.group(1)) for value in self.store.ids() if (match := re.fullmatch(r"task-(\d+)", value))]
        return f"task-{max(numbers, default=0) + 1:02d}"

    def _update_task_options(self) -> None:
        if self.task_state is None:
            return
        selector = self.query_one("#task", Select)
        with self.prevent(Select.Changed):
            selector.set_options([(item, item) for item in self.store.ids()])
            selector.value = self.task_state.task_id

    def _sync_controls_from_task(self) -> None:
        if self.task_state is None:
            return
        self.query_one("#goal", Input).value = self.task_state.goal
        with self.prevent(Select.Changed):
            self.query_one("#profile", Select).value = self.task_state.profile_id

    @on(Select.Changed, "#task")
    def task_changed(self, event: Select.Changed) -> None:
        if self.busy or event.value == Select.NULL or self.task_state is None or event.value == self.task_state.task_id:
            return
        try:
            self.task_state = self.store.load(str(event.value))
        except TaskError as error:
            self._status(str(error))
            return
        self._sync_controls_from_task()
        self.selected_phase = self._active_display_phase()
        self._create_agent()
        self._restore_token_total()
        self.refresh_views()
        self._status(f"Загружен checkpoint {self.task_state.task_id}")

    @on(Select.Changed, "#model")
    def model_changed(self) -> None:
        if not self.busy:
            self._create_agent()

    @on(Select.Changed, "#profile")
    def profile_changed(self, event: Select.Changed) -> None:
        if self.busy or self.task_state is None or event.value == Select.NULL:
            return
        if self.task_state.phase != "planning" or self.task_state.steps:
            self._status("Workflow-профиль фиксируется после начала задачи")
            self._sync_controls_from_task()
            return
        new_profile_id = str(event.value)
        if new_profile_id == self.task_state.profile_id:
            return
        old_profile_id = self.task_state.profile_id
        self.task_state.profile_id = new_profile_id
        self.store.save(self.task_state)
        self.store.log_event(
            self.task_state,
            "profile_changed",
            details={"old_profile_id": old_profile_id, "new_profile_id": self.task_state.profile_id},
        )
        self._create_agent()
        self.refresh_views()

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("stage-"):
            self.selected_phase = event.button.id.removeprefix("stage-")
            self.refresh_views()
            return
        actions = {
            "new-task": self.new_task,
            "run": self.start_run,
            "pause": self.pause_task,
            "resume": self.resume_task,
            "demo": self.action_start_demo,
        }
        action = actions.get(event.button.id or "")
        if action:
            action()

    def new_task(self) -> None:
        if self.busy:
            return
        goal = self.query_one("#goal", Input).value.strip() or DEMO_GOAL
        profile_id = str(self.query_one("#profile", Select).value)
        try:
            self.task_state = self.store.create(self._next_task_id(), goal, profile_id)
        except TaskError as error:
            self._status(str(error))
            return
        self._update_task_options()
        self._create_agent()
        self.selected_phase = "planning"
        self.total_tokens = 0
        self.refresh_views()
        self._status(f"Создана {self.task_state.task_id}")

    def start_run(self) -> None:
        if self.busy or self.task_state is None:
            return
        if self.task_state.status == "paused":
            self._status("Нажмите «Продолжить»")
            return
        if self.task_state.phase == "planning" and not self.task_state.steps:
            goal = self.query_one("#goal", Input).value.strip()
            if not goal:
                self._status("Введите цель задачи")
                return
            if goal != self.task_state.goal:
                old_goal = self.task_state.goal
                self.task_state.goal = goal
                self.store.save(self.task_state)
                self.store.log_event(
                    self.task_state,
                    "goal_updated",
                    details={"old_goal": old_goal, "new_goal": goal},
                )
        self.set_busy(True)
        self.run_to_boundary()

    def pause_task(self) -> None:
        if self.task_state is None:
            return
        try:
            if self.busy and self.agent is not None:
                self.agent.request_pause()
                self._status("Пауза запрошена: текущий API-turn завершится атомарно")
                return
            self.store.pause(self.task_state)
            self.refresh_views()
            self._status("Задача поставлена на паузу")
        except TaskError as error:
            self._status(str(error))

    def resume_task(self) -> None:
        if self.busy or self.task_state is None:
            return
        try:
            self.store.resume(self.task_state)
        except TaskError as error:
            self._status(str(error))
            return
        self.refresh_views()
        self.start_run()

    @work
    async def run_to_boundary(self) -> None:
        assert self.task_state is not None and self.agent is not None
        try:
            while self.task_state.phase != "done" and self.task_state.status != "paused":
                self.selected_phase = self.task_state.phase
                self.inflight_phase = self.task_state.phase
                self.inflight_messages = self.agent.build_messages()
                model = self.agent.model_for_phase(self.inflight_phase)
                self._status(f"⏳ {model}: {self.task_state.expected_action}")
                self.refresh_views()
                await asyncio.sleep(0)
                await self.agent.advance()
                self.inflight_phase = None
                self.inflight_messages = []
                self.total_tokens += self.agent.total_tokens
                self.agent.total_tokens = 0
                self.selected_phase = self._active_display_phase()
                self.refresh_views()
        except (AgentError, TaskError) as error:
            self._status("Остановлено: " + str(error))
        else:
            self._status("Пауза сохранена" if self.task_state.status == "paused" else "Задача завершена")
        finally:
            self.inflight_phase = None
            self.inflight_messages = []
            self._collect_tokens()
            self.set_busy(False)
            self.refresh_views()

    def action_start_demo(self) -> None:
        if self.busy:
            return
        try:
            self.task_state = self.store.create(self._next_task_id(), DEMO_GOAL, "explainer")
        except TaskError as error:
            self._status(str(error))
            return
        self._update_task_options()
        self._sync_controls_from_task()
        self._create_agent()
        self.selected_phase = "planning"
        self.total_tokens = 0
        self.set_busy(True)
        self.run_demo()

    @work
    async def run_demo(self) -> None:
        assert self.task_state is not None and self.agent is not None
        try:
            self._status("Демо 1/2: planning")
            self.selected_phase = self.task_state.phase
            self.inflight_phase = self.task_state.phase
            self.inflight_messages = self.agent.build_messages()
            self.refresh_views()
            await asyncio.sleep(0)
            await self.agent.advance()
            self._status("Демо 2/2: первый execution-шаг")
            self.selected_phase = self.task_state.phase
            self.inflight_phase = self.task_state.phase
            self.inflight_messages = self.agent.build_messages()
            self.refresh_views()
            await asyncio.sleep(0)
            await self.agent.advance()
            self.store.pause(self.task_state)
            self._collect_tokens()
            self._status("Демо на паузе: теперь нажмите «Продолжить»")
        except (AgentError, TaskError) as error:
            self._status("Демо остановлено: " + str(error))
        finally:
            self.inflight_phase = None
            self.inflight_messages = []
            self._collect_tokens()
            self.set_busy(False)
            self.refresh_views()

    def _collect_tokens(self) -> None:
        if self.agent is not None and self.agent.total_tokens:
            self.total_tokens += self.agent.total_tokens
            self.agent.total_tokens = 0

    def _restore_token_total(self) -> None:
        if self.task_state is None:
            return
        try:
            self.total_tokens = self.store.run_log.summary(self.task_state.task_id)["usage"]["total_tokens"]
        except DiagnosticError as error:
            self.store.diagnostic_warning = str(error)
            self.total_tokens = 0

    def set_busy(self, value: bool) -> None:
        self.busy = value
        for selector in ("#new-task", "#run", "#resume", "#demo"):
            self.query_one(selector, Button).disabled = value
        self.query_one("#pause", Button).disabled = not value and bool(self.task_state and self.task_state.phase == "done")
        self.query_one("#goal", Input).disabled = value
        for widget in self.query(Select):
            widget.disabled = value

    def refresh_views(self) -> None:
        if self.task_state is None:
            return
        profile = get_profile(self.task_state.profile_id)
        self.query_one("#scope", Static).update(f"TASK {self.task_state.task_id}  ·  PROFILE {profile.name}")
        if self.agent is not None:
            self.query_one("#model-map", Static).update(
                "МОДЕЛИ  ·  PLAN/EXEC  "
                f"{self.agent.model_for_phase('planning')}  ·  VALIDATION  "
                f"{self.agent.model_for_phase('validation')} [OpenRouter]  ·  REVISION  "
                f"{self.agent.model_for_phase('revision')}"
            )
        try:
            events = self.store.run_log.read(self.task_state.task_id)
        except DiagnosticError as error:
            self.store.diagnostic_warning = str(error)
            events = []
        stage_events = [
            item
            for item in events
            if item["event"] in {"stage_completed", "stage_failed"}
            and item["phase_before"] == self.selected_phase
        ]
        self._refresh_stage_chain()
        self._refresh_stage_chat(stage_events)
        self._refresh_technical_panel(stage_events)

    def _active_display_phase(self) -> str:
        if self.task_state is None:
            return "planning"
        return self.task_state.phase

    def _stage_status(self, phase: str) -> str:
        assert self.task_state is not None
        if phase == self._active_display_phase():
            return "active"
        if phase == "planning" and self.task_state.steps:
            return "complete"
        if phase == "execution" and self.task_state.steps and self.task_state.current_step == len(self.task_state.steps):
            return "complete"
        if phase == "validation" and self.task_state.validation:
            return "complete"
        if phase == "revision" and self.task_state.revision_count >= 1:
            return "complete"
        if phase == "revision" and self.task_state.phase == "done" and self.task_state.revision_count == 0:
            return "skipped"
        return "pending"

    def _refresh_stage_chain(self) -> None:
        labels = dict(STAGES)
        marks = {"active": "●", "complete": "✓", "pending": "○", "skipped": "–"}
        for phase, _ in STAGES:
            button = self.query_one(f"#stage-{phase}", Button)
            status = self._stage_status(phase)
            button.label = f"{marks[status]}  {labels[phase]}"
            button.remove_class("active", "complete", "pending", "skipped", "viewing")
            button.add_class(status)
            if phase == self.selected_phase:
                button.add_class("viewing")

    def _refresh_stage_chat(self, events: list[dict[str, object]]) -> None:
        stage_name = dict(STAGES)[self.selected_phase]
        self.query_one("#chat-title", Label).update(f"ЧАТ ЭТАПА  ·  {stage_name}")
        chat = self.query_one("#stage-chat", RichLog)
        chat.clear()
        if self.selected_phase == "done" and self.task_state is not None:
            if self.task_state.phase != "done":
                chat.write(Text("Итоговый результат появится после завершения проверки.", style="#7f91a8"))
                return
            result_style = "bold #b8d982" if self.task_state.status == "done" else "bold #e7c47b"
            result_label = "ИТОГОВЫЙ РЕЗУЛЬТАТ" if self.task_state.status == "done" else "ИТОГ С ЗАМЕЧАНИЯМИ"
            chat.write(Text(result_label + "\n", style=result_style) + Text(self.task_state.final_result))
            if self.task_state.validation_issues:
                chat.write(
                    Text("\nОСТАВШИЕСЯ ЗАМЕЧАНИЯ\n", style="bold #e68a8a")
                    + Text("\n".join(f"- {item}" for item in self.task_state.validation_issues))
                )
            return
        is_inflight = self.inflight_phase == self.selected_phase
        if not events and not is_inflight:
            if self.selected_phase == "planning" and self.task_state is not None:
                chat.write(Text("ВЫ · ЦЕЛЬ\n", style="bold #8ed6dc") + Text(self.task_state.goal))
            chat.write(Text("\nОтветов этого этапа пока нет.", style="#7f91a8"))
            return
        for index, event in enumerate(events, start=1):
            messages = event["request_messages"]
            user_messages = [item["content"] for item in messages if item["role"] == "user"]
            chat.write(Text(f"ХОД {index} · ЗАДАНИЕ\n", style="bold #8ed6dc") + Text("\n\n".join(user_messages)))
            response = str(event["response"] or "Ответ не получен")
            style = "bold #e68a8a" if event["event"] == "stage_failed" else "bold #b8d982"
            chat.write(Text("\nАГЕНТ\n", style=style) + Text(response))
            if event["error"]:
                chat.write(Text("\nОШИБКА\n", style="bold #e68a8a") + Text(str(event["error"])))
        if is_inflight:
            user_messages = [
                item["content"] for item in self.inflight_messages if item["role"] == "user"
            ]
            chat.write(
                Text(f"ХОД {len(events) + 1} · ЗАПРОС ОТПРАВЛЕН\n", style="bold #8ed6dc")
                + Text("\n\n".join(user_messages))
            )
            model = self.agent.model_for_phase(self.selected_phase) if self.agent else "модель"
            chat.write(Text(f"\n⏳ {model} работает…", style="bold #e7c47b"))

    def _refresh_technical_panel(self, events: list[dict[str, object]]) -> None:
        assert self.task_state is not None
        selected_status = self._stage_status(self.selected_phase)
        tokens = sum(int((event["usage"] or {}).get("total_tokens", 0)) for event in events)
        elapsed = sum(float(event["elapsed_seconds"] or 0.0) for event in events)
        calls = len(events)
        input_tokens = sum(int((event["usage"] or {}).get("input_tokens", 0)) for event in events)
        output_tokens = sum(int((event["usage"] or {}).get("output_tokens", 0)) for event in events)
        if self.selected_phase == "done" and self.task_state.phase == "done":
            try:
                summary = self.store.run_log.summary(self.task_state.task_id)
                calls = int(summary["calls"])
                input_tokens = int(summary["usage"]["input_tokens"])
                output_tokens = int(summary["usage"]["output_tokens"])
                tokens = int(summary["usage"]["total_tokens"])
                elapsed = float(summary["elapsed_seconds"])
            except DiagnosticError as error:
                self.store.diagnostic_warning = str(error)
        if self.inflight_phase == self.selected_phase and self.agent is not None:
            selected_model = self.agent.model_for_phase(self.selected_phase)
        elif events:
            selected_model = str(events[-1]["model"])
        elif self.selected_phase == "done":
            selected_model = "несколько моделей"
        elif self.agent is not None:
            selected_model = self.agent.model_for_phase(self.selected_phase)
        else:
            selected_model = "—"
        state = (
            f"Просмотр:     {self.selected_phase}\n"
            f"Состояние:    {selected_status}\n"
            f"Модель:       {selected_model}\n"
            f"Текущий этап: {self.task_state.phase}\n"
            f"Task status:  {self.task_state.status}\n"
            f"Ходов:        {calls}\n"
            f"Input:        {input_tokens}\n"
            f"Output:       {output_tokens}\n"
            f"Токенов:      {tokens}\n"
            f"Время API:    {elapsed:.2f} c\n"
        )
        if self.inflight_phase == self.selected_phase:
            state += "API:          выполняется…\n"
        if self.selected_phase == "execution":
            state += f"Шаг:         {self.task_state.current_step}/{len(self.task_state.steps)}\n"
        if self.selected_phase == "validation" and self.task_state.validation_passed is not None:
            state += f"Passed:       {self.task_state.validation_passed}\n"
        if self.selected_phase == "revision":
            state += f"Доработки:    {self.task_state.revision_count}/2\n"
        if self.selected_phase == "done" and self.task_state.phase == "done":
            outcome = "успешно" if self.task_state.status == "done" else "с замечаниями"
            state += f"Итог:         {outcome}\nДоработки:    {self.task_state.revision_count}/2\n"
        if self.task_state.last_error and self.selected_phase == self._active_display_phase():
            state += f"Ошибка:       {self.task_state.last_error}\n"
        if self.store.diagnostic_warning:
            state += f"Диагностика:  {self.store.diagnostic_warning}\n"
        self.query_one("#state", Static).update(state)

        prompt = self.query_one("#prompt", RichLog)
        prompt.clear()
        if self.inflight_phase == self.selected_phase:
            self.query_one("#prompt-title", Label).update("PROMPT · ОТПРАВЛЕН")
            prompt.write(_messages_text(self.inflight_messages) + "\n\n⏳ Ожидание ответа модели…")
        elif self.selected_phase == "done":
            self.query_one("#prompt-title", Label).update("ИТОГ ПРОГОНА")
            if self.task_state.phase == "done":
                prompt.write(
                    "Этап завершения не вызывает модель. Итог сформирован последней проверкой.\n\n"
                    f"Статус: {self.task_state.status}\n"
                    f"Проверка: {self.task_state.validation or 'нет данных'}"
                )
            else:
                prompt.write("Прогон ещё не завершён.")
        elif events:
            self.query_one("#prompt-title", Label).update("PROMPT")
            latest = events[-1]
            details = latest["details"]
            prompt.write(
                f"Фактический запрос · ход {len(events)}\n"
                f"format={details.get('response_format', '?')}  "
                f"max_tokens={details.get('max_tokens', '?')}  "
                f"finish={details.get('finish_reason', '?')}\n\n"
                + _messages_text(latest["request_messages"])
            )
        elif self.selected_phase == self.task_state.phase and self.agent is not None:
            self.query_one("#prompt-title", Label).update("PROMPT")
            prompt.write("Следующий запрос\n\n" + _messages_text(self.agent.build_messages()))
        else:
            self.query_one("#prompt-title", Label).update("PROMPT")
            prompt.write("Запрос для этого этапа ещё не формировался.")

    def _status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="День 13: конечный автомат состояния задачи.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Каталог сохранённых задач")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config()
        config.check()
    except ValueError as error:
        print(f"Ошибка конфигурации: {error}")
        return 2
    TaskMachineApp(config, data_dir=args.data_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
