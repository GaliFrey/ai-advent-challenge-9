"""Textual UI for human-controlled stage sessions and transitions."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import httpx
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    LoadingIndicator,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from agent import AgentConfig, AgentError, StageChatAgent, load_config
from task_state import Event, Phase, Status, TaskState, TaskStore, TransitionController, TransitionError


ROOT = Path(__file__).resolve().parent
PHASE_LABELS = {
    Phase.PLANNING: "1 ПЛАН",
    Phase.EXECUTION: "2 ВЫПОЛНЕНИЕ",
    Phase.VALIDATION: "3 ПРОВЕРКА",
    Phase.REVISION: "4 ДОРАБОТКА",
    Phase.DONE: "5 ГОТОВО",
}

class ControlledTransitionsApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "AI Advent · День 15"
    BINDINGS = [
        ("ctrl+enter", "send_stage_message", "Send"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        data_dir: Path,
        config: AgentConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self.store = TaskStore(data_dir)
        self.controller = TransitionController()
        self.config = config
        self.client = client
        self._owns_client = client is None
        self.task_state: TaskState | None = None
        self.agent: StageChatAgent | None = None
        self.selected_run_index: int | None = 0
        self.busy = False
        self.creating_task = not bool(self.store.ids())
        self.pending_message = ""

    def compose(self) -> ComposeResult:
        yield Static("ДЕНЬ 15 · УПРАВЛЯЕМЫЕ СЕССИИ ЭТАПОВ", id="brand")
        with Horizontal(id="goal-bar"):
            yield Input(
                "Подготовить краткий безопасный гайд по asyncio",
                id="goal",
                disabled=not self.creating_task,
            )
            yield Label("Задача", id="task-label")
            yield Select([("нет задач", "none")], value="none", id="task-select", allow_blank=False)
            yield Button("Создать" if self.creating_task else "Новая задача", id="new", variant="primary")
            yield Button("Отмена", id="cancel-new")
        with Horizontal(id="lifecycle"):
            for phase, label in PHASE_LABELS.items():
                yield Button(label, id=f"stage-{phase.value}", classes="phase")
        yield RichLog(id="run-route", wrap=False, markup=False)
        with Horizontal(id="transition-bar"):
            yield Button("Утвердить план", id="primary-transition", variant="success")
            yield Button("Назад", id="secondary-transition")
            yield Button("Пауза", id="pause")
            yield Button("Продолжить", id="resume")
            yield LoadingIndicator(id="api-loading")
            yield Static("", id="api-status")
            yield Static("", id="guard-hint")
        with Horizontal(id="workspace"):
            with Vertical(id="chat-panel"):
                with Horizontal(id="run-header"):
                    yield Static("ЗАПУСК СТАДИИ", id="chat-title", classes="panel-title")
                    yield Button("←", id="previous-run")
                    yield Button("→", id="next-run")
                yield RichLog(id="chat", wrap=True, markup=False)
                with Horizontal(id="message-bar"):
                    yield Input(placeholder="Обсудите план с моделью…", id="message")
                    yield Button("Отправить", id="send", variant="primary")
            with Vertical(id="details-panel"):
                with TabbedContent(initial="state-tab", id="details-tabs"):
                    with TabPane("Состояние", id="state-tab"):
                        yield RichLog(id="state", wrap=True, markup=False)
                    with TabPane("Артефакт", id="artifact-tab"):
                        yield RichLog(id="artifact", wrap=True, markup=False)
                    with TabPane("Переходы", id="transitions-tab"):
                        yield RichLog(id="transitions", wrap=True, markup=False)
                        with Horizontal(id="manual-transition"):
                            yield Label("Цель", id="target-label")
                            yield Select(
                                [(phase.value, phase.value) for phase in Phase],
                                value=Phase.VALIDATION.value,
                                id="target-phase",
                                allow_blank=False,
                            )
                            yield Button("Запросить", id="request-transition", variant="warning")
                    with TabPane("Prompt", id="prompt-tab"):
                        yield RichLog(id="prompt", wrap=True, markup=False)
        yield Static("Готово", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        if self.client is None:
            self.client = httpx.AsyncClient()
        if self.config is None:
            try:
                self.config = load_config()
                self.config.check()
            except ValueError as error:
                self._set_status(str(error))
        ids = self.store.ids()
        if ids:
            try:
                task = self.store.load(ids[-1])
                self._attach(task)
                self.selected_run_index = None if task.phase == Phase.DONE else len(task.runs) - 1
                self.query_one("#goal", Input).value = task.goal
                self.creating_task = False
            except TransitionError as error:
                self._set_status(str(error))
        self._update_task_options()
        self._sync_creation_controls()
        self._refresh()

    async def on_unmount(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()

    def _attach(self, task: TaskState) -> None:
        self.task_state = task
        if self.config is not None and self.client is not None:
            self.agent = StageChatAgent(self.config, self.client, task, self.store, self.controller)

    def _next_id(self) -> str:
        numbers = [int(value.removeprefix("task-")) for value in self.store.ids() if value.removeprefix("task-").isdigit()]
        return f"task-{max(numbers, default=0) + 1:02d}"

    def _update_task_options(self) -> None:
        selector = self.query_one("#task-select", Select)
        ids = self.store.ids()
        with self.prevent(Select.Changed):
            if ids:
                options: list[tuple[str, str]] = []
                for task_id in ids:
                    try:
                        task = self.store.load(task_id)
                        visit = task.runs[-1].visit
                        goal = task.goal if len(task.goal) <= 28 else task.goal[:27] + "…"
                        options.append((f"{task_id} · {goal} · {task.phase.value} #{visit}", task_id))
                    except TransitionError:
                        options.append((f"{task_id} · повреждена", task_id))
                selector.set_options(options)
                selector.value = self.task_state.task_id if self.task_state is not None else ids[-1]
                selector.disabled = self.creating_task or self.busy
            else:
                selector.set_options([("нет задач", "none")])
                selector.value = "none"
                selector.disabled = True

    def _sync_creation_controls(self) -> None:
        self.query_one("#goal", Input).disabled = not self.creating_task
        new_button = self.query_one("#new", Button)
        new_button.label = "Создать" if self.creating_task else "Новая задача"
        new_button.disabled = self.busy
        cancel_button = self.query_one("#cancel-new", Button)
        cancel_button.styles.display = (
            "block" if self.creating_task and self.task_state is not None else "none"
        )
        cancel_button.disabled = self.busy
        self.query_one("#task-select", Select).disabled = self.creating_task or self.busy or not bool(self.store.ids())

    @on(Select.Changed, "#task-select")
    def task_changed(self, event: Select.Changed) -> None:
        if self.busy or event.value in {Select.NULL, "none"}:
            return
        task_id = str(event.value)
        if self.task_state is not None and task_id == self.task_state.task_id:
            return
        try:
            task = self.store.load(task_id)
            self._attach(task)
            self.selected_run_index = None if task.phase == Phase.DONE else len(task.runs) - 1
            self.query_one("#goal", Input).value = task.goal
            self.creating_task = False
            self._sync_creation_controls()
            self._set_status(f"Загружен checkpoint {task_id}")
        except TransitionError as error:
            self._set_status(str(error))
            self._update_task_options()
            return
        self._refresh()

    @on(Button.Pressed, "#cancel-new")
    def cancel_new_task(self) -> None:
        if self.task_state is None:
            return
        self.creating_task = False
        self.query_one("#goal", Input).value = self.task_state.goal
        self._sync_creation_controls()
        self._set_status(f"Создание отменено · активна {self.task_state.task_id}")
        self._refresh()

    @on(Button.Pressed, "#new")
    def new_task(self) -> None:
        if not self.creating_task:
            self.creating_task = True
            goal = self.query_one("#goal", Input)
            goal.value = ""
            self._sync_creation_controls()
            goal.focus()
            self._set_status("Введите цель новой задачи и нажмите «Создать»")
            self._refresh()
            return
        try:
            task = self.store.create(self._next_id(), self.query_one("#goal", Input).value)
            self._attach(task)
            self.selected_run_index = 0
            self.creating_task = False
            self._update_task_options()
            self._sync_creation_controls()
            self._set_status("Начата planning-сессия; ответы модели не изменяют фазу")
        except (TransitionError, ValueError) as error:
            self._set_status(str(error))
        self._refresh()

    @on(Button.Pressed, "#send")
    async def send_message(self) -> None:
        if self.busy or self.task_state is None:
            return
        if self.agent is None:
            self._set_status("API не настроен: добавь ключ в .env")
            return
        if self.selected_run_index != len(self.task_state.runs) - 1 or self.task_state.phase == Phase.DONE:
            self._set_status("Писать можно только в активный запуск; прошлые посещения доступны для просмотра")
            return
        input_widget = self.query_one("#message", Input)
        message = input_widget.value.strip()
        if not message:
            self._set_status("Введите сообщение")
            return
        input_widget.value = ""
        await self._send_text(message)

    @on(Input.Submitted, "#message")
    async def message_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        await self.send_message()

    async def _send_text(self, message: str) -> bool:
        if self.busy or self.task_state is None or self.agent is None:
            return False
        self.busy = True
        self.pending_message = message
        self._set_status(f"Сообщение внутри {self.task_state.phase.value}: модель отвечает…")
        self._refresh()
        try:
            phase_before = self.task_state.phase
            await self.agent.respond(message)
            self._set_status(f"Ход сохранён внутри {phase_before.value}; переход не выполнялся")
            return True
        except AgentError as error:
            self.query_one("#message", Input).value = message
            self._set_status(str(error))
            return False
        finally:
            self.busy = False
            self.pending_message = ""
            self._refresh()

    async def action_send_stage_message(self) -> None:
        await self.send_message()

    @on(Button.Pressed, "#primary-transition")
    async def primary_transition(self) -> None:
        if self.task_state is None:
            return
        phase = self.task_state.phase
        if phase == Phase.PLANNING:
            await self._manual_event(Event.PLAN_APPROVED, "План принят пользователем")
        elif phase == Phase.EXECUTION:
            await self._manual_event(Event.EXECUTION_SUBMITTED, "Результат отправлен на validation")
        elif phase == Phase.VALIDATION:
            event = Event.VALIDATION_ACCEPTED if self.task_state.validation_passed else Event.VALIDATION_SENT_TO_REVISION
            label = "Validation принята; задача завершена" if self.task_state.validation_passed else "Результат отправлен на revision"
            await self._manual_event(event, label)
        elif phase == Phase.REVISION:
            await self._manual_event(Event.REVISION_SUBMITTED, "Доработка отправлена на повторную validation")

    @on(Button.Pressed, "#secondary-transition")
    async def secondary_transition(self) -> None:
        if self.task_state is not None and self.task_state.phase == Phase.EXECUTION:
            await self._manual_event(Event.RETURNED_TO_PLANNING, "Задача возвращена в planning")

    async def _manual_event(self, event: Event, success: str) -> None:
        if self.task_state is None:
            return
        before = self.task_state.phase
        candidate = copy.deepcopy(self.task_state)
        try:
            self.controller.apply(candidate, event)
            self.store.save(candidate)
            self._attach(candidate)
            self.selected_run_index = None if candidate.phase == Phase.DONE else len(candidate.runs) - 1
            self.store.append_event(
                candidate,
                "transition_applied",
                phase_before=before.value,
                transition_event=event.value,
                actor="user",
            )
            self._set_status(success)
        except TransitionError as error:
            self._record_rejection(str(error), requested_event=event.value)
        self._refresh()

    @on(Button.Pressed, ".phase")
    def select_stage(self, event: Button.Pressed) -> None:
        phase = Phase(event.button.id.removeprefix("stage-"))
        if phase == Phase.DONE:
            if self.task_state is not None and self.task_state.phase == Phase.DONE:
                self.selected_run_index = None
        elif self.task_state is not None:
            matching = [index for index, run in enumerate(self.task_state.runs) if run.phase == phase]
            if matching:
                self.selected_run_index = matching[-1]
        self._refresh()

    @on(Button.Pressed, "#previous-run")
    def previous_run(self) -> None:
        if self.task_state is None:
            return
        if self.selected_run_index is None:
            self.selected_run_index = len(self.task_state.runs) - 1
        elif self.selected_run_index > 0:
            self.selected_run_index -= 1
        self._refresh()

    @on(Button.Pressed, "#next-run")
    def next_run(self) -> None:
        if self.task_state is None or self.selected_run_index is None:
            return
        if self.selected_run_index < len(self.task_state.runs) - 1:
            self.selected_run_index += 1
        elif self.task_state.phase == Phase.DONE:
            self.selected_run_index = None
        self._refresh()

    @on(Button.Pressed, "#request-transition")
    async def request_transition(self) -> None:
        if self.task_state is None:
            return
        value = self.query_one("#target-phase", Select).value
        if value == Select.NULL:
            self._set_status("Выберите целевую фазу")
            return
        await self._attempt_target(Phase(str(value)))

    async def _attempt_target(self, target: Phase) -> None:
        if self.task_state is None:
            return
        before = self.task_state.phase
        candidate = copy.deepcopy(self.task_state)
        try:
            self.controller.request_target(candidate, target)
            self.store.save(candidate)
            self._attach(candidate)
            self.selected_run_index = None if candidate.phase == Phase.DONE else len(candidate.runs) - 1
            self.store.append_event(
                candidate,
                "transition_applied",
                phase_before=before.value,
                requested_target=target.value,
                actor="user_target_request",
            )
            self._set_status(f"Переход {before.value} → {target.value} выполнен")
        except TransitionError as error:
            self._record_rejection(str(error), requested_target=target.value)
        self._refresh()

    def _record_rejection(self, message: str, **details: object) -> None:
        if self.task_state is not None:
            self.store.append_event(self.task_state, "transition_rejected", reason=message, **details)
        self._set_status(f"ОТКЛОНЕНО · {message}")

    @on(Button.Pressed, "#pause")
    def pause(self) -> None:
        if self.task_state is None:
            return
        try:
            self.store.pause(self.task_state)
            self._set_status(f"Сессия {self.task_state.phase.value} сохранена на паузе")
        except TransitionError as error:
            self._record_rejection(str(error), operation="pause")
        self._refresh()

    @on(Button.Pressed, "#resume")
    def resume(self) -> None:
        if self.task_state is None:
            return
        try:
            self.store.resume(self.task_state)
            self._set_status(f"Продолжена та же сессия {self.task_state.phase.value}")
        except TransitionError as error:
            self._record_rejection(str(error), operation="resume")
        self._refresh()

    def _refresh(self) -> None:
        task = self.task_state
        self._sync_creation_controls()
        self.query_one("#api-loading", LoadingIndicator).styles.display = "block" if self.busy else "none"
        self.query_one("#api-status", Static).update("DeepSeek отвечает…" if self.busy else "")
        selected_phase = (
            Phase.DONE if task and self.selected_run_index is None else
            task.runs[self.selected_run_index].phase if task else Phase.PLANNING
        )
        for phase in Phase:
            widget = self.query_one(f"#stage-{phase.value}", Button)
            classes = "phase"
            if task and phase == task.phase:
                classes += " active"
            if phase == selected_phase:
                classes += " viewing"
            widget.set_classes(classes)
        if task is None:
            for selector, text in (
                ("#state", "Введите цель и создайте задачу"),
                ("#artifact", "—"),
                ("#prompt", "—"),
                ("#transitions", "Журнал появится после создания задачи"),
                ("#run-route", "Маршрут пока пуст"),
            ):
                log = self.query_one(selector, RichLog)
                log.clear()
                log.write(text)
            for selector in ("#primary-transition", "#secondary-transition", "#pause", "#resume", "#send", "#request-transition"):
                self.query_one(selector, Button).disabled = True
            self.query_one("#guard-hint", Static).update("Сначала создайте задачу")
            return
        allowed = ", ".join(item.value for item in self.controller.allowed_targets(task.phase)) or "нет"
        active_turns = 0 if task.phase == Phase.DONE else task.active_run.turns
        active_tokens = None if task.phase == Phase.DONE else task.active_run
        total_tokens = sum(run.total_tokens for run in task.runs)
        state_log = self.query_one("#state", RichLog)
        state_log.clear()
        state_log.write(
            f"Задача: {task.task_id}\nАктивная фаза: {task.phase.value}\nСтатус: {task.status.value}\n"
            f"Ожидается: {task.expected_action}\nРазрешённые цели: {allowed}\n"
            f"Запусков стадий: {len(task.runs)} · Ходов в активном: {active_turns}\n"
            f"Токены активного запуска: {active_tokens.total_tokens if active_tokens else 0} "
            f"(input {active_tokens.input_tokens if active_tokens else 0} + "
            f"output {active_tokens.output_tokens if active_tokens else 0})\n"
            f"Последний prompt: {active_tokens.last_prompt_tokens if active_tokens else 0} · "
            f"Всего по задаче: {total_tokens}\n"
            "Лимит ответа API: собственный лимит не задан\n"
            f"Доработки: {task.revision_count}/2\n"
            f"Последняя ошибка: {task.last_error or '—'}"
        )
        self._render_selected_stage()
        self._render_route()
        self._render_transition_log()
        self._update_task_options()
        primary = self.query_one("#primary-transition", Button)
        secondary = self.query_one("#secondary-transition", Button)
        has_active_messages = task.phase != Phase.DONE and bool(task.active_run.messages)
        labels = {
            Phase.PLANNING: "Утвердить план",
            Phase.EXECUTION: "Отправить на проверку",
            Phase.VALIDATION: (
                "Проверка не выполнена" if task.validation_passed is None
                else "Завершить" if task.validation_passed
                else "На доработку"
            ),
            Phase.REVISION: "Повторная проверка",
            Phase.DONE: "Завершено",
        }
        primary.label = labels[task.phase]
        ready_for_primary = {
            Phase.PLANNING: bool(task.plan and has_active_messages),
            Phase.EXECUTION: bool(task.execution_result.strip()),
            Phase.VALIDATION: task.validation_passed is not None,
            Phase.REVISION: bool(has_active_messages and task.execution_result.strip()),
            Phase.DONE: False,
        }[task.phase]
        primary.disabled = self.busy or self.creating_task or task.status == Status.PAUSED or not ready_for_primary
        secondary.label = "Вернуть к плану"
        secondary.disabled = self.busy or self.creating_task or task.status == Status.PAUSED or task.phase != Phase.EXECUTION
        self.query_one("#send", Button).disabled = (
            self.busy
            or self.creating_task
            or task.status == Status.PAUSED
            or task.phase == Phase.DONE
            or self.selected_run_index != len(task.runs) - 1
        )
        self.query_one("#pause", Button).disabled = task.phase == Phase.DONE or task.status == Status.PAUSED or self.busy or self.creating_task
        self.query_one("#resume", Button).disabled = task.status != Status.PAUSED or self.busy or self.creating_task
        self.query_one("#request-transition", Button).disabled = self.busy or self.creating_task or task.status == Status.PAUSED
        guard = "Создание новой задачи: завершите или отмените ввод цели" if self.creating_task else self._guard_message(task)
        self.query_one("#guard-hint", Static).update(guard)
        placeholders = {
            Phase.PLANNING: "Уточните план или попросите новую версию…",
            Phase.EXECUTION: "Уточните или доработайте результат…",
            Phase.VALIDATION: "Попросите проверить результат или уточнить замечания…",
            Phase.REVISION: "Обсудите исправление замечаний…",
            Phase.DONE: "Задача завершена",
        }
        self.query_one("#message", Input).placeholder = placeholders[task.phase]

    @staticmethod
    def _guard_message(task: TaskState) -> str:
        if task.status == Status.PAUSED:
            return "Переходы заблокированы: сначала продолжите задачу"
        if task.phase == Phase.PLANNING:
            return "План готов к утверждению" if task.plan and task.active_run.messages else "Недоступно: сначала сформируйте план в чате"
        if task.phase == Phase.EXECUTION:
            return "Результат готов к проверке" if task.execution_result.strip() else "Недоступно: сначала выполните план в чате"
        if task.phase == Phase.VALIDATION:
            if task.validation_passed is None:
                return "Недоступно: сначала выполните validation в чате"
            if task.validation_passed:
                return "Проверка пройдена: результат можно принять"
            return f"Найдено замечаний: {len(task.validation_issues)} · отправьте на доработку"
        if task.phase == Phase.REVISION:
            return "Исправление готово к повторной проверке" if task.active_run.messages else "Недоступно: сначала исправьте замечания в чате"
        return "Задача завершена"

    def _render_route(self) -> None:
        task = self.task_state
        route = self.query_one("#run-route", RichLog)
        route.clear()
        if task is None:
            route.write("Маршрут пока пуст")
            return
        labels: list[str] = []
        for index, run in enumerate(task.runs):
            is_active = task.phase != Phase.DONE and index == len(task.runs) - 1
            if is_active:
                symbol = "Ⅱ" if task.status == Status.PAUSED else "●"
            elif run.phase == Phase.VALIDATION and index + 1 < len(task.runs) and task.runs[index + 1].phase == Phase.REVISION:
                symbol = "✗"
            else:
                symbol = "✓"
            viewed = " ◉" if self.selected_run_index == index else ""
            labels.append(f"{symbol} {run.phase}#{run.visit}{viewed}")
        if task.phase == Phase.DONE:
            labels.append("✓ done")
        route.write("  →  ".join(labels))

    def _render_transition_log(self) -> None:
        task = self.task_state
        log = self.query_one("#transitions", RichLog)
        log.clear()
        if task is None:
            log.write("Журнал появится после создания задачи")
            return
        events = self.store.events(task.task_id)
        if not events:
            log.write("Событий пока нет")
            return
        names = {
            "task_created": "создана задача",
            "stage_turn_completed": "завершён ход LLM",
            "stage_turn_failed": "ошибка хода LLM",
            "transition_applied": "переход выполнен",
            "transition_rejected": "переход отклонён",
            "paused": "пауза",
            "resumed": "продолжено",
        }
        for item in events:
            timestamp = str(item.get("timestamp", ""))[11:19]
            event = str(item.get("event", "событие"))
            details = ""
            if event == "transition_applied":
                details = f" · {item.get('phase_before', '?')} → {item.get('phase', '?')} · {item.get('actor', 'user')}"
            elif event == "transition_rejected":
                details = f" · {item.get('reason', '')}"
            elif event in {"stage_turn_completed", "stage_turn_failed"}:
                details = f" · {item.get('stage', item.get('phase', '?'))}"
            log.write(f"{timestamp} · {names.get(event, event)}{details}")

    def _render_selected_stage(self) -> None:
        task = self.task_state
        if task is None:
            return
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        if self.selected_run_index is None:
            chat.write("Завершённая задача не имеет отдельной LLM-сессии.")
            artifact = task.final_result or "—"
            title = "ИТОГ · DONE"
        else:
            run = task.runs[self.selected_run_index]
            for message in run.messages:
                author = "ВЫ" if message.role == "user" else "АССИСТЕНТ"
                chat.write(f"{author}\n{message.content}\n")
            if not run.messages:
                chat.write("В этом запуске ещё нет сообщений.")
            is_active_view = (
                self.selected_run_index == len(task.runs) - 1
                and task.phase != Phase.DONE
            )
            if is_active_view and self.pending_message:
                chat.write(f"ВЫ\n{self.pending_message}\n\nАССИСТЕНТ\nМодель работает…")
            elif is_active_view and task.status == Status.FAILED and run.last_request_messages:
                failed_message = next(
                    (
                        item["content"]
                        for item in reversed(run.last_request_messages)
                        if item["role"] == "user"
                    ),
                    "",
                )
                if failed_message:
                    chat.write(
                        f"НЕЗАВЕРШЁННЫЙ ХОД\n{failed_message}\n\n"
                        f"ОШИБКА\n{task.last_error}"
                    )
            artifact = run.artifact or "—"
            title = f"ЗАПУСК {self.selected_run_index + 1}/{len(task.runs)} · {run.phase.value.upper()} #{run.visit}"
        self.query_one("#chat-title", Static).update(title)
        artifact_log = self.query_one("#artifact", RichLog)
        artifact_log.clear()
        artifact_log.write(artifact)
        prompt_run = task.runs[-1] if self.selected_run_index is None else task.runs[self.selected_run_index]
        request_messages = prompt_run.last_request_messages
        if not request_messages and prompt_run is task.runs[-1]:
            request_messages = task.last_request_messages
        request = "\n\n".join(
            f"[{item['role'].upper()}]\n{item['content']}"
            for item in request_messages
        ) or "—"
        response = prompt_run.last_response_raw.strip()
        response_label = "СЫРОЙ ОТВЕТ МОДЕЛИ"
        if not response and prompt_run.messages and prompt_run.messages[-1].role == "assistant":
            response = prompt_run.messages[-1].content
            response_label = "ОТВЕТ МОДЕЛИ · сохранён до обновления формата"
        finish_reason = prompt_run.last_finish_reason or "—"
        prompt = (
            f"ФАКТИЧЕСКИЙ ЗАПРОС\n\n{request}\n\n"
            f"{response_label}\n\n{response or '—'}\n\n"
            f"FINISH_REASON · {finish_reason}"
        )
        prompt_log = self.query_one("#prompt", RichLog)
        prompt_log.clear()
        prompt_log.write(prompt)
        self.query_one("#previous-run", Button).disabled = self.selected_run_index == 0
        self.query_one("#next-run", Button).disabled = (
            self.selected_run_index is None
            or (
                self.selected_run_index == len(task.runs) - 1
                and task.phase != Phase.DONE
            )
        )

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="День 15: управляемые сессии и переходы")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ControlledTransitionsApp(args.data_dir).run()
