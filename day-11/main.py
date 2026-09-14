#!/usr/bin/env python3
"""TUI для демонстрации трёх явных слоёв памяти."""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

import httpx
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, Label, RichLog, Select, Static, TabbedContent, TabPane

from agent import Agent, AgentConfig, AgentError, SUPPORTED_MODELS, load_config
from memory import MemoryError, MemoryLayers


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
CONTROL_QUESTION = "Какой цвет выбрать? Ответь одним словом."
LONG_TERM_DEMO_NOTE = "Любимый цвет пользователя — зелёный"
WORKING_DEMO_NOTE = "Текущая задача — выбрать цвет кнопки «Сохранить» в интерфейсе"
SHORT_TERM_DEMO_MESSAGE = "В следующих ответах называй цвет по-английски, одним словом."


def _lines(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{index:02d} {message['role'].upper()}\n{message['content']}"
        for index, message in enumerate(messages, start=1)
    )


class MemoryLayersApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 11 — Memory Layers"
    BINDINGS = [("ctrl+q", "quit", "Выйти"), ("ctrl+d", "start_demo", "Автодемо")]

    def __init__(
        self,
        config: AgentConfig,
        *,
        transport=None,
        data_dir: Path = DEFAULT_DATA_DIR,
    ) -> None:
        super().__init__()
        self.base_config = config
        self.transport = transport
        self.data_dir = data_dir
        self.client: httpx.AsyncClient | None = None
        self.memory: MemoryLayers | None = None
        self.agent: Agent | None = None
        self.total_tokens = 0
        self.busy = False
        self.clear_armed = False
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        models = tuple(dict.fromkeys((self.base_config.model, *SUPPORTED_MODELS)))
        with Horizontal(id="brand"):
            yield Label("MEMORY LAB", id="title")
            yield Label("ДЕНЬ 11  /  ЯВНЫЕ СЛОИ ПАМЯТИ", id="subtitle")
        with Horizontal(id="configuration"):
            yield Label("Модель", classes="config-label")
            yield Select([(item, item) for item in models], value=self.base_config.model, id="model", allow_blank=False)
            yield Label("Задача", classes="config-label spaced")
            yield Select([("task-01", "task-01")], value="task-01", id="task", allow_blank=False)
            yield Label("Сессия", classes="config-label spaced")
            yield Select(
                [("session-01", "session-01")],
                value=Select.NULL,
                id="session",
                prompt="—",
                allow_blank=True,
            )
            yield Button("Новая сессия", id="new-session")
            yield Button("Новая задача", id="new-task")
            yield Button("Автодемо · 6 API", id="demo", variant="primary")
        yield Static("", id="scope")
        with Horizontal(id="workspace"):
            with Vertical(id="chat-panel"):
                yield Label("ДИАЛОГ", classes="panel-title")
                yield RichLog(id="chat", classes="content", wrap=True, markup=False)
                yield Input(placeholder="Введите вопрос или заметку…", id="input")
                with Horizontal(id="actions"):
                    yield Button("Спросить", id="ask", variant="primary")
                    yield Button("В рабочую", id="save-working")
                    yield Button("В долговременную", id="save-long")
                    yield Button("Очистить", id="clear", classes="danger")
            with Vertical(id="memory-panel"):
                with TabbedContent(initial="memory-tab", id="right-tabs"):
                    with TabPane("Память", id="memory-tab"):
                        yield RichLog(id="memory", classes="content", wrap=True, markup=False)
                    with TabPane("Фактический prompt", id="prompt-tab"):
                        yield RichLog(id="prompt", classes="content", wrap=True, markup=False)
                yield Static("", id="stats")
        yield Static("Готово", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(90, connect=20),
            transport=self.transport,
            follow_redirects=False,
        )
        try:
            self.memory = MemoryLayers(self.data_dir)
            sessions = self.memory.session_ids()
            if sessions:
                self.memory.switch_session(sessions[-1])
        except MemoryError as error:
            self.query_one("#status", Static).update(str(error))
            return
        self._create_agent()
        self._update_task_options()
        self._update_session_options()
        self._render_chat("Загружена сохранённая сессия." if self.memory.short_term else "Новая сессия. Краткосрочная память пуста.")
        self.refresh_views()
        self.query_one("#input", Input).focus()

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def _create_agent(self) -> None:
        if self.client is None or self.memory is None:
            return
        config = replace(self.base_config, model=str(self.query_one("#model", Select).value))
        self.agent = Agent(config, self.client, self.memory)

    def _update_task_options(self) -> None:
        if self.memory is None:
            return
        selector = self.query_one("#task", Select)
        with self.prevent(Select.Changed):
            selector.set_options([(name, name) for name in self.memory.task_ids()])
            selector.value = self.memory.task_id

    def _update_session_options(self) -> None:
        if self.memory is None:
            return
        names = set(self.memory.session_ids())
        names.add(self.memory.session_id)
        selector = self.query_one("#session", Select)
        with self.prevent(Select.Changed):
            selector.set_options([(name, name) for name in sorted(names)])
            selector.value = self.memory.session_id

    @on(Select.Changed, "#model")
    def model_changed(self) -> None:
        if not self.busy and self.memory is not None:
            self._create_agent()
            self.refresh_views()

    @on(Select.Changed, "#task")
    def task_changed(self, event: Select.Changed) -> None:
        if self.busy or self.memory is None or event.value == Select.NULL:
            return
        task_id = str(event.value)
        if task_id == self.memory.task_id:
            return
        self._switch_task(task_id)

    @on(Select.Changed, "#session")
    def session_changed(self, event: Select.Changed) -> None:
        if self.busy or self.memory is None:
            return
        if event.value == Select.NULL:
            self._update_session_options()
            return
        session_id = str(event.value)
        if session_id == self.memory.session_id:
            return
        try:
            self.memory.switch_session(session_id)
        except MemoryError as error:
            self._status(str(error))
            self._update_session_options()
            return
        self._create_agent()
        self._render_chat("Загружена сохранённая сессия.")
        self._status(f"Активна сессия {session_id}")
        self.refresh_views()

    @on(Input.Submitted, "#input")
    def input_submitted(self) -> None:
        self.start_ask()

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "ask": self.start_ask,
            "save-working": self.save_working,
            "save-long": self.save_long,
            "new-session": self.new_session,
            "new-task": self.new_task,
            "clear": self.clear_active,
            "demo": self.action_start_demo,
        }
        action = actions.get(event.button.id or "")
        if action is not None:
            action()

    def _input_value(self) -> str:
        return self.query_one("#input", Input).value.strip()

    def _consume_input(self) -> str:
        value = self._input_value()
        if value:
            self.query_one("#input", Input).value = ""
        return value

    def save_working(self) -> None:
        if self.busy or self.memory is None:
            return
        note = self._consume_input()
        if not note:
            self._status("Введите заметку для рабочей памяти")
            return
        try:
            self.memory.add_working(note)
        except MemoryError as error:
            self._status(str(error))
            return
        self.clear_armed = False
        self._status(f"Сохранено в рабочую память · {self.memory.task_id}")
        self.refresh_views()

    def save_long(self) -> None:
        if self.busy or self.memory is None:
            return
        note = self._consume_input()
        if not note:
            self._status("Введите заметку для долговременной памяти")
            return
        try:
            self.memory.add_long_term(note)
        except MemoryError as error:
            self._status(str(error))
            return
        self.clear_armed = False
        self._status(f"Сохранено в долговременную память · {self.memory.user_id}")
        self.refresh_views()

    def start_ask(self) -> None:
        if self.busy or self.agent is None:
            return
        message = self._consume_input()
        if not message:
            self._status("Введите вопрос")
            return
        self.clear_armed = False
        self.set_busy(True)
        self._status("Запрос выполняется")
        self.request_one(message)

    async def _ask(self, message: str) -> None:
        assert self.agent is not None
        log = self.query_one("#chat", RichLog)
        log.write(Text("Вы: ", style="bold #8ed6dc") + Text(message))
        try:
            reply = await self.agent.ask(message)
        except (AgentError, MemoryError) as error:
            log.write(Text("Ошибка: " + str(error), style="bold #e68a8a"))
            self.refresh_views()
            raise
        log.write(Text("Агент: ", style="bold #b8d982") + Text(reply.text))
        self.total_tokens += reply.usage.total_tokens
        log.write(Text(f"Контекст: short {len(self.memory.short_term)} · work {len(self.memory.working)} · long {len(self.memory.long_term)}", style="#8294aa"))
        self.refresh_views()

    @work
    async def request_one(self, message: str) -> None:
        try:
            await self._ask(message)
        except (AgentError, MemoryError):
            self._status("Запрос не сохранён в краткосрочную память")
        else:
            self._status("Ответ получен и сохранён в текущую сессию")
        finally:
            self.set_busy(False)
            self.query_one("#input", Input).focus()

    def new_session(self) -> None:
        if self.busy or self.memory is None:
            return
        session_id = self._next_session_id()
        try:
            self.memory.create_session(session_id)
        except MemoryError as error:
            self._status(str(error))
            return
        self._create_agent()
        self._update_session_options()
        self._render_chat("Новая сессия: short-term пуста, work/long сохранены.")
        self._status(f"Создана сессия {session_id}")
        self.refresh_views()

    def _next_session_id(self) -> str:
        assert self.memory is not None
        candidates = [self.memory.session_id, *self.memory.session_ids()]
        numbers = [
            int(match.group(1))
            for name in candidates
            if (match := re.fullmatch(r"session-(\d+)", name))
        ]
        return f"session-{max(numbers, default=0) + 1:02d}"

    def _switch_task(self, task_id: str, *, clear_chat: bool = True) -> None:
        assert self.memory is not None
        sessions = self.memory.session_ids(task_id)
        session_id = sessions[-1] if sessions else "session-01"
        try:
            self.memory.switch_task(task_id, session_id=session_id)
        except MemoryError as error:
            self._status(str(error))
            self._update_task_options()
            return
        self._create_agent()
        self._update_task_options()
        self._update_session_options()
        if clear_chat:
            self._render_chat("Задача переключена: загружена её последняя сессия.")
        else:
            self.query_one("#chat", RichLog).write(Text("Задача переключена: отдельные working и session, прежняя long-term.", style="#e7c47b"))
        self._status(f"Активна задача {task_id}")
        self.refresh_views()

    def new_task(self) -> None:
        if self.busy or self.memory is None:
            return
        task_id = self._next_task_id()
        self._create_task(task_id)

    def _create_task(self, task_id: str, *, clear_chat: bool = True) -> None:
        assert self.memory is not None
        try:
            self.memory.create_task(task_id)
        except MemoryError as error:
            self._status(str(error))
            return
        self._create_agent()
        self._update_task_options()
        self._update_session_options()
        if clear_chat:
            self._render_chat("Создана новая задача с пустыми working и short-term.")
        else:
            self.query_one("#chat", RichLog).write(
                Text("Создана новая задача: working и short-term пусты, long-term сохранена.", style="#e7c47b")
            )
        self._status(f"Создана задача {task_id}")
        self.refresh_views()

    def _next_task_id(self) -> str:
        assert self.memory is not None
        candidates = [self.memory.task_id, *self.memory.task_ids()]
        numbers = [
            int(match.group(1))
            for name in candidates
            if (match := re.fullmatch(r"task-(\d+)", name))
        ]
        return f"task-{max(numbers, default=0) + 1:02d}"

    def clear_active(self) -> None:
        if self.busy or self.memory is None:
            return
        if not self.clear_armed:
            self.clear_armed = True
            self._status("Нажмите «Очистить» ещё раз: будут очищены только активные short/work/long")
            return
        try:
            self.memory.clear_active()
        except MemoryError as error:
            self._status(str(error))
            return
        self.clear_armed = False
        self.query_one("#chat", RichLog).clear()
        self.query_one("#chat", RichLog).write(Text("Активные слои очищены.", style="#8294aa"))
        self._status("Активные short/work/long очищены")
        self.refresh_views()

    def set_busy(self, value: bool) -> None:
        self.busy = value
        for widget in self.query(Input):
            widget.disabled = value
        for widget in self.query(Button):
            widget.disabled = value
        for widget in self.query(Select):
            widget.disabled = value

    def _status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _render_chat(self, notice: str) -> None:
        assert self.memory is not None
        log = self.query_one("#chat", RichLog)
        log.clear()
        log.write(Text(notice, style="#8294aa"))
        for message in self.memory.short_term:
            label = "Вы: " if message["role"] == "user" else "Агент: "
            style = "bold #8ed6dc" if message["role"] == "user" else "bold #b8d982"
            log.write(Text(label, style=style) + Text(message["content"]))

    def refresh_views(self) -> None:
        if self.memory is None:
            return
        self.query_one("#scope", Static).update(
            f"USER {self.memory.user_id}   ·   TASK {self.memory.task_id}   ·   SESSION {self.memory.session_id}"
        )
        memory_log = self.query_one("#memory", RichLog)
        memory_log.clear()
        memory_log.write(Text(f"КРАТКОСРОЧНАЯ · {self.memory.session_id}", style="bold #8ed6dc"))
        if self.memory.short_term:
            for item in self.memory.short_term:
                memory_log.write(f"{item['role'].upper()}: {item['content']}")
        else:
            memory_log.write("(пусто)")
        memory_log.write(Text(f"\nРАБОЧАЯ · {self.memory.task_id}", style="bold #e7c47b"))
        for note in self.memory.working or ("(пусто)",):
            memory_log.write("• " + note)
        memory_log.write(Text(f"\nДОЛГОВРЕМЕННАЯ · {self.memory.user_id}", style="bold #b8d982"))
        for note in self.memory.long_term or ("(пусто)",):
            memory_log.write("• " + note)

        prompt_log = self.query_one("#prompt", RichLog)
        prompt_log.clear()
        if self.memory.last_prompt:
            prompt_log.write(_lines(list(self.memory.last_prompt)))
        else:
            prompt_log.write("Для этой сессии фактический prompt ещё не сохранялся.")
        self.query_one("#stats", Static).update(
            f"SHORT {len(self.memory.short_term)} сообщ.  ·  WORK {len(self.memory.working)}  ·  LONG {len(self.memory.long_term)}  ·  API {self.total_tokens} токенов"
        )

    def action_start_demo(self) -> None:
        if self.busy or self.memory is None:
            return
        if self.memory.short_term or self.memory.working or self.memory.long_term:
            self._status("Автодемо запускается только с пустой памятью; очистите активные слои")
            return
        self.set_busy(True)
        self.run_demo()

    @work
    async def run_demo(self) -> None:
        assert self.memory is not None
        try:
            self._status("Демо 1/6 · без памяти")
            await self._ask(CONTROL_QUESTION)
            self.new_session_for_demo()
            self.memory.add_long_term(LONG_TERM_DEMO_NOTE)
            self.refresh_views()
            self._status("Демо 2/6 · только long-term")
            await self._ask(CONTROL_QUESTION)
            self.new_session_for_demo()
            self.memory.add_working(WORKING_DEMO_NOTE)
            self.refresh_views()
            self._status("Демо 3/6 · long-term + working")
            await self._ask(CONTROL_QUESTION)
            self._status("Демо 4/6 · создаём short-term указание")
            await self._ask(SHORT_TERM_DEMO_MESSAGE)
            self._status("Демо 5/6 · все три слоя")
            await self._ask(CONTROL_QUESTION)
            self.new_session_for_demo()
            self._status("Демо 6/6 · новая сессия, short-term исчезла")
            await self._ask(CONTROL_QUESTION)
            old_task = self.memory.task_id
            self._create_task(self._next_task_id(), clear_chat=False)
            self.query_one("#chat", RichLog).write(
                Text(
                    f"ПРОВЕРКА SCOPE: {old_task} не перенесена; long-term пользователя сохранена.",
                    style="bold #b8d982",
                )
            )
            self._status("Демо завершено · short/work/long показаны раздельно")
        except (AgentError, MemoryError) as error:
            self._status("Демо остановлено: " + str(error))
        finally:
            self.set_busy(False)
            self.refresh_views()

    def new_session_for_demo(self) -> None:
        assert self.memory is not None
        session_id = self._next_session_id()
        self.memory.create_session(session_id)
        self._create_agent()
        self._update_session_options()
        self.query_one("#chat", RichLog).write(Text(f"НОВАЯ СЕССИЯ → {session_id}", style="bold #e7c47b"))
        self.refresh_views()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="День 11: три явных слоя памяти агента.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Каталог локальной памяти")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config()
        config.check()
    except ValueError as error:
        print(f"Ошибка конфигурации: {error}")
        return 2
    MemoryLayersApp(config, data_dir=args.data_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
