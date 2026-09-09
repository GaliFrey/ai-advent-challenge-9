#!/usr/bin/env python3
"""Две независимые постоянные LLM-сессии в одном TUI."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, Label, RichLog, Static

from agent import Agent, AgentConfig, AgentError, AgentStats, Usage, load_config
from storage import HistoryStorageError, JsonHistoryStore


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY_DIR = ROOT / "data"
SIDES = ("left", "right")


def number(value: int | None) -> str:
    return "—" if value is None else str(value)


def stats_text(config: AgentConfig, stats: AgentStats) -> str:
    last = stats.last_usage or Usage(None, None, None)
    elapsed = "—" if stats.last_elapsed_seconds is None else f"{stats.last_elapsed_seconds:.2f} с"
    finish = stats.last_finish_reason or "—"
    return (
        f"Модель: {config.model}\n"
        f"История: {stats.history_messages} сообщ.  ·  "
        f"Запуск: {stats.successful_requests}/{stats.attempts} запросов\n"
        f"Последний вызов: вход {number(last.input_tokens)}  ·  выход {number(last.output_tokens)}  "
        f"·  всего {number(last.total_tokens)}\n"
        f"Токены запуска: вход {stats.session_input_tokens}  ·  "
        f"выход {stats.session_output_tokens}  ·  всего {stats.session_total_tokens}\n"
        f"Время: {elapsed}  ·  finish_reason: {finish}"
    )


class DualAgentApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 7 — постоянная история двух агентов"
    BINDINGS = [("ctrl+q", "quit", "Выйти")]

    def __init__(
        self,
        config: AgentConfig,
        *,
        history_dir: Path = DEFAULT_HISTORY_DIR,
        transport=None,
    ):
        super().__init__()
        self.config = config
        self.history_dir = history_dir
        self.transport = transport
        self.client: httpx.AsyncClient | None = None
        self.agents: dict[str, Agent] = {}
        self.busy = {side: False for side in SIDES}
        self.clear_armed = {side: False for side in SIDES}
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        with Horizontal(id="brand"):
            yield Label("DUAL MEMORY", id="title")
            yield Label("ДЕНЬ 07  /  ДВЕ ПОСТОЯННЫЕ СЕССИИ", id="subtitle")
        with Horizontal(id="sessions"):
            for side, label in zip(SIDES, ("СЕССИЯ A  ·  LEFT", "СЕССИЯ B  ·  RIGHT"), strict=True):
                with Vertical(id=f"panel-{side}", classes="session-panel"):
                    yield Label(label, classes="session-title")
                    yield RichLog(id=f"chat-{side}", classes="chat", wrap=True, markup=False)
                    yield Static("Статистика появится после запуска.", id=f"stats-{side}", classes="stats", markup=False)
                    yield Input(placeholder="Введите сообщение…", id=f"input-{side}")
                    with Horizontal(classes="actions"):
                        yield Button("Отправить", id=f"send-{side}", variant="primary")
                        yield Button("Очистить историю", id=f"clear-{side}", classes="clear")
                    yield Static("Готово", id=f"status-{side}", classes="status", markup=False)
        yield Static(
            "Вся переписка каждой панели хранится отдельно и передаётся модели при каждом новом запросе.",
            id="legend",
            markup=False,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(90, connect=20),
            transport=self.transport,
            follow_redirects=False,
        )
        self.agents = {
            side: Agent(self.config, self.client, JsonHistoryStore(self.history_dir, side))
            for side in SIDES
        }
        for side in SIDES:
            self.restore_chat(side)
            self.refresh_stats(side)
        self.query_one("#input-left", Input).focus()

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def restore_chat(self, side: str) -> None:
        log = self.query_one(f"#chat-{side}", RichLog)
        conversation = self.agents[side].messages[1:]
        if not conversation:
            log.write(Text("Новая сессия. Сохранённая история пуста.", style="#8294aa"))
            return
        log.write(Text(f"Восстановлено сообщений: {len(conversation)}", style="#8294aa"))
        for message in conversation:
            if message["role"] == "user":
                log.write(Text("Вы: ", style="bold #8ed6dc") + Text(message["content"]))
            else:
                log.write(Text("Агент: ", style="bold #b8d982") + Text(message["content"]))

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("send-"):
            self.start_request(button_id.removeprefix("send-"))
        elif button_id.startswith("clear-"):
            self.confirm_or_clear_history(button_id.removeprefix("clear-"))

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        if event.input.id and event.input.id.startswith("input-"):
            self.start_request(event.input.id.removeprefix("input-"))

    @on(Input.Changed)
    def handle_input_changed(self, event: Input.Changed) -> None:
        if event.input.id and event.input.id.startswith("input-") and event.value:
            self.disarm_clear(event.input.id.removeprefix("input-"))

    def start_request(self, side: str) -> None:
        if side not in SIDES or self.busy[side]:
            return
        self.disarm_clear(side)
        field = self.query_one(f"#input-{side}", Input)
        message = field.value.strip()
        if not message:
            self.query_one(f"#status-{side}", Static).update("Введите непустое сообщение")
            return
        field.value = ""
        self.query_one(f"#chat-{side}", RichLog).write(Text("Вы: ", style="bold #8ed6dc") + Text(message))
        self.set_busy(side, True)
        self.query_one(f"#status-{side}", Static).update("Запрос выполняется…")
        self.request_agent(side, message)

    def confirm_or_clear_history(self, side: str) -> None:
        if side not in SIDES or self.busy[side]:
            return
        if not self.clear_armed[side]:
            self.clear_armed[side] = True
            self.query_one(f"#clear-{side}", Button).label = "ПОДТВЕРДИТЬ ОЧИСТКУ"
            self.query_one(f"#status-{side}", Static).update("Нажмите ещё раз; ввод текста отменит очистку")
            return
        self.disarm_clear(side)
        try:
            self.agents[side].clear_history()
        except AgentError as error:
            self.query_one(f"#status-{side}", Static).update(str(error))
            return
        log = self.query_one(f"#chat-{side}", RichLog)
        log.clear()
        log.write(Text("История этой панели очищена.", style="#8294aa"))
        self.query_one(f"#status-{side}", Static).update("История очищена")
        self.refresh_stats(side)

    def disarm_clear(self, side: str) -> None:
        if side not in SIDES or not self.clear_armed[side]:
            return
        self.clear_armed[side] = False
        self.query_one(f"#clear-{side}", Button).label = "Очистить историю"

    def set_busy(self, side: str, value: bool) -> None:
        self.busy[side] = value
        self.query_one(f"#input-{side}", Input).disabled = value
        self.query_one(f"#send-{side}", Button).disabled = value
        self.query_one(f"#clear-{side}", Button).disabled = value

    @work
    async def request_agent(self, side: str, message: str) -> None:
        try:
            reply = await self.agents[side].ask(message)
            self.query_one(f"#chat-{side}", RichLog).write(
                Text("Агент: ", style="bold #b8d982") + Text(reply.text)
            )
            self.query_one(f"#status-{side}", Static).update("Ответ получен и сохранён")
        except AgentError as error:
            self.query_one(f"#chat-{side}", RichLog).write(
                Text("Ошибка: " + str(error), style="bold #e68a8a")
            )
            self.query_one(f"#status-{side}", Static).update(str(error))
        except Exception:
            self.query_one(f"#chat-{side}", RichLog).write(
                Text("Внутренняя ошибка без автоматического повтора", style="bold #e68a8a")
            )
            self.query_one(f"#status-{side}", Static).update("Внутренняя ошибка")
        finally:
            self.refresh_stats(side)
            self.set_busy(side, False)
            self.query_one(f"#input-{side}", Input).focus()

    def refresh_stats(self, side: str) -> None:
        self.query_one(f"#stats-{side}", Static).update(
            stats_text(self.config, self.agents[side].stats)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 7: две независимые LLM-сессии с историей между запусками."
    )
    parser.parse_args()
    try:
        config = load_config()
        config.check()
        for side in SIDES:
            JsonHistoryStore(DEFAULT_HISTORY_DIR, side).load()
    except (ValueError, HistoryStorageError) as error:
        parser.error(str(error))
    DualAgentApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
