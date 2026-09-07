#!/usr/bin/env python3
"""Две независимые сессии LLM-агента в одном TUI."""

from __future__ import annotations

import argparse

import httpx
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, Label, RichLog, Static

from agent import Agent, AgentConfig, AgentError, AgentStats, Usage, load_config


SIDES = ("left", "right")


def number(value: int | None) -> str:
    return "—" if value is None else str(value)


def stats_text(config: AgentConfig, stats: AgentStats) -> str:
    last = stats.last_usage or Usage(None, None, None)
    elapsed = "—" if stats.last_elapsed_seconds is None else f"{stats.last_elapsed_seconds:.2f} с"
    finish = stats.last_finish_reason or "—"
    return (
        f"Модель: {config.model}\n"
        f"История: {stats.history_messages} сообщ.  ·  Запросы: {stats.successful_requests}/{stats.attempts}\n"
        f"Последний вызов: вход {number(last.input_tokens)}  ·  выход {number(last.output_tokens)}  "
        f"·  всего {number(last.total_tokens)}\n"
        f"Сессия: вход {stats.session_input_tokens}  ·  выход {stats.session_output_tokens}  "
        f"·  всего {stats.session_total_tokens}\n"
        f"Время: {elapsed}  ·  finish_reason: {finish}"
    )


class DualAgentApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 6 — два независимых агента"
    BINDINGS = [("ctrl+q", "quit", "Выйти")]

    def __init__(self, config: AgentConfig, *, transport=None):
        super().__init__()
        self.config = config
        self.transport = transport
        self.client: httpx.AsyncClient | None = None
        self.agents: dict[str, Agent] = {}
        self.busy = {side: False for side in SIDES}
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        with Horizontal(id="brand"):
            yield Label("DUAL AGENT", id="title")
            yield Label("ДЕНЬ 06  /  ДВЕ ИЗОЛИРОВАННЫЕ СЕССИИ", id="subtitle")
        with Horizontal(id="sessions"):
            for side, label in zip(SIDES, ("СЕССИЯ A", "СЕССИЯ B"), strict=True):
                with Vertical(id=f"panel-{side}", classes="session-panel"):
                    yield Label(label, classes="session-title")
                    yield RichLog(id=f"chat-{side}", classes="chat", wrap=True, markup=False)
                    yield Static("Статистика появится после запуска.", id=f"stats-{side}", classes="stats", markup=False)
                    yield Input(placeholder="Введите сообщение…", id=f"input-{side}")
                    yield Button("Отправить", id=f"send-{side}", variant="primary")
                    yield Static("Готово", id=f"status-{side}", classes="status", markup=False)
        yield Static(
            "Каждая панель хранит собственную историю. Входные токены последнего вызова включают весь отправленный контекст.",
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
        self.agents = {side: Agent(self.config, self.client) for side in SIDES}
        for side in SIDES:
            self.query_one(f"#chat-{side}", RichLog).write(
                Text("Новая независимая сессия. История пока пуста.", style="#8294aa")
            )
            self.refresh_stats(side)
        self.query_one("#input-left", Input).focus()

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("send-"):
            self.start_request(event.button.id.removeprefix("send-"))

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        if event.input.id and event.input.id.startswith("input-"):
            self.start_request(event.input.id.removeprefix("input-"))

    def start_request(self, side: str) -> None:
        if side not in SIDES or self.busy[side]:
            return
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

    def set_busy(self, side: str, value: bool) -> None:
        self.busy[side] = value
        self.query_one(f"#input-{side}", Input).disabled = value
        self.query_one(f"#send-{side}", Button).disabled = value

    @work
    async def request_agent(self, side: str, message: str) -> None:
        try:
            reply = await self.agents[side].ask(message)
            log = self.query_one(f"#chat-{side}", RichLog)
            log.write(Text("Агент: ", style="bold #b8d982") + Text(reply.text))
            if reply.finish_reason == "stop":
                self.query_one(f"#status-{side}", Static).update("Ответ получен")
            else:
                self.query_one(f"#status-{side}", Static).update(
                    "Ответ получен не полностью: " + reply.finish_reason
                )
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
        description="День 6: две независимые сессии одного типа агента в терминале."
    )
    parser.parse_args()
    try:
        config = load_config()
        config.check()
    except ValueError as error:
        parser.error(str(error))
    DualAgentApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
