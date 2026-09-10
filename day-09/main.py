#!/usr/bin/env python3
"""Двухпанельный TUI: полная история против recursive summary."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from pathlib import Path

import httpx
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, Label, RichLog, Select, Static

from agent import (
    Agent,
    AgentConfig,
    AgentReply,
    SUPPORTED_MODELS,
    load_config,
)
from demo import DEMO_CHECK_MESSAGE, DEMO_MESSAGES, QualityResult, evaluate_answer
from memory import FullHistoryMemory, SummaryMemory
from report import DEFAULT_RESULTS_DIR, ReportError, save_demo_report


SIDES = ("full", "summary")
RECENT_OPTIONS = (4, 6, 8)
SUMMARY_INTERVAL = 10


def number(value: int | None) -> str:
    return "—" if value is None else f"{value:,}".replace(",", " ")


def quality_text(result: QualityResult | None) -> str:
    return "—" if result is None else f"{result.score}/{result.total}"


def savings_text(full_tokens: int, summary_tokens: int) -> str:
    if full_tokens <= 0:
        return "Экономия: —"
    delta = (full_tokens - summary_tokens) / full_tokens * 100
    if delta >= 0:
        return f"Экономия: {delta:.1f}%"
    return f"Перерасход: {-delta:.1f}%"


def stats_text(
    side: str,
    agent: Agent,
    *,
    full_total_tokens: int,
    quality: QualityResult | None,
) -> str:
    stats = agent.stats
    memory = agent.memory
    base = (
        f"Модель: {agent.config.model}  ·  Качество: {quality_text(quality)}\n"
        f"Диалог: {memory.total_message_count} сообщ.  ·  "
        f"Последний prompt: {number(stats.last_prompt_tokens)} токенов\n"
        f"Рабочие запросы: {stats.successful_chat_requests}/{stats.chat_attempts}  ·  "
        f"вход {number(stats.chat_input_tokens)}  ·  выход {number(stats.chat_output_tokens)}\n"
    )
    if side == "full":
        return (
            base
            + f"Контекст: все {memory.raw_message_count} исходных сообщений\n"
            + f"Общий расход: {number(stats.total_tokens)} токенов"
        )
    summary_length = len(memory.summary or "")
    return (
        base
        + f"Контекст: summary v{getattr(memory, 'summary_version', 0)} + "
        f"{memory.raw_message_count} исходных сообщ.\n"
        + f"Сжато: {memory.summarized_message_count} сообщ.  ·  "
        f"summary {number(summary_length)} симв.  ·  "
        f"следующее через {getattr(memory, 'messages_until_summary', '—')} сообщ.\n"
        + f"Суммаризация: {stats.successful_summary_requests}/{stats.summary_attempts}  ·  "
        f"{number(stats.summary_total_tokens)} токенов\n"
        + f"Общий расход: {number(stats.total_tokens)}  ·  "
        + savings_text(full_total_tokens, stats.total_tokens)
    )


class ContextComparisonApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 9 — полная история и summary"
    BINDINGS = [
        ("ctrl+q", "quit", "Выйти"),
        ("ctrl+d", "start_demo", "Демо"),
    ]

    def __init__(
        self,
        config: AgentConfig,
        *,
        transport=None,
        report_dir: Path = DEFAULT_RESULTS_DIR,
    ) -> None:
        super().__init__()
        self.base_config = config
        self.transport = transport
        self.report_dir = report_dir
        self.client: httpx.AsyncClient | None = None
        self.agents: dict[str, Agent] = {}
        self.busy = False
        self.clear_armed = False
        self.quality: dict[str, QualityResult | None] = {side: None for side in SIDES}
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        models = tuple(dict.fromkeys((self.base_config.model, *SUPPORTED_MODELS)))
        with Horizontal(id="brand"):
            yield Label("CONTEXT LAB", id="title")
            yield Label("ДЕНЬ 09  /  ОДИН ДИАЛОГ, ДВА КОНТЕКСТА", id="subtitle")
        with Horizontal(id="configuration"):
            yield Label("Модель", classes="config-label")
            yield Select(
                [(model, model) for model in models],
                value=self.base_config.model,
                id="model-select",
                allow_blank=False,
            )
            yield Label("Последние N сообщений", classes="config-label n-label")
            yield Select(
                [(str(value), value) for value in RECENT_OPTIONS],
                value=4,
                id="recent-select",
                allow_blank=False,
            )
            yield Static(
                f"Суммаризация после каждых {SUMMARY_INTERVAL} новых сообщений",
                id="interval-label",
            )
        with Horizontal(id="sessions"):
            for side, label in zip(
                SIDES,
                ("БЕЗ СЖАТИЯ  ·  ВСЯ ИСТОРИЯ", "СО СЖАТИЕМ  ·  RECURSIVE SUMMARY"),
                strict=True,
            ):
                with Vertical(id=f"panel-{side}", classes="session-panel"):
                    yield Label(label, classes="session-title")
                    yield RichLog(id=f"chat-{side}", classes="chat", wrap=True, markup=False)
                    yield Static("Статистика появится после запуска.", id=f"stats-{side}", classes="stats")
        yield Input(placeholder="Одно сообщение будет отправлено обоим агентам…", id="message-input")
        with Horizontal(id="actions"):
            yield Button("Отправить обоим", id="send", variant="primary")
            yield Button("Демо · 35 API-вызовов", id="demo")
            yield Button("Показать summary", id="show-summary")
            yield Button("Очистить обе истории", id="clear", classes="clear")
        yield Static(
            "Одинаковые модель, параметры и сообщения. Справа старые сообщения заменяются summary; "
            "его токены включены в общий расход.",
            id="legend",
        )
        yield Static("Готово", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(90, connect=20),
            transport=self.transport,
            follow_redirects=False,
        )
        self._create_agents()
        for side in SIDES:
            self.query_one(f"#chat-{side}", RichLog).write(
                Text("Новая сессия. История пуста.", style="#8294aa")
            )
        self.refresh_stats()
        self.query_one("#message-input", Input).focus()

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def _selected_model(self) -> str:
        value = self.query_one("#model-select", Select).value
        return str(value)

    def _selected_recent(self) -> int:
        value = self.query_one("#recent-select", Select).value
        return int(value)

    def _create_agents(self) -> None:
        if self.client is None:
            return
        config = replace(self.base_config, model=self._selected_model())
        self.agents = {
            "full": Agent(config, self.client, FullHistoryMemory()),
            "summary": Agent(
                config,
                self.client,
                SummaryMemory(
                    keep_recent=self._selected_recent(),
                    interval=SUMMARY_INTERVAL,
                ),
            ),
        }
        self.quality = {side: None for side in SIDES}

    @on(Select.Changed)
    def handle_select_changed(self, event: Select.Changed) -> None:
        if self.busy or not self.agents:
            return
        if any(agent.memory.total_message_count for agent in self.agents.values()):
            return
        self._create_agents()
        self.refresh_stats()

    @on(Input.Submitted, "#message-input")
    def handle_submit(self) -> None:
        self.start_message()

    @on(Input.Changed, "#message-input")
    def handle_input_changed(self, event: Input.Changed) -> None:
        if event.value:
            self._disarm_clear()

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send":
            self.start_message()
        elif button_id == "demo":
            self.action_start_demo()
        elif button_id == "show-summary":
            self.show_summary()
        elif button_id == "clear":
            self.confirm_or_clear()

    def start_message(self) -> None:
        if self.busy:
            return
        field = self.query_one("#message-input", Input)
        message = field.value.strip()
        if not message:
            self.query_one("#status", Static).update("Введите непустое сообщение")
            return
        field.value = ""
        self._disarm_clear()
        self._lock_configuration()
        self.set_busy(True)
        self.query_one("#status", Static).update("Оба запроса выполняются…")
        self.request_both(message)

    def action_start_demo(self) -> None:
        if self.busy:
            return
        if any(agent.memory.total_message_count for agent in self.agents.values()):
            self.query_one("#status", Static).update(
                "Для честного демо сначала очистите обе истории"
            )
            return
        self._disarm_clear()
        self._lock_configuration()
        self.set_busy(True)
        self.run_demo()

    def _write_user(self, message: str) -> None:
        for side in SIDES:
            self.query_one(f"#chat-{side}", RichLog).write(
                Text("Вы: ", style="bold #8ed6dc") + Text(message)
            )

    async def _ask_pair(self, message: str) -> dict[str, AgentReply | Exception]:
        self._write_user(message)
        raw_results = await asyncio.gather(
            *(self.agents[side].ask(message) for side in SIDES),
            return_exceptions=True,
        )
        results = dict(zip(SIDES, raw_results, strict=True))
        for side, result in results.items():
            log = self.query_one(f"#chat-{side}", RichLog)
            if isinstance(result, Exception):
                log.write(Text("Ошибка: " + str(result), style="bold #e68a8a"))
                continue
            log.write(Text("Агент: ", style="bold #b8d982") + Text(result.text))
            if result.compression is not None:
                event = result.compression
                log.write(Text(
                    f"SUMMARY v{event.version}: сжато {event.summarized_messages} сообщ., "
                    f"оставлено как есть {event.remaining_raw_messages}; "
                    f"затрачено {event.usage.total_tokens} токенов.",
                    style="bold #e7c47b",
                ))
            if result.compression_error:
                log.write(Text(
                    "Summary не обновлено: " + result.compression_error,
                    style="bold #e68a8a",
                ))
        self.refresh_stats()
        return results

    @staticmethod
    def _report_result(result: AgentReply | Exception) -> dict:
        if isinstance(result, Exception):
            return {"ok": False, "error": str(result)}
        compression = None
        if result.compression is not None:
            compression = asdict(result.compression)
        return {
            "ok": True,
            "answer": result.text,
            "usage": asdict(result.usage),
            "elapsed_seconds": result.elapsed_seconds,
            "finish_reason": result.finish_reason,
            "compression": compression,
            "compression_error": result.compression_error,
        }

    def _report_agents(self) -> dict[str, dict]:
        agents: dict[str, dict] = {}
        for side in SIDES:
            agent = self.agents[side]
            stats = asdict(agent.stats)
            stats.update({
                "chat_total_tokens": agent.stats.chat_total_tokens,
                "summary_total_tokens": agent.stats.summary_total_tokens,
                "total_tokens": agent.stats.total_tokens,
            })
            agents[side] = {
                "stats": stats,
                "memory": {
                    "total_message_count": agent.memory.total_message_count,
                    "raw_message_count": agent.memory.raw_message_count,
                    "summarized_message_count": agent.memory.summarized_message_count,
                    "summary": agent.memory.summary,
                    "summary_version": getattr(agent.memory, "summary_version", 0),
                    "messages_until_summary": getattr(
                        agent.memory, "messages_until_summary", None
                    ),
                },
                "quality": (
                    None if self.quality[side] is None else asdict(self.quality[side])
                ),
            }
        return agents

    def _save_demo_report(self, outcome: str, turns: list[dict]) -> Path:
        full_stats = self.agents["full"].stats
        summary_stats = self.agents["summary"].stats
        full_total = full_stats.total_tokens
        summary_total = summary_stats.total_tokens
        savings_percent = (
            None
            if full_total <= 0
            else (full_total - summary_total) / full_total * 100
        )
        return save_demo_report(
            {
                "kind": "day-09-summary-comparison",
                "outcome": outcome,
                "configuration": {
                    "model": self.agents["full"].config.model,
                    "temperature": self.agents["full"].config.temperature,
                    "max_tokens": self.agents["full"].config.max_tokens,
                    "summary_max_tokens": self.agents["summary"].config.summary_max_tokens,
                    "keep_recent_messages": self.agents["summary"].memory.keep_recent,
                    "summary_interval_messages": SUMMARY_INTERVAL,
                },
                "actual_api_attempts": (
                    full_stats.chat_attempts
                    + full_stats.summary_attempts
                    + summary_stats.chat_attempts
                    + summary_stats.summary_attempts
                ),
                "savings_percent": savings_percent,
                "turns": turns,
                "agents": self._report_agents(),
            },
            self.report_dir,
        )

    @work
    async def request_both(self, message: str) -> None:
        try:
            results = await self._ask_pair(message)
            failures = sum(isinstance(result, Exception) for result in results.values())
            status = "Оба ответа получены" if failures == 0 else f"Ошибок: {failures}"
            self.query_one("#status", Static).update(status)
        finally:
            self.set_busy(False)
            self.query_one("#message-input", Input).focus()

    @work
    async def run_demo(self) -> None:
        turns: list[dict] = []
        outcome = "failed"
        final_status = "Демо остановлено"
        try:
            for index, message in enumerate(DEMO_MESSAGES, start=1):
                self.query_one("#status", Static).update(
                    f"Демо: сообщение {index}/{len(DEMO_MESSAGES)}"
                )
                results = await self._ask_pair(message)
                turns.append({
                    "turn": index,
                    "kind": "dialogue",
                    "user_message": message,
                    "results": {
                        side: self._report_result(results[side]) for side in SIDES
                    },
                })
                if any(isinstance(result, Exception) for result in results.values()):
                    final_status = "Демо остановлено из-за ошибки рабочего запроса"
                    return

            self.query_one("#status", Static).update("Демо: контрольный вопрос")
            results = await self._ask_pair(DEMO_CHECK_MESSAGE)
            turns.append({
                "turn": len(DEMO_MESSAGES) + 1,
                "kind": "quality_check",
                "user_message": DEMO_CHECK_MESSAGE,
                "results": {side: self._report_result(results[side]) for side in SIDES},
            })
            failed_check = False
            for side, result in results.items():
                if isinstance(result, AgentReply):
                    self.quality[side] = evaluate_answer(result.text)
                    checks = ", ".join(
                        f"{name}={'OK' if passed else 'FAIL'}"
                        for name, passed in self.quality[side].checks
                    )
                    self.query_one(f"#chat-{side}", RichLog).write(Text(
                        f"КАЧЕСТВО: {self.quality[side].score}/{self.quality[side].total} · {checks}",
                        style="bold #e7c47b",
                    ))
                else:
                    failed_check = True
            self.refresh_stats()
            final_status = (
                "Контрольный вопрос завершился с ошибкой в одной из панелей"
                if failed_check
                else "Демо завершено: качество и общий расход показаны в обеих панелях"
            )
            has_summary_errors = any(
                result.get("compression_error")
                for turn in turns
                for result in turn["results"].values()
                if result.get("ok")
            )
            if failed_check:
                outcome = "failed"
            elif has_summary_errors:
                outcome = "completed_with_summary_errors"
            else:
                outcome = "completed"
        finally:
            try:
                report_path = self._save_demo_report(outcome, turns)
            except ReportError as error:
                final_status += f"; {error}"
            else:
                final_status += f"; отчёт: {report_path}"
                for side in SIDES:
                    self.query_one(f"#chat-{side}", RichLog).write(Text(
                        f"ОТЧЁТ: {report_path}",
                        style="bold #8ed6dc",
                    ))
            self.query_one("#status", Static).update(final_status)
            self.set_busy(False)
            self.query_one("#message-input", Input).focus()

    def show_summary(self) -> None:
        summary = self.agents["summary"].summary
        log = self.query_one("#chat-summary", RichLog)
        if summary:
            log.write(Text("ТЕКУЩЕЕ SUMMARY:\n", style="bold #e7c47b") + Text(summary))
            self.query_one("#status", Static).update("Summary выведено в правой панели")
        else:
            self.query_one("#status", Static).update("Summary ещё не создано")

    def confirm_or_clear(self) -> None:
        if self.busy:
            return
        if not self.clear_armed:
            self.clear_armed = True
            self.query_one("#clear", Button).label = "ПОДТВЕРДИТЬ ОЧИСТКУ"
            self.query_one("#status", Static).update("Нажмите ещё раз для очистки обеих историй")
            return
        self._disarm_clear()
        for agent in self.agents.values():
            agent.clear()
        self.quality = {side: None for side in SIDES}
        for side in SIDES:
            log = self.query_one(f"#chat-{side}", RichLog)
            log.clear()
            log.write(Text("Обе истории очищены.", style="#8294aa"))
        self._unlock_configuration()
        self.refresh_stats()
        self.query_one("#status", Static).update("Обе истории и статистика очищены")

    def _disarm_clear(self) -> None:
        if not self.clear_armed:
            return
        self.clear_armed = False
        self.query_one("#clear", Button).label = "Очистить обе истории"

    def _lock_configuration(self) -> None:
        self.query_one("#model-select", Select).disabled = True
        self.query_one("#recent-select", Select).disabled = True

    def _unlock_configuration(self) -> None:
        self.query_one("#model-select", Select).disabled = False
        self.query_one("#recent-select", Select).disabled = False

    def set_busy(self, value: bool) -> None:
        self.busy = value
        for selector in ("#message-input", "#send", "#demo", "#clear"):
            self.query_one(selector).disabled = value

    def refresh_stats(self) -> None:
        if not self.agents:
            return
        full_total = self.agents["full"].stats.total_tokens
        for side in SIDES:
            self.query_one(f"#stats-{side}", Static).update(stats_text(
                side,
                self.agents[side],
                full_total_tokens=full_total,
                quality=self.quality[side],
            ))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 9: сравнение полной истории и recursive summary в TUI."
    )
    parser.parse_args()
    try:
        config = load_config()
        config.check()
    except ValueError as error:
        parser.error(str(error))
    ContextComparisonApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
