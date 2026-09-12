#!/usr/bin/env python3
"""TUI с тремя независимыми стратегиями управления контекстом."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from agent import Agent, AgentConfig, AgentReply, SUPPORTED_MODELS, load_config
from demo import (
    BRANCH_A_MESSAGES,
    BRANCH_B_MESSAGES,
    CHECK_MESSAGE,
    COMMON_MESSAGES,
    EXPECTED_A,
    EXPECTED_B,
    LINEAR_MESSAGES,
    QualityResult,
    contamination_checks,
    evaluate_answer,
)
from memory import BranchingMemory, FactsMemory, SlidingWindowMemory
from report import DEFAULT_RESULTS_DIR, ReportError, save_demo_report


STRATEGIES = ("sliding", "facts", "branching")
RECENT_OPTIONS = (4, 6, 8)


def number(value: int | None) -> str:
    return "—" if value is None else f"{value:,}".replace(",", " ")


def quality_text(result: QualityResult | None) -> str:
    return "—" if result is None else f"{result.score}/{result.total}"


def _message_lines(messages: list[dict[str, str]] | tuple[dict[str, str], ...]) -> list[str]:
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        content = message["content"].replace("\n", " ")
        lines.append(f"{index:02d} {message['role'].upper()}: {content}")
    return lines


def context_text(agent: Agent) -> str:
    memory = agent.memory
    snapshot = memory.snapshot()
    lines: list[str] = []
    if isinstance(memory, SlidingWindowMemory):
        cutoff = snapshot["discarded_message_count"]
        lines.append(
            f"ЛОКАЛЬНАЯ ИСТОРИЯ · {snapshot['total_message_count']} сообщ.\n"
            f"В API: последние {snapshot['context_message_count']} · отброшено: {cutoff}\n"
        )
        for index, message in enumerate(snapshot["all_messages"]):
            marker = "OUT" if index < cutoff else "IN "
            content = message["content"].replace("\n", " ")
            lines.append(f"[{marker}] {message['role'].upper()}: {content}")
    elif isinstance(memory, FactsMemory):
        lines.append("FACTS\n" + json.dumps(memory.facts, ensure_ascii=False, indent=2))
        lines.append(
            f"\nОКНО · {memory.context_message_count}/{memory.total_message_count} сообщ."
        )
        lines.extend(_message_lines(memory.context_messages))
    else:
        assert isinstance(memory, BranchingMemory)
        active = memory.active_branch or "до checkpoint"
        lines.append(
            f"ГРАФ · общий префикс {len(memory.common_messages)} сообщ. · "
            f"активная ветка: {active}"
        )
        if memory.has_checkpoint:
            branch_sizes = snapshot["branches"]
            lines.append(
                "Ветки: "
                + " · ".join(f"{name}={len(items)} сообщ." for name, items in branch_sizes.items())
            )
        lines.append("\nКОНТЕКСТ АКТИВНОЙ ВЕТКИ")
        lines.extend(_message_lines(memory.context_messages))

    lines.append("\nПОСЛЕДНИЙ ФАКТИЧЕСКИЙ PROMPT")
    if agent.last_request_messages:
        lines.extend(_message_lines(agent.last_request_messages))
    else:
        lines.append("Запросов ещё не было.")
    return "\n".join(lines)


def stats_text(strategy: str, agent: Agent, quality: QualityResult | None) -> str:
    stats = agent.stats
    memory = agent.memory
    text = (
        f"Качество: {quality_text(quality)}  ·  Последний prompt: "
        f"{number(stats.last_prompt_tokens)}\n"
        f"Рабочие API: {stats.successful_chat_requests}/{stats.chat_attempts}  ·  "
        f"вход {number(stats.chat_input_tokens)}  ·  выход {number(stats.chat_output_tokens)}\n"
        f"Контекст: {memory.context_message_count} сообщ.  ·  "
        f"Общий расход: {number(stats.total_tokens)} токенов"
    )
    if strategy == "facts":
        text += (
            f"\nFacts API: {stats.successful_facts_requests}/{stats.facts_attempts}  ·  "
            f"{number(stats.facts_total_tokens)} токенов"
        )
    elif strategy == "branching":
        branch = getattr(memory, "active_branch", None) or "до checkpoint"
        text += f"\nАктивная ветка: {branch}"
    return text


class ContextStrategiesApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 10 — стратегии контекста"
    BINDINGS = [("ctrl+q", "quit", "Выйти"), ("ctrl+d", "start_demo", "Демо")]

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
        self.quality: dict[str, QualityResult | None] = {
            "sliding": None,
            "facts": None,
            "branch_A": None,
            "branch_B": None,
        }
        self.contamination: dict[str, tuple[tuple[str, bool], ...]] = {}
        self.branch_tokens: dict[str, int] = {}
        self.busy = False
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        models = tuple(dict.fromkeys((self.base_config.model, *SUPPORTED_MODELS)))
        with Horizontal(id="brand"):
            yield Label("CONTEXT LAB", id="title")
            yield Label("ДЕНЬ 10  /  ТРИ НЕЗАВИСИМЫЕ СЕССИИ", id="subtitle")
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
            yield Button("Автодемо · 49 API-вызовов", id="demo", variant="primary")
            yield Button("Очистить всё", id="clear-all", classes="danger")

        with TabbedContent(initial="tab-sliding", id="strategy-tabs"):
            with TabPane("Sliding Window", id="tab-sliding"):
                yield from self._strategy_widgets("sliding", "Последние N; старое удаляется из prompt")
            with TabPane("Sticky Facts", id="tab-facts"):
                yield from self._strategy_widgets("facts", "Facts JSON + последние N сообщений")
            with TabPane("Branching", id="tab-branching"):
                yield from self._strategy_widgets("branching", "Общий checkpoint + независимые ветки")
            with TabPane("Сравнение", id="tab-comparison"):
                yield Static("Демо ещё не запускалось.", id="comparison")
        yield Static("Готово", id="status")
        yield Footer()

    def _strategy_widgets(self, strategy: str, description: str) -> ComposeResult:
        with Horizontal(classes="workspace"):
            with Vertical(classes="chat-panel"):
                yield Label(f"ЧАТ · {description}", classes="panel-title")
                yield RichLog(id=f"chat-{strategy}", classes="chat", wrap=True, markup=False)
                yield Input(placeholder="Сообщение этой независимой сессии…", id=f"input-{strategy}")
                with Horizontal(classes="actions"):
                    yield Button("Отправить", id=f"send-{strategy}", variant="primary")
                    if strategy == "branching":
                        yield Button("Checkpoint + A/B", id="checkpoint")
                        yield Button("Ветка A", id="switch-a")
                        yield Button("Ветка B", id="switch-b")
                    yield Button("Очистить", id=f"clear-{strategy}", classes="danger")
            with Vertical(classes="context-panel"):
                yield Label("КОНТЕКСТ, ПЕРЕДАВАЕМЫЙ В API", classes="panel-title")
                yield RichLog(id=f"context-{strategy}", classes="context", wrap=True, markup=False)
                yield Static("", id=f"stats-{strategy}", classes="stats")

    def on_mount(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(90, connect=20),
            transport=self.transport,
            follow_redirects=False,
        )
        self._create_agents()
        for strategy in STRATEGIES:
            self.query_one(f"#chat-{strategy}", RichLog).write(
                Text("Новая независимая сессия.", style="#8294aa")
            )
        self.refresh_all()
        self._refresh_branch_buttons()
        self.query_one("#input-sliding", Input).focus()

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def _selected_model(self) -> str:
        return str(self.query_one("#model-select", Select).value)

    def _selected_recent(self) -> int:
        return int(self.query_one("#recent-select", Select).value)

    def _create_agents(self) -> None:
        if self.client is None:
            return
        config = replace(self.base_config, model=self._selected_model())
        recent = self._selected_recent()
        self.agents = {
            "sliding": Agent(config, self.client, SlidingWindowMemory(recent)),
            "facts": Agent(config, self.client, FactsMemory(recent)),
            "branching": Agent(config, self.client, BranchingMemory()),
        }

    @on(Select.Changed)
    def handle_select_changed(self) -> None:
        if self.busy or not self.agents or self._has_any_history():
            return
        self._create_agents()
        self.refresh_all()

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        if event.input.id and event.input.id.startswith("input-"):
            self.start_message(event.input.id.removeprefix("input-"))

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("send-"):
            self.start_message(button_id.removeprefix("send-"))
        elif button_id.startswith("clear-") and button_id != "clear-all":
            self.clear_strategy(button_id.removeprefix("clear-"))
        elif button_id == "clear-all":
            self.clear_all()
        elif button_id == "demo":
            self.action_start_demo()
        elif button_id == "checkpoint":
            self.create_checkpoint()
        elif button_id == "switch-a":
            self.switch_branch("A")
        elif button_id == "switch-b":
            self.switch_branch("B")

    def _has_any_history(self) -> bool:
        return any(agent.memory.total_message_count for agent in self.agents.values())

    def _lock_configuration(self) -> None:
        self.query_one("#model-select", Select).disabled = True
        self.query_one("#recent-select", Select).disabled = True

    def _unlock_configuration_if_empty(self) -> None:
        empty = not self._has_any_history()
        self.query_one("#model-select", Select).disabled = not empty
        self.query_one("#recent-select", Select).disabled = not empty

    def set_busy(self, value: bool) -> None:
        self.busy = value
        for widget in self.query(Input):
            widget.disabled = value
        for button in self.query(Button):
            button.disabled = value
        if not value:
            self._refresh_branch_buttons()

    def _refresh_branch_buttons(self) -> None:
        if not self.agents:
            return
        memory = self.agents["branching"].memory
        assert isinstance(memory, BranchingMemory)
        self.query_one("#checkpoint", Button).disabled = self.busy or memory.has_checkpoint
        self.query_one("#switch-a", Button).disabled = self.busy or not memory.has_checkpoint
        self.query_one("#switch-b", Button).disabled = self.busy or not memory.has_checkpoint

    def start_message(self, strategy: str) -> None:
        if self.busy or strategy not in STRATEGIES:
            return
        field = self.query_one(f"#input-{strategy}", Input)
        message = field.value.strip()
        if not message:
            self.query_one("#status", Static).update("Введите непустое сообщение")
            return
        field.value = ""
        self._lock_configuration()
        self.set_busy(True)
        self.query_one("#status", Static).update(f"Запрос выполняется: {strategy}")
        self.request_one(strategy, message)

    async def _ask(self, strategy: str, message: str) -> AgentReply:
        log = self.query_one(f"#chat-{strategy}", RichLog)
        branch = ""
        memory = self.agents[strategy].memory
        if isinstance(memory, BranchingMemory) and memory.active_branch:
            branch = f" [{memory.active_branch}]"
        log.write(Text(f"Вы{branch}: ", style="bold #8ed6dc") + Text(message))
        try:
            reply = await self.agents[strategy].ask(message)
        except Exception as error:
            log.write(Text("Ошибка: " + str(error), style="bold #e68a8a"))
            self.refresh_strategy(strategy)
            raise
        log.write(Text(f"Агент{branch}: ", style="bold #b8d982") + Text(reply.text))
        if reply.facts_usage is not None:
            log.write(
                Text(
                    f"FACTS обновлены · {reply.facts_usage.total_tokens} токенов",
                    style="bold #e7c47b",
                )
            )
        self.refresh_strategy(strategy)
        return reply

    @work
    async def request_one(self, strategy: str, message: str) -> None:
        try:
            await self._ask(strategy, message)
        except Exception:
            self.query_one("#status", Static).update(f"Ошибка в сессии {strategy}")
        else:
            self.query_one("#status", Static).update(f"Ответ получен: {strategy}")
        finally:
            self.set_busy(False)
            self.query_one(f"#input-{strategy}", Input).focus()

    def create_checkpoint(self) -> None:
        if self.busy:
            return
        memory = self.agents["branching"].memory
        assert isinstance(memory, BranchingMemory)
        try:
            memory.create_checkpoint()
        except ValueError as error:
            self.query_one("#status", Static).update(str(error))
            return
        self.query_one("#chat-branching", RichLog).write(
            Text("CHECKPOINT создан · активна ветка A", style="bold #e7c47b")
        )
        self.refresh_strategy("branching")
        self._refresh_branch_buttons()
        self.query_one("#status", Static).update("Созданы независимые ветки A и B")

    def switch_branch(self, name: str) -> None:
        if self.busy:
            return
        memory = self.agents["branching"].memory
        assert isinstance(memory, BranchingMemory)
        try:
            memory.switch_branch(name)
        except ValueError as error:
            self.query_one("#status", Static).update(str(error))
            return
        self.query_one("#chat-branching", RichLog).write(
            Text(f"ПЕРЕКЛЮЧЕНИЕ → ветка {name}", style="bold #e7c47b")
        )
        self.refresh_strategy("branching")
        self.query_one("#status", Static).update(f"Активна ветка {name}")

    def clear_strategy(self, strategy: str) -> None:
        if self.busy:
            return
        self.agents[strategy].clear()
        if strategy == "branching":
            self.quality["branch_A"] = None
            self.quality["branch_B"] = None
            self.contamination.clear()
            self.branch_tokens.clear()
        else:
            self.quality[strategy] = None
        log = self.query_one(f"#chat-{strategy}", RichLog)
        log.clear()
        log.write(Text("Сессия очищена.", style="#8294aa"))
        self.refresh_strategy(strategy)
        self.refresh_comparison()
        self._unlock_configuration_if_empty()
        self._refresh_branch_buttons()
        self.query_one("#status", Static).update(f"Очищена сессия {strategy}")

    def clear_all(self) -> None:
        if self.busy:
            return
        for strategy in STRATEGIES:
            self.agents[strategy].clear()
            log = self.query_one(f"#chat-{strategy}", RichLog)
            log.clear()
            log.write(Text("Сессия очищена.", style="#8294aa"))
        self.quality = {"sliding": None, "facts": None, "branch_A": None, "branch_B": None}
        self.contamination.clear()
        self.branch_tokens.clear()
        self._unlock_configuration_if_empty()
        self.refresh_all()
        self._refresh_branch_buttons()
        self.query_one("#status", Static).update("Все сессии и метрики очищены")

    def refresh_strategy(self, strategy: str) -> None:
        agent = self.agents[strategy]
        context = self.query_one(f"#context-{strategy}", RichLog)
        context.clear()
        context.write(context_text(agent))
        quality = self.quality.get(strategy)
        if strategy == "branching":
            memory = agent.memory
            assert isinstance(memory, BranchingMemory)
            if memory.active_branch:
                quality = self.quality.get(f"branch_{memory.active_branch}")
        self.query_one(f"#stats-{strategy}", Static).update(
            stats_text(strategy, agent, quality)
        )

    def refresh_all(self) -> None:
        if not self.agents:
            return
        for strategy in STRATEGIES:
            self.refresh_strategy(strategy)
        self.refresh_comparison()

    def refresh_comparison(self) -> None:
        if not self.agents:
            return
        sliding = self.agents["sliding"].stats
        facts = self.agents["facts"].stats
        branching = self.agents["branching"].stats
        a_contamination = self.contamination.get("A")
        b_contamination = self.contamination.get("B")
        common_tokens = self.branch_tokens.get("common")
        branch_a_tokens = self.branch_tokens.get("A")
        branch_b_tokens = self.branch_tokens.get("B")

        def branch_cost(branch_tokens: int | None) -> str:
            if common_tokens is None or branch_tokens is None:
                return "—"
            return number(common_tokens + branch_tokens)

        def isolation(value: tuple[tuple[str, bool], ...] | None) -> str:
            if value is None:
                return "—"
            return f"{sum(passed for _, passed in value)}/{len(value)}"

        text = (
            "ИТОГОВОЕ СРАВНЕНИЕ\n\n"
            "Стратегия       Качество       Рабочие токены   Служебные   Всего\n"
            "─────────────────────────────────────────────────────────────────\n"
            f"Sliding         {quality_text(self.quality['sliding']):<15}"
            f"{number(sliding.chat_total_tokens):<17}0             {number(sliding.total_tokens)}\n"
            f"Facts           {quality_text(self.quality['facts']):<15}"
            f"{number(facts.chat_total_tokens):<17}{number(facts.facts_total_tokens):<14}"
            f"{number(facts.total_tokens)}\n"
            f"Branch A        {quality_text(self.quality['branch_A']):<15}"
            f"{branch_cost(branch_a_tokens):<17}0             {branch_cost(branch_a_tokens)}\n"
            f"Branch B        {quality_text(self.quality['branch_B']):<15}"
            f"{branch_cost(branch_b_tokens):<17}0             {branch_cost(branch_b_tokens)}\n\n"
            f"Изоляция веток: A={isolation(a_contamination)}, B={isolation(b_contamination)}.\n"
            f"Branching целиком: {number(branching.total_tokens)} токенов; общий префикс "
            f"учтён один раз. Стоимость строки A/B включает общий префикс."
        )
        self.query_one("#comparison", Static).update(text)

    def action_start_demo(self) -> None:
        if self.busy:
            return
        if self._has_any_history():
            self.query_one("#status", Static).update(
                "Для честного демо сначала очистите все три сессии"
            )
            return
        self._lock_configuration()
        self.set_busy(True)
        self.run_demo()

    @staticmethod
    def _reply_payload(reply: AgentReply | Exception) -> dict[str, Any]:
        if isinstance(reply, Exception):
            return {"ok": False, "error": str(reply)}
        return {
            "ok": True,
            "answer": reply.text,
            "usage": asdict(reply.usage),
            "elapsed_seconds": reply.elapsed_seconds,
            "finish_reason": reply.finish_reason,
            "facts_usage": None if reply.facts_usage is None else asdict(reply.facts_usage),
            "facts": reply.facts,
        }

    async def _demo_ask(self, strategy: str, message: str) -> AgentReply:
        return await self._ask(strategy, message)

    @work
    async def run_demo(self) -> None:
        turns: list[dict[str, Any]] = []
        outcome = "failed"
        status = "Демо остановлено"
        try:
            for index, message in enumerate(LINEAR_MESSAGES, start=1):
                self.query_one("#status", Static).update(
                    f"Демо Sliding/Facts: ход {index}/{len(LINEAR_MESSAGES)}"
                )
                results = await asyncio.gather(
                    self._demo_ask("sliding", message),
                    self._demo_ask("facts", message),
                    return_exceptions=True,
                )
                turns.append(
                    {
                        "phase": "linear",
                        "turn": index,
                        "user_message": message,
                        "results": {
                            name: self._reply_payload(result)
                            for name, result in zip(("sliding", "facts"), results, strict=True)
                        },
                    }
                )
                if any(isinstance(result, Exception) for result in results):
                    status = "Демо остановлено: ошибка линейной стратегии"
                    return

            results = await asyncio.gather(
                self._demo_ask("sliding", CHECK_MESSAGE),
                self._demo_ask("facts", CHECK_MESSAGE),
                return_exceptions=True,
            )
            turns.append(
                {
                    "phase": "linear_check",
                    "user_message": CHECK_MESSAGE,
                    "results": {
                        name: self._reply_payload(result)
                        for name, result in zip(("sliding", "facts"), results, strict=True)
                    },
                }
            )
            if any(isinstance(result, Exception) for result in results):
                status = "Демо остановлено: ошибка контрольного вопроса"
                return
            self.quality["sliding"] = evaluate_answer(results[0].text, EXPECTED_A)
            self.quality["facts"] = evaluate_answer(results[1].text, EXPECTED_A)

            for index, message in enumerate(COMMON_MESSAGES, start=1):
                self.query_one("#status", Static).update(
                    f"Демо Branching: общий ход {index}/{len(COMMON_MESSAGES)}"
                )
                reply = await self._demo_ask("branching", message)
                turns.append(
                    {
                        "phase": "branch_common",
                        "turn": index,
                        "user_message": message,
                        "result": self._reply_payload(reply),
                    }
                )

            memory = self.agents["branching"].memory
            assert isinstance(memory, BranchingMemory)
            self.branch_tokens["common"] = self.agents["branching"].stats.total_tokens
            memory.create_checkpoint()
            self.query_one("#chat-branching", RichLog).write(
                Text("CHECKPOINT · созданы A и B", style="bold #e7c47b")
            )

            for name, messages, expected in (
                ("A", BRANCH_A_MESSAGES, EXPECTED_A),
                ("B", BRANCH_B_MESSAGES, EXPECTED_B),
            ):
                memory.switch_branch(name)
                branch_start_tokens = self.agents["branching"].stats.total_tokens
                self.query_one("#chat-branching", RichLog).write(
                    Text(f"ПЕРЕКЛЮЧЕНИЕ → ветка {name}", style="bold #e7c47b")
                )
                for index, message in enumerate(messages, start=1):
                    self.query_one("#status", Static).update(
                        f"Демо Branching: ветка {name}, ход {index}/{len(messages)}"
                    )
                    reply = await self._demo_ask("branching", message)
                    turns.append(
                        {
                            "phase": f"branch_{name}",
                            "turn": index,
                            "user_message": message,
                            "result": self._reply_payload(reply),
                        }
                    )
                check_reply = await self._demo_ask("branching", CHECK_MESSAGE)
                quality = evaluate_answer(check_reply.text, expected)
                isolation = contamination_checks(check_reply.text, expected_branch=name)
                self.quality[f"branch_{name}"] = quality
                self.contamination[name] = isolation
                self.branch_tokens[name] = (
                    self.agents["branching"].stats.total_tokens - branch_start_tokens
                )
                turns.append(
                    {
                        "phase": f"branch_{name}_check",
                        "user_message": CHECK_MESSAGE,
                        "result": self._reply_payload(check_reply),
                        "quality": asdict(quality),
                        "isolation": isolation,
                    }
                )

            outcome = "completed"
            status = "Демо завершено; результаты показаны во вкладке «Сравнение»"
            self.refresh_all()
        except Exception as error:
            status = f"Демо остановлено: {error}"
        finally:
            try:
                report_path = save_demo_report(
                    {
                        "kind": "day-10-context-strategies",
                        "outcome": outcome,
                        "configuration": {
                            "model": self.agents["sliding"].config.model,
                            "temperature": self.agents["sliding"].config.temperature,
                            "max_tokens": self.agents["sliding"].config.max_tokens,
                            "facts_max_tokens": self.agents["facts"].config.facts_max_tokens,
                            "keep_recent_messages": self._selected_recent(),
                        },
                        "actual_api_attempts": sum(
                            agent.stats.chat_attempts + agent.stats.facts_attempts
                            for agent in self.agents.values()
                        ),
                        "turns": turns,
                        "quality": {
                            key: None if value is None else asdict(value)
                            for key, value in self.quality.items()
                        },
                        "branch_isolation": self.contamination,
                        "branch_tokens": self.branch_tokens,
                        "agents": {
                            strategy: agent.snapshot()
                            for strategy, agent in self.agents.items()
                        },
                    },
                    self.report_dir,
                )
            except ReportError as error:
                status += f"; {error}"
            else:
                status += f"; отчёт: {report_path}"
            self.refresh_all()
            self.set_busy(False)
            self.query_one("#status", Static).update(status)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 10: Sliding Window, Sticky Facts и Branching в TUI."
    )
    parser.parse_args()
    try:
        config = load_config()
        config.check()
    except ValueError as error:
        parser.error(str(error))
    ContextStrategiesApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
