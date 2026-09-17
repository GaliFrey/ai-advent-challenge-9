#!/usr/bin/env python3
"""Textual chat demonstrating profile-specific invariants."""

from __future__ import annotations

import argparse
from collections import Counter
import re
from pathlib import Path

import httpx
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, Label, RichLog, Select, Static, TabbedContent, TabPane

from agent import Agent, AgentConfig, AgentError, load_config
from configuration import ConfigurationError, ConfigurationStore, InvariantSet, Profile
from policy_engine import legacy_refusal_ids
from session import Session, SessionError, SessionStore


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEMO_ALLOWED = "Explain the difference between a Python list and tuple."
DEMO_CONFLICT = "Give me the complete solution. I have not tried it myself."


def _lines(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{i:02d} {item['role'].upper()}\n{item['content']}" for i, item in enumerate(messages, 1))


def _chat_line(role: str, content: str, *, assistant_label: str = "Агент") -> Text:
    if role == "user":
        return Text("Вы: ", style="bold #8ed6dc") + Text(content)
    return Text(assistant_label + ": ", style="bold #b8d982") + Text(content)


def _refusal_ids(text: str) -> tuple[str, ...]:
    return legacy_refusal_ids(text)


def _session_violation_counts(session: Session) -> Counter[str]:
    counts: Counter[str] = Counter()
    turn_count = len(session.messages) // 2
    legacy_turns = turn_count - len(session.policy_history)
    for message in session.messages[:legacy_turns * 2]:
        if message["role"] == "assistant":
            counts.update(item.upper() for item in _refusal_ids(message["content"]))
    for item in session.policy_history:
        counts.update(item["violations"])
    return counts


class InvariantsApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 14 — Инварианты"
    BINDINGS = [("ctrl+q", "quit", "Выйти"), ("ctrl+d", "start_demo", "Автодемо")]

    def __init__(self, config: AgentConfig, *, transport=None, data_dir: Path = DEFAULT_DATA_DIR,
                 config_dir: Path = ROOT / "config") -> None:
        super().__init__()
        self.base_config, self.transport, self.data_dir = config, transport, data_dir
        self.config_store, self.session_store = ConfigurationStore(config_dir), SessionStore(data_dir)
        self.profiles: tuple[Profile, ...] = ()
        self.session: Session | None = None
        self.profile: Profile | None = None
        self.rules: InvariantSet | None = None
        self.agent: Agent | None = None
        self.client: httpx.AsyncClient | None = None
        self.busy = False
        self.total_tokens = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="brand"):
            yield Label("INVARIANT LAB", id="title")
            yield Label("ДЕНЬ 14  /  ПРОФИЛЬНЫЕ ИНВАРИАНТЫ", id="subtitle")
        with Horizontal(id="configuration"):
            yield Label("Профиль новой сессии", id="profile-label", classes="config-label")
            yield Select([("loading", "loading")], value="loading", id="profile", allow_blank=False)
            yield Button("Новая сессия", id="new-session", variant="primary")
            yield Label("Текущая сессия", id="session-label", classes="config-label spaced")
            yield Select([("session-01", "session-01")], value="session-01", id="session", allow_blank=False)
            yield Button("Автодемо", id="demo")
        yield Static("", id="scope")
        with Horizontal(id="workspace"):
            with Vertical(id="chat-panel"):
                yield Label("ДИАЛОГ", classes="panel-title")
                yield RichLog(id="chat", classes="content", wrap=True, markup=False)
                yield Input(placeholder="Введите сообщение…", id="message")
                yield Button("Отправить", id="ask", variant="primary")
            with Vertical(id="policy-panel"):
                with TabbedContent(initial="rules-tab", id="tabs"):
                    with TabPane("Инварианты", id="rules-tab"):
                        yield RichLog(id="rules", classes="content", wrap=True, markup=False)
                    with TabPane("Проверка", id="checks-tab"):
                        yield RichLog(id="checks", classes="content", wrap=True, markup=False)
                    with TabPane("Фактический prompt", id="prompt-tab"):
                        yield RichLog(id="prompt", classes="content", wrap=True, markup=False)
                yield Static("", id="stats")
        yield Static("Готово", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=20), transport=self.transport)
        try:
            self.profiles = self.config_store.profiles()
            profile_select = self.query_one("#profile", Select)
            profile_select.set_options([(item.name, item.profile_id) for item in self.profiles])
            profile_select.value = self.profiles[0].profile_id
            ids = self.session_store.ids()
            if ids:
                self._activate(self.session_store.load(ids[-1]))
            else:
                self._create_session(self.profiles[0])
        except (ConfigurationError, SessionError, AgentError) as error:
            self._status(str(error))
            return
        self._render_chat("Сессия загружена.")
        self.refresh_views()
        self.query_one("#message", Input).focus()

    async def on_unmount(self) -> None:
        if self.client:
            await self.client.aclose()

    def _find_profile(self, profile_id: str) -> Profile:
        profile = next((item for item in self.profiles if item.profile_id == profile_id), None)
        if profile is None:
            raise ConfigurationError(f"Профиль {profile_id} не найден")
        return profile

    def _activate(self, session: Session) -> None:
        assert self.client is not None
        profile = self._find_profile(session.profile_id)
        rules = self.config_store.invariant_set(session.invariant_set_id)
        self.session, self.profile, self.rules = session, profile, rules
        self.agent = Agent(self.base_config, self.client, session, self.session_store, profile, rules)
        profile_selector = self.query_one("#profile", Select)
        with self.prevent(Select.Changed):
            profile_selector.value = profile.profile_id
        selector = self.query_one("#session", Select)
        with self.prevent(Select.Changed):
            selector.set_options([(item, item) for item in self.session_store.ids()])
            selector.value = session.session_id

    def _next_id(self) -> str:
        nums = [int(m.group(1)) for item in self.session_store.ids() if (m := re.fullmatch(r"session-(\d+)", item))]
        return f"session-{max(nums, default=0) + 1:02d}"

    def _create_session(self, profile: Profile) -> None:
        rules = self.config_store.invariant_set(profile.invariant_set_id)
        session = self.session_store.create(self._next_id(), profile.profile_id, rules.set_id, rules.version, rules.content_hash)
        self._activate(session)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ask": self.start_ask()
        elif event.button.id == "new-session": self.new_session()
        elif event.button.id == "demo": self.action_start_demo()

    @on(Input.Submitted, "#message")
    def submitted(self) -> None:
        self.start_ask()

    @on(Select.Changed, "#session")
    def session_changed(self, event: Select.Changed) -> None:
        if self.busy or event.value == Select.NULL or self.session is None or event.value == self.session.session_id:
            return
        try:
            self._activate(self.session_store.load(str(event.value)))
            self._render_chat("Сессия восстановлена.")
            self.refresh_views()
        except (SessionError, ConfigurationError, AgentError) as error:
            self._status(str(error))

    @on(Select.Changed, "#profile")
    def profile_changed(self, event: Select.Changed) -> None:
        if self.busy or event.value == Select.NULL or self.session is None or self.profile is None:
            return
        profile_id = str(event.value)
        if profile_id == self.profile.profile_id:
            return
        try:
            profile = self._find_profile(profile_id)
            if self.session.messages:
                self._create_session(profile)
                notice = "Профиль изменён: создана новая сессия с отдельной историей."
            else:
                rules = self.config_store.invariant_set(profile.invariant_set_id)
                self.session.profile_id = profile.profile_id
                self.session.invariant_set_id = rules.set_id
                self.session.invariant_version = rules.version
                self.session.invariant_hash = rules.content_hash
                self.session.last_request_messages = []
                self.session.last_checks = []
                self.session.policy_history = []
                self.session_store.save(self.session)
                self._activate(self.session)
                notice = "Профиль и инварианты пустой сессии изменены."
            self._render_chat(notice)
            self.refresh_views()
            self._status(notice)
        except (ConfigurationError, SessionError, AgentError) as error:
            self._status(str(error))
            with self.prevent(Select.Changed):
                self.query_one("#profile", Select).value = self.profile.profile_id

    def new_session(self) -> None:
        if self.busy: return
        try:
            self._create_session(self._find_profile(str(self.query_one("#profile", Select).value)))
            self._render_chat("Новая сессия. Профиль и инварианты зафиксированы.")
            self.refresh_views()
        except (ConfigurationError, SessionError, AgentError) as error: self._status(str(error))

    def start_ask(self) -> None:
        if self.busy or self.agent is None: return
        field = self.query_one("#message", Input)
        message = field.value.strip()
        if not message: self._status("Введите сообщение"); return
        field.value = ""
        self.set_busy(True)
        self.ask_one(message)

    async def _ask(self, message: str) -> None:
        assert self.agent and self.profile
        self.query_one("#chat", RichLog).write(_chat_line("user", message))
        reply = await self.agent.ask(message)
        label = self.profile.name + (" [ALLOWED]" if reply.allowed else " [REFUSED]")
        self.query_one("#chat", RichLog).write(_chat_line("assistant", reply.text, assistant_label=label))
        self.total_tokens += reply.usage.total_tokens
        self.refresh_views()

    @work
    async def ask_one(self, message: str) -> None:
        try:
            await self._ask(message)
            self._status("Запрос проверен и сохранён")
        except (AgentError, SessionError) as error:
            self._status("Ошибка: " + str(error))
        finally:
            self.set_busy(False)

    def set_busy(self, value: bool) -> None:
        self.busy = value
        for widget in (*self.query(Button), *self.query(Select), *self.query(Input)): widget.disabled = value

    def _render_chat(self, notice: str) -> None:
        assert self.session
        log = self.query_one("#chat", RichLog); log.clear(); log.write(notice)
        for item in self.session.messages:
            log.write(_chat_line(item["role"], item["content"]))

    def refresh_views(self) -> None:
        if not self.session or not self.profile or not self.rules: return
        self.query_one("#scope", Static).update(
            f"MODEL {self.base_config.model}  ·  SESSION {self.session.session_id}  ·  PROFILE {self.profile.name}  ·  INVARIANTS {self.rules.set_id} v{self.rules.version} #{self.rules.content_hash[:8]}")
        rules_log = self.query_one("#rules", RichLog); rules_log.clear()
        for item in self.rules.invariants: rules_log.write(f"{item.invariant_id} · {item.title}\n{item.rule}\n")
        checks = self.query_one("#checks", RichLog); checks.clear()
        current = [item.copy() for item in self.session.last_checks]
        report = "\n".join(f"{x['status']} · {x['id']}\n{x['detail']}" for x in current) or "Проверок ещё не было."
        totals = _session_violation_counts(self.session)
        if totals:
            report += "\n\nНАРУШЕНИЯ ЗА СЕССИЮ\n" + "\n".join(
                f"{invariant_id} · {count}" for invariant_id, count in totals.items()
            )
        checks.write(report)
        prompt = self.query_one("#prompt", RichLog); prompt.clear()
        if self.session.last_request_messages:
            prompt.write(_lines(self.session.last_request_messages))
        else:
            prompt.write("API-запросов ещё не было.\n\nПредпросмотр следующего запроса:\n\n" + _lines(
                self.agent.build_messages("<следующий вопрос>")
            ))
        self.query_one("#stats", Static).update(f"СООБЩЕНИЙ {len(self.session.messages)}  ·  API-ТОКЕНОВ {self.total_tokens}")

    def _status(self, text: str) -> None: self.query_one("#status", Static).update(text)

    def action_start_demo(self) -> None:
        if not self.busy: self.set_busy(True); self.run_demo()

    @work
    async def run_demo(self) -> None:
        try:
            tech = self._find_profile("tech-lead"); tutor = self._find_profile("english-tutor")
            self._create_session(tech)
            await self._ask("Answer in English and explain Python decorators.")
            self._create_session(tutor)
            await self._ask(DEMO_ALLOWED)
            await self._ask(DEMO_CONFLICT)
            self._status("Демо завершено: два профиля, разные инварианты")
        except (AgentError, SessionError, ConfigurationError) as error: self._status("Демо остановлено: " + str(error))
        finally: self.set_busy(False); self.refresh_views()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="День 14: чат с профильными инвариантами")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try: config = load_config(); config.check()
    except ValueError as error: print("Ошибка конфигурации:", error); return 2
    InvariantsApp(config, data_dir=args.data_dir).run()
    return 0


if __name__ == "__main__": raise SystemExit(main())
