#!/usr/bin/env python3
"""TUI для создания профилей и проверки их автоматического применения."""

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
from user_profile import DEFAULT_PROFILES, Profile, ProfileError, ProfileStore
from session import Session, SessionStore


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
CONTROL_QUESTION = "Составь план изучения асинхронного Python на неделю."
FOLLOW_UP_QUESTION = "С чего мне начать сегодня?"
FIELD_IDS = {
    "profile-name": "name",
    "profile-address": "address",
    "profile-language": "language",
    "profile-style": "style",
    "profile-format": "response_format",
    "profile-constraints": "constraints",
}


def _lines(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{index:02d} {message['role'].upper()}\n{message['content']}"
        for index, message in enumerate(messages, start=1)
    )


class ProfilesApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "День 12 — User Profiles"
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
        self.profile_store = ProfileStore(data_dir)
        self.session_store = SessionStore(data_dir)
        self.profiles: tuple[Profile, ...] = ()
        self.session: Session | None = None
        self.agent: Agent | None = None
        self.editing_id: str | None = None
        self.total_tokens = 0
        self.busy = False
        self.clear_armed = False
        self.delete_armed = False
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        models = tuple(dict.fromkeys((self.base_config.model, *SUPPORTED_MODELS)))
        with Horizontal(id="brand"):
            yield Label("PROFILE LAB", id="title")
            yield Label("ДЕНЬ 12  /  ПЕРСОНАЛИЗАЦИЯ АССИСТЕНТА", id="subtitle")
        with Horizontal(id="configuration"):
            yield Label("Модель", classes="config-label")
            yield Select([(item, item) for item in models], value=self.base_config.model, id="model", allow_blank=False)
            yield Label("Профиль", classes="config-label spaced")
            yield Select(
                [(item.name, item.profile_id) for item in DEFAULT_PROFILES],
                value=DEFAULT_PROFILES[0].profile_id,
                id="profile",
                allow_blank=False,
            )
            yield Label("Сессия", classes="config-label spaced")
            yield Select([("session-01", "session-01")], value="session-01", id="session", allow_blank=False)
            yield Button("Новая сессия", id="new-session")
            yield Button("Автодемо · 3 API", id="demo", variant="primary")
        yield Static("", id="scope")
        with Horizontal(id="workspace"):
            with Vertical(id="chat-panel"):
                yield Label("ДИАЛОГ", classes="panel-title")
                yield RichLog(id="chat", classes="content", wrap=True, markup=False)
                yield Input(placeholder="Введите вопрос…", id="message")
                with Horizontal(id="chat-actions"):
                    yield Button("Спросить", id="ask", variant="primary")
                    yield Button("Очистить сессию", id="clear-session", classes="danger")
            with Vertical(id="profile-panel"):
                with TabbedContent(initial="profile-tab", id="right-tabs"):
                    with TabPane("Профиль", id="profile-tab"):
                        yield Label("Название")
                        yield Input(id="profile-name")
                        yield Label("Обращение")
                        yield Input(id="profile-address")
                        yield Label("Язык")
                        yield Input(id="profile-language")
                        yield Label("Стиль")
                        yield Input(id="profile-style")
                        yield Label("Формат ответа")
                        yield Input(id="profile-format")
                        yield Label("Ограничения")
                        yield Input(id="profile-constraints")
                        with Horizontal(id="profile-actions"):
                            yield Button("Сохранить", id="save-profile", variant="primary")
                            yield Button("Новый", id="new-profile")
                            yield Button("Удалить", id="delete-profile", classes="danger")
                        yield Static("", id="profile-notice")
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
            self.profiles = self.profile_store.ensure_defaults()
            session_ids = self.session_store.ids()
            self.session = (
                self.session_store.load(session_ids[-1])
                if session_ids
                else self.session_store.create("session-01", self.profiles[0].profile_id)
            )
            if self._profile(self.session.profile_id) is None:
                self.session.profile_id = self.profiles[0].profile_id
                self.session_store.save(self.session)
        except ProfileError as error:
            self._status(str(error))
            return
        self._update_profile_options()
        self._update_session_options()
        self._create_agent()
        self._load_profile_form(self.active_profile)
        self._render_chat("Загружена сохранённая сессия." if self.session.messages else "Новая сессия. История пуста.")
        self.refresh_views()
        self.query_one("#message", Input).focus()

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    @property
    def active_profile(self) -> Profile:
        assert self.session is not None
        profile = self._profile(self.session.profile_id)
        if profile is None:
            raise ProfileError(f"Профиль {self.session.profile_id} не найден")
        return profile

    def _profile(self, profile_id: str) -> Profile | None:
        return next((item for item in self.profiles if item.profile_id == profile_id), None)

    def _create_agent(self) -> None:
        if self.client is None or self.session is None:
            return
        config = replace(self.base_config, model=str(self.query_one("#model", Select).value))
        self.agent = Agent(config, self.client, self.session, self.session_store, self.active_profile)

    def _update_profile_options(self) -> None:
        if self.session is None:
            return
        selector = self.query_one("#profile", Select)
        with self.prevent(Select.Changed):
            selector.set_options([(item.name, item.profile_id) for item in self.profiles])
            selector.value = self.session.profile_id

    def _update_session_options(self) -> None:
        if self.session is None:
            return
        selector = self.query_one("#session", Select)
        with self.prevent(Select.Changed):
            selector.set_options([(item, item) for item in self.session_store.ids()])
            selector.value = self.session.session_id

    @on(Select.Changed, "#model")
    def model_changed(self) -> None:
        if not self.busy and self.session is not None:
            self._create_agent()
            self.refresh_views()

    @on(Select.Changed, "#profile")
    def profile_changed(self, event: Select.Changed) -> None:
        if self.busy or self.session is None or event.value == Select.NULL:
            return
        profile_id = str(event.value)
        if self._profile(profile_id) is None:
            return
        try:
            self.session.profile_id = profile_id
            self.session_store.save(self.session)
        except ProfileError as error:
            self._status(str(error))
            return
        self.editing_id = profile_id
        self.delete_armed = False
        self._load_profile_form(self.active_profile)
        self._create_agent()
        self._status(f"Профиль «{self.active_profile.name}» применяется к каждому запросу")
        self.refresh_views()

    @on(Select.Changed, "#session")
    def session_changed(self, event: Select.Changed) -> None:
        if self.busy or event.value == Select.NULL or self.session is None:
            return
        session_id = str(event.value)
        if session_id == self.session.session_id:
            return
        try:
            candidate = self.session_store.load(session_id)
            if self._profile(candidate.profile_id) is None:
                raise ProfileError(f"Профиль сессии {candidate.profile_id} не найден")
        except ProfileError as error:
            self._status(str(error))
            self._update_session_options()
            return
        self.session = candidate
        self._update_profile_options()
        self._create_agent()
        self._load_profile_form(self.active_profile)
        self._render_chat("Загружена сохранённая сессия.")
        self._status(f"Активна {session_id}")
        self.refresh_views()

    @on(Input.Submitted, "#message")
    def input_submitted(self) -> None:
        self.start_ask()

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "ask": self.start_ask,
            "clear-session": self.clear_session,
            "new-session": self.new_session,
            "demo": self.action_start_demo,
            "save-profile": self.save_profile,
            "new-profile": self.new_profile,
            "delete-profile": self.delete_profile,
        }
        action = actions.get(event.button.id or "")
        if action is not None:
            action()

    def start_ask(self) -> None:
        if self.busy or self.agent is None:
            return
        field = self.query_one("#message", Input)
        message = field.value.strip()
        if not message:
            self._status("Введите вопрос")
            return
        field.value = ""
        self.clear_armed = False
        self.delete_armed = False
        self.set_busy(True)
        self._status(f"Запрос с профилем «{self.active_profile.name}»")
        self.request_one(message)

    async def _ask(self, message: str) -> None:
        assert self.agent is not None and self.session is not None
        log = self.query_one("#chat", RichLog)
        log.write(Text("Вы: ", style="bold #8ed6dc") + Text(message))
        try:
            reply = await self.agent.ask(message)
        except (AgentError, ProfileError) as error:
            log.write(Text("Ошибка: " + str(error), style="bold #e68a8a"))
            self.refresh_views()
            raise
        log.write(Text(f"{self.active_profile.name}: ", style="bold #b8d982") + Text(reply.text))
        self.total_tokens += reply.usage.total_tokens
        self.refresh_views()

    @work
    async def request_one(self, message: str) -> None:
        try:
            await self._ask(message)
        except (AgentError, ProfileError):
            self._status("Ошибка: диалог не сохранён")
        else:
            self._status("Ответ получен; профиль был добавлен в system автоматически")
        finally:
            self.set_busy(False)
            self.query_one("#message", Input).focus()

    def _next_session_id(self) -> str:
        numbers = [
            int(match.group(1))
            for name in self.session_store.ids()
            if (match := re.fullmatch(r"session-(\d+)", name))
        ]
        return f"session-{max(numbers, default=0) + 1:02d}"

    def _create_session(self, profile_id: str, *, notice: str) -> None:
        try:
            self.session = self.session_store.create(self._next_session_id(), profile_id)
        except ProfileError as error:
            self._status(str(error))
            return
        self._update_session_options()
        self._update_profile_options()
        self._create_agent()
        self._load_profile_form(self.active_profile)
        self._render_chat(notice)
        self.refresh_views()

    def new_session(self) -> None:
        if self.busy or self.session is None:
            return
        self._create_session(
            self.session.profile_id,
            notice="Новая сессия: история пуста, активный профиль сохранён.",
        )
        assert self.session is not None
        self._status(f"Создана {self.session.session_id}")

    def clear_session(self) -> None:
        if self.busy or self.session is None:
            return
        if not self.clear_armed:
            self.clear_armed = True
            self._status("Нажмите ещё раз: история активной сессии будет очищена")
            return
        self.session.messages = []
        self.session.last_request_messages = []
        try:
            self.session_store.save(self.session)
        except ProfileError as error:
            self._status(str(error))
            return
        self.clear_armed = False
        self._render_chat("История активной сессии очищена. Профиль сохранён.")
        self._status("Активная сессия очищена")
        self.refresh_views()

    def _load_profile_form(self, profile: Profile) -> None:
        self.editing_id = profile.profile_id
        values = {
            "profile-name": profile.name,
            "profile-address": profile.address,
            "profile-language": profile.language,
            "profile-style": profile.style,
            "profile-format": profile.response_format,
            "profile-constraints": profile.constraints,
        }
        for widget_id, value in values.items():
            self.query_one(f"#{widget_id}", Input).value = value

    def new_profile(self) -> None:
        if self.busy:
            return
        numbers = [
            int(match.group(1))
            for item in self.profiles
            if (match := re.fullmatch(r"profile-(\d+)", item.profile_id))
        ]
        self.editing_id = f"profile-{max(numbers, default=0) + 1:02d}"
        defaults = {
            "profile-name": "Новый профиль",
            "profile-address": "Пользователь",
            "profile-language": "Русский",
            "profile-style": "Нейтральный, ясный и доброжелательный",
            "profile-format": "Свободный текст с короткими абзацами",
            "profile-constraints": "Без дополнительных ограничений",
        }
        for widget_id, value in defaults.items():
            self.query_one(f"#{widget_id}", Input).value = value
        self.delete_armed = False
        self._profile_notice("Новый профиль подготовлен. Измените нужные поля и сохраните.")
        self._status("Новый профиль ещё не сохранён")
        self.query_one("#profile-name", Input).focus()

    def save_profile(self) -> None:
        if self.busy or self.editing_id is None or self.session is None:
            return
        values = {
            attribute: self.query_one(f"#{widget_id}", Input).value
            for widget_id, attribute in FIELD_IDS.items()
        }
        try:
            profile = Profile(profile_id=self.editing_id, **values).normalized()
            self.profiles = self.profile_store.upsert(profile)
            self.session.profile_id = profile.profile_id
            self.session_store.save(self.session)
        except (ProfileError, ValueError) as error:
            self._profile_notice("Ошибка: " + str(error))
            self._status(str(error))
            return
        self._update_profile_options()
        self._create_agent()
        self.delete_armed = False
        self._profile_notice(f"Сохранено: «{profile.name}». Профиль появился в списке и активирован.")
        self._status(f"Профиль «{profile.name}» сохранён и активирован")
        self.refresh_views()

    def delete_profile(self) -> None:
        if self.busy or self.editing_id is None or self.session is None:
            return
        profile = self._profile(self.editing_id)
        if profile is None:
            self._status("Сначала сохраните новый профиль")
            return
        if len(self.profiles) == 1:
            self._status("Нельзя удалить единственный профиль")
            return
        other_sessions = tuple(
            item for item in self.session_store.sessions_using(profile.profile_id)
            if item != self.session.session_id
        )
        if other_sessions:
            self._status("Профиль используется другими сессиями: " + ", ".join(other_sessions))
            return
        if not self.delete_armed:
            self.delete_armed = True
            self._status(f"Нажмите ещё раз, чтобы удалить «{profile.name}»")
            return
        fallback = next(item for item in self.profiles if item.profile_id != profile.profile_id)
        try:
            self.session.profile_id = fallback.profile_id
            self.session_store.save(self.session)
            self.profiles = self.profile_store.delete(profile.profile_id)
        except ProfileError as error:
            self._status(str(error))
            return
        self.delete_armed = False
        self._update_profile_options()
        self._load_profile_form(fallback)
        self._create_agent()
        self._status(f"Профиль «{profile.name}» удалён; активирован «{fallback.name}»")
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

    def _profile_notice(self, text: str) -> None:
        self.query_one("#profile-notice", Static).update(text)

    def _render_chat(self, notice: str) -> None:
        assert self.session is not None
        log = self.query_one("#chat", RichLog)
        log.clear()
        log.write(Text(notice, style="#8294aa"))
        for message in self.session.messages:
            label = "Вы: " if message["role"] == "user" else "Агент: "
            style = "bold #8ed6dc" if message["role"] == "user" else "bold #b8d982"
            log.write(Text(label, style=style) + Text(message["content"]))

    def refresh_views(self) -> None:
        if self.session is None:
            return
        self.query_one("#scope", Static).update(
            f"USER demo   ·   SESSION {self.session.session_id}   ·   PROFILE {self.active_profile.name}"
        )
        prompt = self.query_one("#prompt", RichLog)
        prompt.clear()
        if self.session.last_request_messages:
            prompt.write(_lines(self.session.last_request_messages))
        else:
            preview = self.agent.build_messages("<следующий вопрос>") if self.agent else []
            prompt.write("Следующий запрос будет собран так:\n\n" + _lines(preview))
        self.query_one("#stats", Static).update(
            f"ПРОФИЛЕЙ {len(self.profiles)}  ·  СЕССИЯ {len(self.session.messages)} сообщ.  ·  API {self.total_tokens} токенов"
        )

    def action_start_demo(self) -> None:
        if self.busy:
            return
        if self._profile("tech-lead") is None or self._profile("mentor") is None:
            self._status("Для автодемо нужны профили tech-lead и mentor")
            return
        self.set_busy(True)
        self.run_demo()

    @work
    async def run_demo(self) -> None:
        try:
            self._create_session("tech-lead", notice="Демо 1/3: тот же вопрос с кратким профилем.")
            self._status("Демо 1/3 · Краткий техлид")
            await self._ask(CONTROL_QUESTION)
            self._create_session("mentor", notice="Демо 2/3: тот же вопрос с обучающим профилем.")
            self._status("Демо 2/3 · Обучающий наставник")
            await self._ask(CONTROL_QUESTION)
            self._status("Демо 3/3 · профиль не упоминается в вопросе")
            await self._ask(FOLLOW_UP_QUESTION)
            self._status("Демо завершено: профиль автоматически применён в трёх запросах")
        except (AgentError, ProfileError) as error:
            self._status("Демо остановлено: " + str(error))
        finally:
            self.set_busy(False)
            self.refresh_views()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="День 12: профили пользователя в system каждого запроса.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Каталог профилей и сессий")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config()
        config.check()
    except ValueError as error:
        print(f"Ошибка конфигурации: {error}")
        return 2
    ProfilesApp(config, data_dir=args.data_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
