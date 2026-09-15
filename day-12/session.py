"""Сессии диалога с привязкой одного активного профиля."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from user_profile import SAFE_ID, ProfileError, _atomic_save


Message = dict[str, str]


def _message(value: Any, roles: set[str]) -> Message:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise ValueError("неверная структура сообщения")
    role, content = value["role"], value["content"]
    if role not in roles or not isinstance(content, str) or not content.strip():
        raise ValueError("неверная роль или пустое сообщение")
    return {"role": role, "content": content.strip()}


@dataclass
class Session:
    session_id: str
    profile_id: str
    messages: list[Message] = field(default_factory=list)
    last_request_messages: list[Message] = field(default_factory=list)

    def checked(self) -> Session:
        if SAFE_ID.fullmatch(self.session_id) is None or SAFE_ID.fullmatch(self.profile_id) is None:
            raise ValueError("ID сессии или профиля содержит недопустимые символы")
        messages = [_message(item, {"user", "assistant"}) for item in self.messages]
        if len(messages) % 2:
            raise ValueError("история должна состоять из завершённых пар")
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise ValueError("в истории нарушен порядок ролей")
        prompt = [
            _message(item, {"system", "user", "assistant"})
            for item in self.last_request_messages
        ]
        return Session(self.session_id, self.profile_id, messages, prompt)


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "sessions"

    def path(self, session_id: str) -> Path:
        if SAFE_ID.fullmatch(session_id) is None:
            raise ValueError("ID сессии содержит недопустимые символы")
        return self.root / f"{session_id}.json"

    def save(self, session: Session) -> None:
        try:
            checked = session.checked()
        except (TypeError, ValueError) as error:
            raise ProfileError("Нельзя сохранить неверную сессию") from error
        _atomic_save(
            self.path(checked.session_id),
            {
                "session_id": checked.session_id,
                "profile_id": checked.profile_id,
                "messages": checked.messages,
                "last_request_messages": checked.last_request_messages,
            },
        )

    def load(self, session_id: str) -> Session:
        path = self.path(session_id)
        try:
            with path.open(encoding="utf-8") as source:
                raw = json.load(source)
        except json.JSONDecodeError as error:
            raise ProfileError(f"Повреждён JSON сессии: {path.name}") from error
        except OSError as error:
            raise ProfileError(f"Не удалось прочитать {path.name}") from error
        if not isinstance(raw, dict) or set(raw) != {
            "session_id",
            "profile_id",
            "messages",
            "last_request_messages",
        }:
            raise ProfileError(f"Сессия {path.name} имеет неверную структуру")
        try:
            return Session(**raw).checked()
        except (TypeError, ValueError) as error:
            raise ProfileError(f"Сессия {path.name} имеет неверную структуру") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in self.root.glob("session-*.json")))

    def create(self, session_id: str, profile_id: str) -> Session:
        path = self.path(session_id)
        if path.exists():
            raise ProfileError(f"Сессия {session_id} уже существует")
        session = Session(session_id, profile_id)
        self.save(session)
        return session

    def sessions_using(self, profile_id: str) -> tuple[str, ...]:
        return tuple(
            session_id for session_id in self.ids() if self.load(session_id).profile_id == profile_id
        )
