"""Три явно разделённых файловых слоя памяти агента."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable


SAFE_ID = re.compile(r"[a-zA-Z0-9_-]+")
Message = dict[str, str]


class MemoryError(RuntimeError):
    """Ошибка чтения, проверки или атомарной записи памяти."""


def _validate_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} содержит недопустимые символы")
    return value


def _atomic_save(path: Path, payload: Any) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary = Path(destination.name)
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise MemoryError(f"Не удалось сохранить память: {path.name}") from error


class JsonListStore:
    """Атомарно хранит один JSON-массив с проверяемой схемой элементов."""

    def __init__(self, path: Path, validate_item: Callable[[Any], Any]) -> None:
        self.path = path
        self._validate_item = validate_item

    def load(self) -> list[Any]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8") as source:
                raw = json.load(source)
        except json.JSONDecodeError as error:
            raise MemoryError(f"Повреждён JSON памяти: {self.path.name}") from error
        except OSError as error:
            raise MemoryError(f"Не удалось прочитать память: {self.path.name}") from error
        if not isinstance(raw, list):
            raise MemoryError(f"Память {self.path.name} должна быть JSON-массивом")
        try:
            return [self._validate_item(item) for item in raw]
        except (TypeError, ValueError) as error:
            raise MemoryError(f"Память {self.path.name} имеет неверную структуру") from error

    def save(self, items: list[Any]) -> None:
        try:
            checked = [self._validate_item(item) for item in items]
        except (TypeError, ValueError) as error:
            raise MemoryError(f"Нельзя сохранить неверную память: {self.path.name}") from error
        _atomic_save(self.path, checked)


def _note(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("заметка должна быть непустой строкой")
    return value.strip()


def _message(value: Any) -> Message:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise ValueError("неверная структура сообщения")
    role = value["role"]
    content = value["content"]
    if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
        raise ValueError("неверная роль или пустое сообщение")
    return {"role": role, "content": content.strip()}


def _request_message(value: Any) -> Message:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise ValueError("неверная структура сообщения prompt")
    role = value["role"]
    content = value["content"]
    if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
        raise ValueError("неверная роль или пустое сообщение prompt")
    return {"role": role, "content": content.strip()}


class JsonSessionStore:
    """Атомарно хранит историю сессии и её последний фактический prompt."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[list[Message], list[Message]]:
        if not self.path.exists():
            return [], []
        try:
            with self.path.open(encoding="utf-8") as source:
                raw = json.load(source)
        except json.JSONDecodeError as error:
            raise MemoryError(f"Повреждён JSON памяти: {self.path.name}") from error
        except OSError as error:
            raise MemoryError(f"Не удалось прочитать память: {self.path.name}") from error

        # Совместимость с ранним форматом, где файл содержал только массив диалога.
        if isinstance(raw, list):
            raw_messages = raw
            raw_prompt: object = []
        elif isinstance(raw, dict) and set(raw) == {"messages", "last_request_messages"}:
            raw_messages = raw["messages"]
            raw_prompt = raw["last_request_messages"]
        else:
            raise MemoryError(f"Память {self.path.name} имеет неверную структуру")
        if not isinstance(raw_messages, list) or not isinstance(raw_prompt, list):
            raise MemoryError(f"Память {self.path.name} имеет неверную структуру")
        try:
            messages = [_message(item) for item in raw_messages]
            prompt = [_request_message(item) for item in raw_prompt]
        except (TypeError, ValueError) as error:
            raise MemoryError(f"Память {self.path.name} имеет неверную структуру") from error
        return messages, prompt

    def save(self, messages: list[Message], prompt: list[Message]) -> None:
        try:
            checked_messages = [_message(item) for item in messages]
            checked_prompt = [_request_message(item) for item in prompt]
        except (TypeError, ValueError) as error:
            raise MemoryError(f"Нельзя сохранить неверную память: {self.path.name}") from error
        _atomic_save(
            self.path,
            {
                "messages": checked_messages,
                "last_request_messages": checked_prompt,
            },
        )


class MemoryLayers:
    """Управляет scope пользователя, задачи и сессии без смешивания файлов."""

    def __init__(
        self,
        root: Path,
        *,
        user_id: str = "demo",
        task_id: str = "task-01",
        session_id: str = "session-01",
    ) -> None:
        self.root = root
        self.user_id = _validate_id(user_id, "user_id")
        self.task_id = _validate_id(task_id, "task_id")
        self.session_id = _validate_id(session_id, "session_id")
        self._reload()

    def _short_path(self, task_id: str, session_id: str) -> Path:
        nested = self.root / "short_term" / task_id / f"{session_id}.json"
        legacy = self.root / "short_term" / f"{session_id}.json"
        # Первые версии дня 11 не связывали сессии с задачей. Сохраняем доступ
        # к этим пользовательским данным как к истории task-01 без переноса файлов.
        if task_id == "task-01" and not nested.exists() and legacy.exists():
            return legacy
        return nested

    def _short_store(self) -> JsonSessionStore:
        return JsonSessionStore(self._short_path(self.task_id, self.session_id))

    def _working_store(self) -> JsonListStore:
        return JsonListStore(self.root / "working" / f"{self.task_id}.json", _note)

    def _long_store(self) -> JsonListStore:
        return JsonListStore(self.root / "long_term" / f"{self.user_id}.json", _note)

    def _reload(self) -> None:
        self._short_term, self._last_prompt = self._short_store().load()
        self._working: list[str] = self._working_store().load()
        self._long_term: list[str] = self._long_store().load()
        self._validate_exchanges(self._short_term)

    @staticmethod
    def _validate_exchanges(messages: list[Message]) -> None:
        if len(messages) % 2:
            raise MemoryError("Краткосрочная память должна состоять из завершённых пар")
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise MemoryError("В краткосрочной памяти нарушен порядок ролей")

    @property
    def short_term(self) -> tuple[Message, ...]:
        return tuple(message.copy() for message in self._short_term)

    @property
    def working(self) -> tuple[str, ...]:
        return tuple(self._working)

    @property
    def long_term(self) -> tuple[str, ...]:
        return tuple(self._long_term)

    @property
    def last_prompt(self) -> tuple[Message, ...]:
        return tuple(message.copy() for message in self._last_prompt)

    def add_working(self, note: str) -> None:
        value = _note(note)
        candidate = [*self._working, value]
        self._working_store().save(candidate)
        self._working = candidate

    def add_long_term(self, note: str) -> None:
        value = _note(note)
        candidate = [*self._long_term, value]
        self._long_store().save(candidate)
        self._long_term = candidate

    def commit_exchange(
        self,
        user_message: str,
        assistant_message: str,
        *,
        request_messages: list[Message] | None = None,
    ) -> None:
        candidate = [
            *self._short_term,
            _message({"role": "user", "content": user_message}),
            _message({"role": "assistant", "content": assistant_message}),
        ]
        prompt = [] if request_messages is None else [
            _request_message(message) for message in request_messages
        ]
        self._short_store().save(candidate, prompt)
        self._short_term = candidate
        self._last_prompt = prompt

    def new_session(self, session_id: str) -> None:
        candidate_id = _validate_id(session_id, "session_id")
        candidate, prompt = JsonSessionStore(
            self._short_path(self.task_id, candidate_id)
        ).load()
        self._validate_exchanges(candidate)
        self.session_id = candidate_id
        self._short_term = candidate
        self._last_prompt = prompt

    def create_session(self, session_id: str) -> None:
        candidate_id = _validate_id(session_id, "session_id")
        path = self._short_path(self.task_id, candidate_id)
        if path.exists():
            raise MemoryError(f"Сессия {candidate_id} уже существует")
        JsonSessionStore(path).save([], [])
        self.session_id = candidate_id
        self._short_term = []
        self._last_prompt = []

    def switch_task(self, task_id: str, *, session_id: str) -> None:
        candidate_task = _validate_id(task_id, "task_id")
        candidate_session = _validate_id(session_id, "session_id")
        short_term, prompt = JsonSessionStore(
            self._short_path(candidate_task, candidate_session)
        ).load()
        working = JsonListStore(
            self.root / "working" / f"{candidate_task}.json", _note
        ).load()
        self._validate_exchanges(short_term)
        self.task_id = candidate_task
        self.session_id = candidate_session
        self._short_term = short_term
        self._last_prompt = prompt
        self._working = working

    def switch_session(self, session_id: str) -> None:
        self.new_session(session_id)

    def create_task(self, task_id: str, *, session_id: str = "session-01") -> None:
        candidate_task = _validate_id(task_id, "task_id")
        candidate_session = _validate_id(session_id, "session_id")
        working_path = self.root / "working" / f"{candidate_task}.json"
        session_path = self._short_path(candidate_task, candidate_session)
        if working_path.exists() or session_path.exists():
            raise MemoryError(f"Задача {candidate_task} уже существует")
        JsonListStore(working_path, _note).save([])
        JsonSessionStore(session_path).save([], [])
        self.task_id = candidate_task
        self.session_id = candidate_session
        self._working = []
        self._short_term = []
        self._last_prompt = []

    def session_ids(self, task_id: str | None = None) -> tuple[str, ...]:
        selected_task = self.task_id if task_id is None else _validate_id(task_id, "task_id")
        names = {
            path.stem
            for path in (self.root / "short_term" / selected_task).glob("session-*.json")
        }
        if selected_task == "task-01":
            names.update(
                path.stem
                for path in (self.root / "short_term").glob("session-*.json")
            )
        return tuple(sorted(names, key=_numeric_suffix))

    def task_ids(self) -> tuple[str, ...]:
        names = {"task-01", self.task_id}
        names.update(path.stem for path in (self.root / "working").glob("task-*.json"))
        short_root = self.root / "short_term"
        if short_root.exists():
            names.update(path.name for path in short_root.iterdir() if path.is_dir())
        return tuple(sorted(names, key=_numeric_suffix))

    def clear_active(self) -> None:
        """Очищает только текущие три scope, не затрагивая другие задачи и сессии."""
        self._short_store().save([], [])
        self._working_store().save([])
        self._long_store().save([])
        self._short_term = []
        self._last_prompt = []
        self._working = []
        self._long_term = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "scope": {
                "user_id": self.user_id,
                "task_id": self.task_id,
                "session_id": self.session_id,
            },
            "short_term": [message.copy() for message in self._short_term],
            "last_prompt": [message.copy() for message in self._last_prompt],
            "working": list(self._working),
            "long_term": list(self._long_term),
        }


def _numeric_suffix(value: str) -> tuple[str, int]:
    prefix, separator, suffix = value.rpartition("-")
    return (prefix if separator else value, int(suffix) if suffix.isdigit() else 0)
