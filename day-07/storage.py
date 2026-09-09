"""Атомарное JSON-хранилище истории одной сессии."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


SESSION_ID_PATTERN = re.compile(r"[a-z0-9_-]+")


class HistoryStorageError(RuntimeError):
    """Ошибка чтения, проверки или записи истории."""


class JsonHistoryStore:
    """Хранит пары user/assistant в отдельном JSON-файле сессии."""

    def __init__(self, directory: Path, session_id: str):
        if SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError("Идентификатор сессии содержит недопустимые символы")
        self.path = directory / f"{session_id}.json"

    def load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8") as source:
                raw_messages = json.load(source)
        except json.JSONDecodeError as error:
            raise HistoryStorageError(
                f"История {self.path.name} содержит повреждённый JSON"
            ) from error
        except OSError as error:
            raise HistoryStorageError(
                f"Не удалось прочитать историю {self.path.name}"
            ) from error
        return self._validate(raw_messages)

    def save(self, messages: Sequence[Mapping[str, str]]) -> None:
        checked = self._validate(messages)
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                json.dump(checked, destination, ensure_ascii=False, indent=2)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, self.path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise HistoryStorageError(
                f"Не удалось сохранить историю {self.path.name}"
            ) from error

    def clear(self) -> None:
        self.save([])

    def _validate(self, raw_messages: object) -> list[dict[str, str]]:
        if not isinstance(raw_messages, list):
            self._invalid("корень должен быть JSON-массивом")

        checked: list[dict[str, str]] = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                self._invalid(f"сообщение {index + 1} имеет неверную структуру")
            expected_role = "user" if index % 2 == 0 else "assistant"
            role = message["role"]
            content = message["content"]
            if role != expected_role:
                self._invalid(
                    f"сообщение {index + 1} должно иметь роль {expected_role}"
                )
            if not isinstance(content, str) or not content.strip():
                self._invalid(f"сообщение {index + 1} не содержит текст")
            checked.append({"role": role, "content": content})

        if len(checked) % 2:
            self._invalid("история должна состоять из завершённых пар user/assistant")
        return checked

    def _invalid(self, reason: str) -> None:
        raise HistoryStorageError(f"История {self.path.name} неверна: {reason}")
