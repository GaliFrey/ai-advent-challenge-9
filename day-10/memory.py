"""Три стратегии управления контекстом без summary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


Message = dict[str, str]


def _copy_messages(messages: Sequence[Message]) -> list[Message]:
    return [message.copy() for message in messages]


class SlidingWindowMemory:
    """Хранит полную локальную историю, но передаёт только последние N сообщений."""

    kind = "sliding"

    def __init__(self, keep_recent: int) -> None:
        if keep_recent <= 0 or keep_recent % 2:
            raise ValueError("N должно быть положительным чётным числом сообщений")
        self.keep_recent = keep_recent
        self._messages: list[Message] = []

    @property
    def context_messages(self) -> tuple[Message, ...]:
        return tuple(_copy_messages(self._messages[-self.keep_recent :]))

    @property
    def all_messages(self) -> tuple[Message, ...]:
        return tuple(_copy_messages(self._messages))

    @property
    def total_message_count(self) -> int:
        return len(self._messages)

    @property
    def context_message_count(self) -> int:
        return min(len(self._messages), self.keep_recent)

    @property
    def discarded_message_count(self) -> int:
        return max(0, len(self._messages) - self.keep_recent)

    def commit_exchange(self, user_message: str, assistant_message: str) -> None:
        self._messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )

    def clear(self) -> None:
        self._messages.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "keep_recent": self.keep_recent,
            "total_message_count": self.total_message_count,
            "context_message_count": self.context_message_count,
            "discarded_message_count": self.discarded_message_count,
            "all_messages": _copy_messages(self._messages),
            "context_messages": _copy_messages(self.context_messages),
        }


class FactsMemory:
    """Хранит key-value facts отдельно от короткого хвоста диалога."""

    kind = "facts"

    def __init__(self, keep_recent: int) -> None:
        if keep_recent <= 0 or keep_recent % 2:
            raise ValueError("N должно быть положительным чётным числом сообщений")
        self.keep_recent = keep_recent
        self._messages: list[Message] = []
        self._facts: dict[str, Any] = {}

    @property
    def context_messages(self) -> tuple[Message, ...]:
        return tuple(_copy_messages(self._messages[-self.keep_recent :]))

    @property
    def all_messages(self) -> tuple[Message, ...]:
        return tuple(_copy_messages(self._messages))

    @property
    def facts(self) -> dict[str, Any]:
        return deepcopy(self._facts)

    @property
    def total_message_count(self) -> int:
        return len(self._messages)

    @property
    def context_message_count(self) -> int:
        return min(len(self._messages), self.keep_recent)

    @property
    def discarded_message_count(self) -> int:
        return max(0, len(self._messages) - self.keep_recent)

    def commit_exchange(
        self,
        user_message: str,
        assistant_message: str,
        *,
        facts: Mapping[str, Any] | None = None,
    ) -> None:
        if facts is not None:
            self.replace_facts(facts)
        self._messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )

    def replace_facts(self, facts: Mapping[str, Any]) -> None:
        if not isinstance(facts, Mapping):
            raise ValueError("Facts должны быть JSON-объектом")
        self._facts = deepcopy(dict(facts))

    def clear(self) -> None:
        self._messages.clear()
        self._facts.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "keep_recent": self.keep_recent,
            "total_message_count": self.total_message_count,
            "context_message_count": self.context_message_count,
            "discarded_message_count": self.discarded_message_count,
            "facts": self.facts,
            "all_messages": _copy_messages(self._messages),
            "context_messages": _copy_messages(self.context_messages),
        }


class BranchingMemory:
    """Общий префикс и независимые продолжения от одного checkpoint."""

    kind = "branching"

    def __init__(self) -> None:
        self._common: list[Message] = []
        self._branches: dict[str, list[Message]] = {}
        self._active_branch: str | None = None

    @property
    def has_checkpoint(self) -> bool:
        return bool(self._branches)

    @property
    def active_branch(self) -> str | None:
        return self._active_branch

    @property
    def branches(self) -> tuple[str, ...]:
        return tuple(self._branches)

    @property
    def common_messages(self) -> tuple[Message, ...]:
        return tuple(_copy_messages(self._common))

    @property
    def context_messages(self) -> tuple[Message, ...]:
        messages = _copy_messages(self._common)
        if self._active_branch is not None:
            messages.extend(_copy_messages(self._branches[self._active_branch]))
        return tuple(messages)

    @property
    def total_message_count(self) -> int:
        return len(self._common) + sum(len(messages) for messages in self._branches.values())

    @property
    def context_message_count(self) -> int:
        return len(self.context_messages)

    def create_checkpoint(self, branch_names: Sequence[str] = ("A", "B")) -> None:
        if self.has_checkpoint:
            raise ValueError("Checkpoint уже создан")
        if not self._common:
            raise ValueError("Нельзя создать checkpoint в пустом диалоге")
        names = tuple(name.strip() for name in branch_names)
        if len(names) < 2 or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Нужны минимум две ветки с уникальными именами")
        self._branches = {name: [] for name in names}
        self._active_branch = names[0]

    def switch_branch(self, name: str) -> None:
        if name not in self._branches:
            raise ValueError(f"Неизвестная ветка: {name}")
        self._active_branch = name

    def commit_exchange(self, user_message: str, assistant_message: str) -> None:
        exchange = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        if self._active_branch is None:
            self._common.extend(exchange)
        else:
            self._branches[self._active_branch].extend(exchange)

    def clear(self) -> None:
        self._common.clear()
        self._branches.clear()
        self._active_branch = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "has_checkpoint": self.has_checkpoint,
            "active_branch": self.active_branch,
            "total_message_count": self.total_message_count,
            "context_message_count": self.context_message_count,
            "common_messages": _copy_messages(self._common),
            "branches": {
                name: _copy_messages(messages) for name, messages in self._branches.items()
            },
            "context_messages": _copy_messages(self.context_messages),
        }
