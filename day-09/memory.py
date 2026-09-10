"""Стратегии формирования контекста для полного и сжатого диалога."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


Message = dict[str, str]


def _copy_messages(messages: Sequence[Message]) -> list[Message]:
    return [message.copy() for message in messages]


@dataclass(frozen=True)
class CompressionPlan:
    """Снимок старой части истории, которую можно заменить summary."""

    previous_summary: str | None
    messages: tuple[Message, ...]
    prefix_length: int


class ConversationMemory(Protocol):
    @property
    def context_messages(self) -> tuple[Message, ...]: ...

    @property
    def summary(self) -> str | None: ...

    @property
    def raw_message_count(self) -> int: ...

    @property
    def summarized_message_count(self) -> int: ...

    @property
    def total_message_count(self) -> int: ...

    def commit_exchange(self, user_message: str, assistant_message: str) -> None: ...

    def compression_plan(self) -> CompressionPlan | None: ...

    def commit_summary(self, plan: CompressionPlan, summary: str) -> None: ...

    def clear(self) -> None: ...


class FullHistoryMemory:
    """Передаёт модели все завершённые сообщения без преобразований."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    @property
    def context_messages(self) -> tuple[Message, ...]:
        return tuple(_copy_messages(self._messages))

    @property
    def summary(self) -> None:
        return None

    @property
    def raw_message_count(self) -> int:
        return len(self._messages)

    @property
    def summarized_message_count(self) -> int:
        return 0

    @property
    def total_message_count(self) -> int:
        return len(self._messages)

    def commit_exchange(self, user_message: str, assistant_message: str) -> None:
        self._messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])

    def compression_plan(self) -> None:
        return None

    def commit_summary(self, plan: CompressionPlan, summary: str) -> None:
        raise RuntimeError("Полная история не поддерживает summary")

    def clear(self) -> None:
        self._messages.clear()


class SummaryMemory:
    """Хранит recursive summary отдельно от недавних исходных сообщений."""

    def __init__(self, *, keep_recent: int, interval: int = 10) -> None:
        if keep_recent <= 0 or keep_recent % 2:
            raise ValueError("N должно быть положительным чётным числом сообщений")
        if interval <= 0 or interval % 2:
            raise ValueError("Интервал должен быть положительным чётным числом")
        if keep_recent >= interval:
            raise ValueError("N должно быть меньше интервала суммаризации")
        self.keep_recent = keep_recent
        self.interval = interval
        self._recent: list[Message] = []
        self._summary: str | None = None
        self._summarized_message_count = 0
        self._messages_since_summary = 0
        self._summary_version = 0

    @property
    def context_messages(self) -> tuple[Message, ...]:
        return tuple(_copy_messages(self._recent))

    @property
    def summary(self) -> str | None:
        return self._summary

    @property
    def summary_version(self) -> int:
        return self._summary_version

    @property
    def messages_until_summary(self) -> int:
        return max(0, self.interval - self._messages_since_summary)

    @property
    def raw_message_count(self) -> int:
        return len(self._recent)

    @property
    def summarized_message_count(self) -> int:
        return self._summarized_message_count

    @property
    def total_message_count(self) -> int:
        return self._summarized_message_count + len(self._recent)

    def commit_exchange(self, user_message: str, assistant_message: str) -> None:
        self._recent.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])
        self._messages_since_summary += 2

    def compression_plan(self) -> CompressionPlan | None:
        if self._messages_since_summary < self.interval:
            return None
        prefix_length = len(self._recent) - self.keep_recent
        if prefix_length <= 0 or prefix_length % 2:
            raise RuntimeError("Нельзя сжать незавершённую пару сообщений")
        return CompressionPlan(
            previous_summary=self._summary,
            messages=tuple(_copy_messages(self._recent[:prefix_length])),
            prefix_length=prefix_length,
        )

    def commit_summary(self, plan: CompressionPlan, summary: str) -> None:
        text = summary.strip()
        if not text:
            raise ValueError("Summary не должно быть пустым")
        current_prefix = tuple(_copy_messages(self._recent[:plan.prefix_length]))
        if current_prefix != plan.messages or self._summary != plan.previous_summary:
            raise RuntimeError("История изменилась во время суммаризации")
        del self._recent[:plan.prefix_length]
        self._summary = text
        self._summarized_message_count += plan.prefix_length
        self._messages_since_summary = 0
        self._summary_version += 1

    def clear(self) -> None:
        self._recent.clear()
        self._summary = None
        self._summarized_message_count = 0
        self._messages_since_summary = 0
        self._summary_version = 0
