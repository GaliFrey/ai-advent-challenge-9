"""Локальная оценка токенов для зафиксированной модели эксперимента."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import tiktoken


SUPPORTED_MODEL = "openai/gpt-3.5-turbo-16k"


class UnsupportedTokenizerError(ValueError):
    """Для модели не зафиксирован совместимый локальный токенизатор."""


class TokenCounter:
    """Считает текст и оценивает OpenAI-формат массива messages.

    Содержимое считается кодировкой модели. Служебная обвязка chat messages
    оценивается по опубликованной схеме GPT-3.5 Turbo 0613. Фактический
    ``prompt_tokens`` из API всё равно остаётся источником истины.
    """

    def __init__(self, model: str):
        if model != SUPPORTED_MODEL:
            raise UnsupportedTokenizerError(
                f"Локальный счётчик поддерживает только {SUPPORTED_MODEL}"
            )
        self.model = model
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        # GPT-3.5 Turbo: три служебных токена на сообщение, один вместо name
        # и три токена для начала ответа assistant.
        total = 3
        for message in messages:
            total += 3
            for key, value in message.items():
                total += self.count_text(value)
                if key == "name":
                    total += 1
        return total
