"""Единый тип ошибки локального сервера поддержки.

Ошибка несёт HTTP-статус и машиночитаемый код; формат ответа описан в app.py.
"""

from __future__ import annotations


class SupportError(Exception):
    """Ошибка обработки запроса с понятным клиенту кодом и статусом."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
