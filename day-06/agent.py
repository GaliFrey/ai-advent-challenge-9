"""Изолированный агент с собственной историей и статистикой."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_SYSTEM_PROMPT = (
    "Ты полезный ассистент. Отвечай кратко и по-русски. "
    "Используй сведения только из текущего диалога. "
    "Если пользователь не сообщал запрошенную информацию, прямо скажи, что она неизвестна."
)


class AgentError(RuntimeError):
    """Безопасная ошибка агента, которую можно показать пользователю."""


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    api_url: str = DEFAULT_API_URL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.0
    max_tokens: int = 500

    def check(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Добавь DEEPSEEK_API_KEY в day-06/.env")
        if not self.model.strip():
            raise ValueError("Название модели не должно быть пустым")
        if not self.system_prompt.strip():
            raise ValueError("Системный промпт не должен быть пустым")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature должна находиться в диапазоне от 0 до 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens должен быть положительным")


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class AgentReply:
    text: str
    usage: Usage
    elapsed_seconds: float
    finish_reason: str


@dataclass(frozen=True)
class AgentStats:
    attempts: int = 0
    successful_requests: int = 0
    history_messages: int = 0
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    session_total_tokens: int = 0
    last_usage: Usage | None = None
    last_elapsed_seconds: float | None = None
    last_finish_reason: str | None = None
    last_error: str | None = None


def load_config() -> AgentConfig:
    load_dotenv(ROOT / ".env", override=False)
    raw_max_tokens = os.getenv("DEEPSEEK_MAX_TOKENS", "500")
    raw_temperature = os.getenv("DEEPSEEK_TEMPERATURE", "0")
    try:
        max_tokens = int(raw_max_tokens)
        temperature = float(raw_temperature)
    except ValueError as error:
        raise ValueError("Проверь DEEPSEEK_MAX_TOKENS и DEEPSEEK_TEMPERATURE в day-06/.env") from error
    return AgentConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _non_negative_integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def parse_usage(payload: dict[str, Any]) -> Usage:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        raw = {}
    input_tokens = _non_negative_integer(raw.get("prompt_tokens"))
    output_tokens = _non_negative_integer(raw.get("completion_tokens"))
    total_tokens = _non_negative_integer(raw.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return Usage(input_tokens, output_tokens, total_tokens)


class Agent:
    """Один независимо создаваемый агент и его краткосрочная сессия."""

    def __init__(self, config: AgentConfig, client: httpx.AsyncClient):
        config.check()
        self.config = config
        self._client = client
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": config.system_prompt}
        ]
        self._stats = AgentStats()
        self._lock = asyncio.Lock()

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        """Защитная копия истории для диагностики и тестов."""
        return tuple(message.copy() for message in self._messages)

    @property
    def stats(self) -> AgentStats:
        return self._stats

    async def ask(self, user_message: str) -> AgentReply:
        text = user_message.strip()
        if not text:
            raise AgentError("Сообщение не должно быть пустым")

        async with self._lock:
            candidate_messages = [*self._messages, {"role": "user", "content": text}]
            request = {
                "model": self.config.model,
                "messages": candidate_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False,
                "thinking": {"type": "disabled"},
            }
            started = time.perf_counter()
            self._stats = replace(
                self._stats,
                attempts=self._stats.attempts + 1,
                last_error=None,
            )
            try:
                response = await self._client.post(
                    self.config.api_url,
                    headers={"Authorization": "Bearer " + self.config.api_key},
                    json=request,
                )
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                answer = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "unknown")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty answer")
                if not isinstance(finish_reason, str):
                    finish_reason = "unknown"
                usage = parse_usage(payload)
            except httpx.HTTPStatusError as error:
                message = f"API вернул HTTP {error.response.status_code}; автоматического повтора нет"
                self._record_error(message, started)
                raise AgentError(message) from error
            except httpx.TimeoutException as error:
                message = "Истекло время ожидания; сервер мог обработать запрос"
                self._record_error(message, started)
                raise AgentError(message) from error
            except httpx.RequestError as error:
                message = "Не удалось подключиться к API"
                self._record_error(message, started)
                raise AgentError(message) from error
            except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
                message = "API не вернул ожидаемый текстовый ответ"
                self._record_error(message, started)
                raise AgentError(message) from error

            elapsed = time.perf_counter() - started
            answer = answer.strip()
            self._messages = [
                *candidate_messages,
                {"role": "assistant", "content": answer},
            ]
            self._stats = replace(
                self._stats,
                successful_requests=self._stats.successful_requests + 1,
                history_messages=len(self._messages) - 1,
                session_input_tokens=self._stats.session_input_tokens + (usage.input_tokens or 0),
                session_output_tokens=self._stats.session_output_tokens + (usage.output_tokens or 0),
                session_total_tokens=self._stats.session_total_tokens + (usage.total_tokens or 0),
                last_usage=usage,
                last_elapsed_seconds=elapsed,
                last_finish_reason=finish_reason,
                last_error=None,
            )
            return AgentReply(answer, usage, elapsed, finish_reason)

    def _record_error(self, message: str, started: float) -> None:
        self._stats = replace(
            self._stats,
            last_elapsed_seconds=time.perf_counter() - started,
            last_finish_reason=None,
            last_error=message,
        )
