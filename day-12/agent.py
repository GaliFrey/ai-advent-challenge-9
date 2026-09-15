"""DeepSeek-агент, применяющий активный профиль в каждом запросе."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from user_profile import Profile, render_profile
from session import Session, SessionStore


ROOT = Path(__file__).resolve().parent
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
SUPPORTED_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
BASE_SYSTEM_PROMPT = (
    "Ты полезный ассистент. Следуй активному профилю пользователя во всех ответах. "
    "Базовые правила безопасности и фактической точности имеют приоритет над профилем. "
    "Не сообщай, что применяешь профиль, если пользователь об этом не спрашивает."
)


class AgentError(RuntimeError):
    """Безопасная ошибка для показа в интерфейсе."""


@dataclass(frozen=True)
class AgentConfig:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    api_url: str = DEFAULT_API_URL
    max_tokens: int = 700
    temperature: float = 0.0

    def check(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Добавь DEEPSEEK_API_KEY в day-12/.env")
        if self.api_url != DEFAULT_API_URL:
            raise ValueError("Приложение рассчитано на официальный API DeepSeek")
        if not self.model.strip() or self.max_tokens <= 0:
            raise ValueError("Проверь модель и DEEPSEEK_MAX_TOKENS")


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class AgentReply:
    text: str
    usage: Usage
    elapsed_seconds: float
    finish_reason: str


def load_config() -> AgentConfig:
    load_dotenv(ROOT / ".env", override=False)
    try:
        max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "700"))
    except ValueError as error:
        raise ValueError("DEEPSEEK_MAX_TOKENS должен быть целым числом") from error
    return AgentConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens,
    )


def _usage(payload: dict[str, Any]) -> Usage:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        raise ValueError("usage отсутствует")
    values = (raw.get("prompt_tokens"), raw.get("completion_tokens"))
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("usage не содержит точные токены")
    input_tokens, output_tokens = values
    total = raw.get("total_tokens", input_tokens + output_tokens)
    if type(total) is not int or total != input_tokens + output_tokens:
        raise ValueError("total_tokens не совпадает с суммой")
    return Usage(input_tokens, output_tokens, total)


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        client: httpx.AsyncClient,
        session: Session,
        session_store: SessionStore,
        profile: Profile,
    ) -> None:
        config.check()
        self.config = config
        self.client = client
        self.session = session
        self.session_store = session_store
        self.profile = profile.normalized()
        self.total_tokens = 0
        self._lock = asyncio.Lock()

    def build_messages(self, user_message: str) -> list[dict[str, str]]:
        system = BASE_SYSTEM_PROMPT + "\n\n" + render_profile(self.profile)
        return [
            {"role": "system", "content": system},
            *[message.copy() for message in self.session.messages],
            {"role": "user", "content": user_message.strip()},
        ]

    async def ask(self, user_message: str) -> AgentReply:
        message = user_message.strip()
        if not message:
            raise AgentError("Сообщение не должно быть пустым")
        async with self._lock:
            messages = self.build_messages(message)
            started = time.perf_counter()
            try:
                response = await self.client.post(
                    self.config.api_url,
                    headers={"Authorization": "Bearer " + self.config.api_key},
                    json={
                        "model": self.config.model,
                        "messages": messages,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "stream": False,
                        "thinking": {"type": "disabled"},
                    },
                )
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "unknown")
                usage = _usage(payload)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("пустой ответ")
                if finish_reason != "stop":
                    raise AgentError(f"Ответ не завершён: {finish_reason}")
            except httpx.HTTPStatusError as error:
                raise AgentError(f"API вернул HTTP {error.response.status_code}") from error
            except AgentError:
                raise
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise AgentError("Некорректный ответ API; подробности скрыты") from error

            reply = AgentReply(text.strip(), usage, time.perf_counter() - started, finish_reason)
            candidate = Session(
                session_id=self.session.session_id,
                profile_id=self.profile.profile_id,
                messages=[
                    *self.session.messages,
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply.text},
                ],
                last_request_messages=[item.copy() for item in messages],
            )
            self.session_store.save(candidate)
            self.session.profile_id = candidate.profile_id
            self.session.messages = candidate.messages
            self.session.last_request_messages = candidate.last_request_messages
            self.total_tokens += usage.total_tokens
            return reply
