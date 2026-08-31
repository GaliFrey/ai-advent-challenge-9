#!/usr/bin/env python3
"""Минимальный CLI-клиент для DeepSeek API."""

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
ENV_FILE = Path(__file__).with_name(".env")


class DeepSeekApiError(RuntimeError):
    """Ошибка обращения к DeepSeek API."""


def extract_error_message(raw_body: str) -> str:
    """Извлекает безопасное описание ошибки из ответа API."""
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body.strip() or "API вернул пустой ответ"

    message = payload.get("error", {}).get("message")
    return message if isinstance(message, str) else raw_body.strip()


def extract_answer(payload: dict[str, Any]) -> str:
    """Возвращает текст первого ответа модели."""
    try:
        answer = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise DeepSeekApiError("API вернул ответ неожиданного формата") from error

    if not isinstance(answer, str) or not answer.strip():
        raise DeepSeekApiError("Модель вернула пустой ответ")

    return answer.strip()


def ask_deepseek(prompt: str, api_key: str) -> str:
    """Отправляет запрос в DeepSeek и возвращает ответ модели."""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты полезный ассистент. Отвечай кратко и по-русски.",
                },
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "max_tokens": 300,
            "stream": False,
        }
    ).encode("utf-8")

    request = Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        message = extract_error_message(error_body)
        raise DeepSeekApiError(f"DeepSeek API вернул ошибку {error.code}: {message}") from error
    except URLError as error:
        raise DeepSeekApiError(f"Не удалось подключиться к DeepSeek API: {error.reason}") from error

    try:
        response_payload = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise DeepSeekApiError("DeepSeek API вернул некорректный JSON") from error

    return extract_answer(response_payload)


def main() -> int:
    """Читает запрос пользователя и выводит ответ DeepSeek в консоль."""
    load_dotenv(dotenv_path=ENV_FILE)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("Ошибка: добавьте DEEPSEEK_API_KEY в файл day-01/.env.", file=sys.stderr)
        return 1

    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = input("Ваш запрос: ").strip()

    if not prompt:
        print("Ошибка: запрос не должен быть пустым.", file=sys.stderr)
        return 1

    try:
        answer = ask_deepseek(prompt, api_key)
    except DeepSeekApiError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    print(f"\nDeepSeek:\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
