#!/usr/bin/env python3
"""Сравнение свободного и контролируемого ответов DeepSeek API."""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
ENV_FILE = Path(__file__).with_name(".env")

MODE_FREE = "free"
MODE_CONTROLLED = "controlled"
MODE_COMPARE = "compare"

CONTROLLED_MAX_TOKENS = 220
MAX_CONTROLLED_RUNS = 20
STOP_SEQUENCE = "<END>"
CONTROLLED_SYSTEM_PROMPT = f"""\
Ответь строго в следующем формате Markdown:

# Название
Краткое название блюда.

## Ингредиенты
Маркированный список максимум из пяти ингредиентов с количеством.

## Приготовление
Нумерованный список максимум из четырех коротких шагов.

Не добавляй вступление, заключение и другие разделы.
После последнего шага напиши отдельной строкой маркер {STOP_SEQUENCE}.
"""


class DeepSeekApiError(RuntimeError):
    """Ошибка обращения к DeepSeek API."""


@dataclass(frozen=True)
class CompletionResult:
    """Ответ модели и метаданные, полезные для сравнения режимов."""

    answer: str
    finish_reason: str
    completion_tokens: int | None


@dataclass(frozen=True)
class ValidationCheck:
    """Результат одной проверки контролируемого ответа."""

    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class ValidationResult:
    """Совокупный результат проверок контролируемого ответа."""

    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        """Возвращает True, только если пройдены все проверки."""
        return all(check.passed for check in self.checks)


def build_request_payload(prompt: str, mode: str) -> dict[str, Any]:
    """Формирует тело API-запроса для выбранного режима."""
    if mode == MODE_FREE:
        messages = [{"role": "user", "content": prompt}]
        return {
            "model": MODEL,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "stream": False,
        }

    if mode == MODE_CONTROLLED:
        messages = [
            {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return {
            "model": MODEL,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "max_tokens": CONTROLLED_MAX_TOKENS,
            "stop": [STOP_SEQUENCE],
            "stream": False,
        }

    raise ValueError(f"Неизвестный режим: {mode}")


def extract_error_message(raw_body: str) -> str:
    """Извлекает безопасное описание ошибки из ответа API."""
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body.strip() or "API вернул пустой ответ"

    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else None
    return message if isinstance(message, str) else raw_body.strip()


def extract_completion(payload: dict[str, Any]) -> CompletionResult:
    """Извлекает текст ответа, причину завершения и число токенов."""
    try:
        choice = payload["choices"][0]
        answer = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise DeepSeekApiError("API вернул ответ неожиданного формата") from error

    if not isinstance(answer, str) or not answer.strip():
        raise DeepSeekApiError("Модель вернула пустой ответ")

    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        finish_reason = "unknown"

    usage = payload.get("usage")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(completion_tokens, int):
        completion_tokens = None

    return CompletionResult(
        answer=answer.strip(),
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
    )


def validate_markdown_format(answer: str) -> tuple[str, ...]:
    """Проверяет структуру Markdown, число ингредиентов и шагов."""
    lines = [line.strip() for line in answer.splitlines()]
    errors: list[str] = []

    title_indices = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"#\s+\S.*", line) is not None
    ]
    if len(title_indices) != 1:
        errors.append("ожидался один заголовок блюда первого уровня")

    ingredient_indices = [
        index for index, line in enumerate(lines) if line == "## Ингредиенты"
    ]
    preparation_indices = [
        index for index, line in enumerate(lines) if line == "## Приготовление"
    ]
    if len(ingredient_indices) != 1:
        errors.append("ожидался один раздел «Ингредиенты»")
    if len(preparation_indices) != 1:
        errors.append("ожидался один раздел «Приготовление»")

    allowed_level_two_headings = {"## Ингредиенты", "## Приготовление"}
    unexpected_headings = [
        line
        for line in lines
        if line.startswith("## ") and line not in allowed_level_two_headings
    ]
    if unexpected_headings:
        errors.append("обнаружены лишние разделы")

    if len(title_indices) == len(ingredient_indices) == len(preparation_indices) == 1:
        title_index = title_indices[0]
        ingredient_index = ingredient_indices[0]
        preparation_index = preparation_indices[0]

        if not title_index < ingredient_index < preparation_index:
            errors.append("разделы расположены в неверном порядке")
        else:
            ingredient_lines = lines[ingredient_index + 1 : preparation_index]
            ingredients = [
                line
                for line in ingredient_lines
                if re.fullmatch(r"[-*]\s+\S.*", line) is not None
            ]
            preparation_lines = lines[preparation_index + 1 :]
            steps = [
                line
                for line in preparation_lines
                if re.fullmatch(r"\d+[.)]\s+\S.*", line) is not None
            ]

            if not 1 <= len(ingredients) <= 5:
                errors.append(f"ожидалось от 1 до 5 ингредиентов, получено {len(ingredients)}")
            if not 1 <= len(steps) <= 4:
                errors.append(f"ожидалось от 1 до 4 шагов, получено {len(steps)}")

    return tuple(errors)


def validate_controlled_completion(result: CompletionResult) -> ValidationResult:
    """Проверяет формат, длину и завершение контролируемого ответа."""
    format_errors = validate_markdown_format(result.answer)
    format_check = ValidationCheck(
        name="Формат",
        passed=not format_errors,
        details="структура Markdown соблюдена" if not format_errors else "; ".join(format_errors),
    )

    length_passed = (
        result.completion_tokens is not None
        and result.completion_tokens <= CONTROLLED_MAX_TOKENS
        and result.finish_reason != "length"
    )
    if result.completion_tokens is None:
        length_details = "API не вернул completion_tokens"
    elif result.finish_reason == "length":
        length_details = f"ответ оборван на лимите {CONTROLLED_MAX_TOKENS} токенов"
    else:
        length_details = (
            f"{result.completion_tokens}/{CONTROLLED_MAX_TOKENS} токенов, "
            "ответ не оборван по лимиту"
        )
    length_check = ValidationCheck(
        name="Длина",
        passed=length_passed,
        details=length_details,
    )

    finish_passed = result.finish_reason == "stop" and STOP_SEQUENCE not in result.answer
    finish_check = ValidationCheck(
        name="Завершение",
        passed=finish_passed,
        details=(
            "finish_reason=stop, служебный маркер отсутствует в тексте"
            if finish_passed
            else f"finish_reason={result.finish_reason}, завершение не подтверждено"
        ),
    )

    return ValidationResult(checks=(format_check, length_check, finish_check))


def ask_deepseek(prompt: str, api_key: str, mode: str) -> CompletionResult:
    """Отправляет запрос в DeepSeek и возвращает ответ с метаданными."""
    body = json.dumps(
        build_request_payload(prompt, mode),
        ensure_ascii=False,
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

    return extract_completion(response_payload)


def print_mode_settings(mode: str) -> None:
    """Показывает настройки режима перед ответом модели."""
    if mode == MODE_FREE:
        print("Настройки: формат — свободный; max_tokens — не задан; stop — не задан")
        return

    print(
        "Настройки: формат — фиксированный Markdown; "
        f"max_tokens — {CONTROLLED_MAX_TOKENS}; stop — {STOP_SEQUENCE}"
    )


def print_result(mode: str, result: CompletionResult) -> None:
    """Выводит результат одного режима в удобном для сравнения виде."""
    title = "Свободный ответ" if mode == MODE_FREE else "Контролируемый ответ"
    print(f"\n=== {title} ===")
    print_mode_settings(mode)
    print(f"\n{result.answer}")

    token_count = str(result.completion_tokens) if result.completion_tokens is not None else "нет данных"
    print(f"\nМетаданные: completion_tokens={token_count}, finish_reason={result.finish_reason}")

    if mode == MODE_CONTROLLED:
        print(
            f"Примечание: {STOP_SEQUENCE} — служебная stop-последовательность; "
            "API не включает её в текст ответа."
        )

    if result.finish_reason == "length":
        print("Предупреждение: ответ оборван из-за ограничения длины.")


def print_validation(result: ValidationResult) -> None:
    """Выводит подробный результат валидации контролируемого ответа."""
    print("\nВалидация контролируемого ответа:")
    for check in result.checks:
        status = "OK" if check.passed else "FAIL"
        print(f"- {check.name}: {status} — {check.details}")

    summary = "PASS" if result.passed else "FAIL"
    print(f"Итог валидации: {summary}")


def controlled_run_count(value: str) -> int:
    """Проверяет безопасное число повторов контролируемого запроса."""
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("число запусков должно быть целым") from error

    if not 1 <= count <= MAX_CONTROLLED_RUNS:
        raise argparse.ArgumentTypeError(
            f"число запусков должно быть от 1 до {MAX_CONTROLLED_RUNS}"
        )

    return count


def parse_arguments() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Сравнивает один запрос к DeepSeek с разным уровнем контроля ответа."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Запрос к модели. Если не указан, программа запросит его интерактивно.",
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_FREE, MODE_CONTROLLED, MODE_COMPARE),
        default=MODE_COMPARE,
        help="Режим запуска (по умолчанию: compare).",
    )
    parser.add_argument(
        "--runs",
        type=controlled_run_count,
        default=1,
        help=(
            "Число контролируемых вызовов для проверки стабильности формата "
            f"(1–{MAX_CONTROLLED_RUNS}, только с --mode controlled)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Получает один запрос и запускает выбранные режимы сравнения."""
    arguments = parse_arguments()
    load_dotenv(dotenv_path=ENV_FILE)

    if arguments.runs != 1 and arguments.mode != MODE_CONTROLLED:
        print(
            "Ошибка: --runs больше 1 можно использовать только с --mode controlled.",
            file=sys.stderr,
        )
        return 2

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("Ошибка: добавьте DEEPSEEK_API_KEY в файл day-02/.env.", file=sys.stderr)
        return 1

    prompt = " ".join(arguments.prompt).strip()
    if not prompt:
        prompt = input("Ваш запрос: ").strip()

    if not prompt:
        print("Ошибка: запрос не должен быть пустым.", file=sys.stderr)
        return 1

    modes = (
        (MODE_FREE, MODE_CONTROLLED)
        if arguments.mode == MODE_COMPARE
        else (arguments.mode,)
    )

    print(f"Запрос для всех режимов: {prompt}")

    validation_results: list[ValidationResult] = []

    try:
        for mode in modes:
            run_count = arguments.runs if mode == MODE_CONTROLLED else 1
            for run_number in range(1, run_count + 1):
                if run_count > 1:
                    print(f"\n--- Контролируемый запуск {run_number}/{run_count} ---")

                result = ask_deepseek(prompt, api_key, mode)
                print_result(mode, result)

                if mode == MODE_CONTROLLED:
                    validation = validate_controlled_completion(result)
                    print_validation(validation)
                    validation_results.append(validation)
    except DeepSeekApiError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    if len(validation_results) > 1:
        passed_count = sum(result.passed for result in validation_results)
        print("\n=== Итог серии контролируемых запросов ===")
        print(f"Валидацию прошли: {passed_count}/{len(validation_results)}")

    if any(not result.passed for result in validation_results):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
