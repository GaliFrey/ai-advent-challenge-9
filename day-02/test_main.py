"""Локальные тесты формирования запросов и обработки ответов дня 2."""

import argparse
import unittest
from contextlib import redirect_stdout
from io import StringIO

from main import (
    CONTROLLED_MAX_TOKENS,
    MAX_CONTROLLED_RUNS,
    MODE_CONTROLLED,
    MODE_FREE,
    STOP_SEQUENCE,
    CompletionResult,
    build_request_payload,
    controlled_run_count,
    extract_completion,
    print_result,
    validate_controlled_completion,
)


PROMPT = "Составь рецепт салата «Цезарь» для двух человек"


class RequestPayloadTests(unittest.TestCase):
    """Проверяет различия режимов при неизменном запросе пользователя."""

    def test_free_mode_has_no_explicit_output_constraints(self) -> None:
        payload = build_request_payload(PROMPT, MODE_FREE)

        self.assertEqual(payload["messages"], [{"role": "user", "content": PROMPT}])
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("stop", payload)

    def test_controlled_mode_has_all_required_constraints(self) -> None:
        payload = build_request_payload(PROMPT, MODE_CONTROLLED)

        self.assertEqual(payload["messages"][-1], {"role": "user", "content": PROMPT})
        self.assertEqual(payload["max_tokens"], CONTROLLED_MAX_TOKENS)
        self.assertEqual(payload["stop"], [STOP_SEQUENCE])
        self.assertIn("строго в следующем формате Markdown", payload["messages"][0]["content"])

    def test_user_prompt_is_identical_in_both_modes(self) -> None:
        free_payload = build_request_payload(PROMPT, MODE_FREE)
        controlled_payload = build_request_payload(PROMPT, MODE_CONTROLLED)

        self.assertEqual(
            free_payload["messages"][-1]["content"],
            controlled_payload["messages"][-1]["content"],
        )


class CompletionParsingTests(unittest.TestCase):
    """Проверяет извлечение ответа и сравнительных метаданных."""

    def test_extracts_answer_and_metadata(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {"content": "Готовый ответ"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 42},
        }

        self.assertEqual(
            extract_completion(payload),
            CompletionResult(
                answer="Готовый ответ",
                finish_reason="stop",
                completion_tokens=42,
            ),
        )


class ResultOutputTests(unittest.TestCase):
    """Проверяет пояснение о невидимой stop-последовательности."""

    def test_controlled_output_explains_absent_stop_sequence(self) -> None:
        output = StringIO()
        result = CompletionResult(
            answer="# Салат",
            finish_reason="stop",
            completion_tokens=20,
        )

        with redirect_stdout(output):
            print_result(MODE_CONTROLLED, result)

        self.assertIn(
            f"{STOP_SEQUENCE} — служебная stop-последовательность",
            output.getvalue(),
        )


class ControlledValidationTests(unittest.TestCase):
    """Проверяет формат, длину и завершение контролируемого ответа."""

    def test_valid_controlled_answer_passes_all_checks(self) -> None:
        result = CompletionResult(
            answer="""# Салат Цезарь

## Ингредиенты
- Куриное филе — 200 г
- Салат ромэн — 150 г
- Пармезан — 50 г

## Приготовление
1. Обжарьте курицу.
2. Смешайте ингредиенты.
3. Посыпьте пармезаном.""",
            finish_reason="stop",
            completion_tokens=120,
        )

        validation = validate_controlled_completion(result)

        self.assertTrue(validation.passed)
        self.assertTrue(all(check.passed for check in validation.checks))

    def test_too_many_ingredients_fails_format_check(self) -> None:
        ingredients = "\n".join(f"- Ингредиент {number}" for number in range(1, 7))
        result = CompletionResult(
            answer=(
                "# Салат\n\n"
                f"## Ингредиенты\n{ingredients}\n\n"
                "## Приготовление\n1. Смешайте ингредиенты."
            ),
            finish_reason="stop",
            completion_tokens=100,
        )

        validation = validate_controlled_completion(result)

        self.assertFalse(validation.passed)
        self.assertFalse(validation.checks[0].passed)
        self.assertIn("получено 6", validation.checks[0].details)

    def test_length_finish_reason_fails_length_check(self) -> None:
        result = CompletionResult(
            answer="""# Салат

## Ингредиенты
- Огурец

## Приготовление
1. Нарежьте огурец.""",
            finish_reason="length",
            completion_tokens=CONTROLLED_MAX_TOKENS,
        )

        validation = validate_controlled_completion(result)

        self.assertFalse(validation.passed)
        self.assertFalse(validation.checks[1].passed)

    def test_run_count_is_bounded(self) -> None:
        self.assertEqual(controlled_run_count("10"), 10)

        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            f"от 1 до {MAX_CONTROLLED_RUNS}",
        ):
            controlled_run_count(str(MAX_CONTROLLED_RUNS + 1))


if __name__ == "__main__":
    unittest.main()
