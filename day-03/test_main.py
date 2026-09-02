"""Локальные тесты четырёх способов рассуждения дня 3."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from main import (
    COMPARISON_SYSTEM_PROMPT,
    GENERATED_PROMPT_REQUEST,
    MAX_TOKENS,
    METHOD_DIRECT,
    METHOD_EXPERTS,
    METHOD_GENERATED_PROMPT,
    METHOD_STEP_BY_STEP,
    MODEL,
    CompletionResult,
    TokenUsage,
    build_comparison_prompt,
    build_request_payload,
    build_solution_prompt,
    compare_saved_results,
    evaluate_expression,
    execute_method,
    extract_completion,
    main,
    parse_arguments,
    run_llm_comparison,
    save_method_result,
    serialize_method_result,
    validate_solution,
)


def completion(answer: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> CompletionResult:
    """Создаёт тестовый ответ без обращения к API."""
    return CompletionResult(
        answer=answer,
        finish_reason="stop",
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class PromptTests(unittest.TestCase):
    """Проверяет различия промптов при общей исходной задаче."""

    def test_direct_prompt_has_no_reasoning_method(self) -> None:
        prompt = build_solution_prompt(METHOD_DIRECT)

        self.assertNotIn("Решай пошагово", prompt)
        self.assertNotIn("Аналитик", prompt)
        self.assertIn("FINAL:", prompt)

    def test_step_by_step_adds_only_its_reasoning_instruction(self) -> None:
        direct = build_solution_prompt(METHOD_DIRECT)
        prompt = build_solution_prompt(METHOD_STEP_BY_STEP)

        self.assertTrue(prompt.startswith(direct))
        self.assertIn("Решай пошагово", prompt)

    def test_experts_prompt_requests_every_role(self) -> None:
        prompt = build_solution_prompt(METHOD_EXPERTS)

        for role in ("ANALYST:", "ENGINEER:", "CRITIC:"):
            self.assertIn(role, prompt)

    def test_generated_prompt_request_does_not_contain_known_solution(self) -> None:
        self.assertIn("не решай задачу сам", GENERATED_PROMPT_REQUEST)
        self.assertNotIn("8 / (3 - 8 / 3)", GENERATED_PROMPT_REQUEST)

    def test_all_calls_use_same_model_settings(self) -> None:
        payload = build_request_payload("Проверка")

        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], MAX_TOKENS)

    def test_comparison_uses_a_separate_system_role(self) -> None:
        payload = build_request_payload("Данные", COMPARISON_SYSTEM_PROMPT)

        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], COMPARISON_SYSTEM_PROMPT)
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Данные"})


class CompletionParsingTests(unittest.TestCase):
    """Проверяет извлечение текста и всех сравнительных метрик."""

    def test_extracts_answer_and_usage(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {"content": "FINAL: 8 / (3 - 8 / 3)"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }

        result = extract_completion(payload)

        self.assertEqual(result.answer, "FINAL: 8 / (3 - 8 / 3)")
        self.assertEqual(result.usage, TokenUsage(15, 25, 40))


class ExpressionValidationTests(unittest.TestCase):
    """Проверяет точный и безопасный валидатор выражений."""

    def test_valid_expression_is_exactly_24(self) -> None:
        value, numbers = evaluate_expression("8 / (3 - 8 / 3)")

        self.assertEqual(value, Fraction(24))
        self.assertEqual(sorted(numbers), [3, 3, 8, 8])

    def test_valid_answer_passes(self) -> None:
        validation = validate_solution(
            "Проверим вычисление.\nFINAL: 8 / (3 - 8 / 3)",
            METHOD_DIRECT,
        )

        self.assertTrue(validation.passed)

    def test_wrong_number_set_fails(self) -> None:
        validation = validate_solution("FINAL: 24", METHOD_DIRECT)

        self.assertFalse(validation.passed)
        self.assertFalse(next(check for check in validation.checks if check.name == "Исходные числа").passed)

    def test_unsafe_python_is_rejected(self) -> None:
        validation = validate_solution(
            "FINAL: __import__('os').system('echo unsafe')",
            METHOD_DIRECT,
        )

        self.assertFalse(validation.passed)
        self.assertFalse(next(check for check in validation.checks if check.name == "Безопасный синтаксис").passed)

    def test_multiple_final_lines_fail_format(self) -> None:
        validation = validate_solution(
            "FINAL: 8 / (3 - 8 / 3)\nFINAL: 8 / (3 - 8 / 3)",
            METHOD_DIRECT,
        )

        self.assertFalse(validation.passed)
        self.assertIsNone(validation.expression)

    def test_final_line_must_be_last_non_empty_line(self) -> None:
        validation = validate_solution(
            "FINAL: 8 / (3 - 8 / 3)\nДополнительное пояснение",
            METHOD_DIRECT,
        )

        self.assertFalse(validation.passed)
        self.assertIn("последней непустой", validation.checks[0].details)

    def test_experts_require_all_role_labels(self) -> None:
        validation = validate_solution(
            "ANALYST: готово\nFINAL: 8 / (3 - 8 / 3)",
            METHOD_EXPERTS,
        )

        self.assertFalse(validation.passed)
        self.assertIn("ENGINEER", validation.checks[0].details)


class ExecutionTests(unittest.TestCase):
    """Проверяет одно- и двухэтапное выполнение с подменённым API."""

    def test_direct_uses_one_call(self) -> None:
        prompts: list[str] = []

        def requester(prompt: str, api_key: str) -> CompletionResult:
            prompts.append(prompt)
            self.assertEqual(api_key, "test-key")
            return completion("FINAL: 8 / (3 - 8 / 3)")

        with redirect_stdout(StringIO()):
            result = execute_method(METHOD_DIRECT, "test-key", requester)

        self.assertEqual(len(prompts), 1)
        self.assertTrue(result.validation.passed)

    def test_generated_prompt_is_used_unchanged_for_second_call(self) -> None:
        generated_prompt = "Новый точный промпт с форматом FINAL: <выражение>"
        prompts: list[str] = []

        def requester(prompt: str, api_key: str) -> CompletionResult:
            prompts.append(prompt)
            if len(prompts) == 1:
                return completion(generated_prompt)
            return completion("FINAL: 8 / (3 - 8 / 3)")

        with redirect_stdout(StringIO()):
            result = execute_method(METHOD_GENERATED_PROMPT, "test-key", requester)

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[1], generated_prompt)
        self.assertTrue(result.validation.passed)

    def test_successful_console_output_is_compact_and_does_not_repeat_task(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            execute_method(
                METHOD_DIRECT,
                "test-key",
                lambda prompt, api_key: completion("FINAL: 8 / (3 - 8 / 3)"),
            )

        text = output.getvalue()
        self.assertEqual(text.count("Используя числа 3, 3, 8 и 8"), 1)
        self.assertIn("МЕТАДАННЫЕ: finish_reason=stop; tokens:", text)
        self.assertIn("ПРОВЕРКА: PASS", text)
        self.assertNotIn("АВТОМАТИЧЕСКАЯ ПРОВЕРКА", text)
        self.assertNotIn("Планируемых API-вызовов", text)


class PersistenceAndCliTests(unittest.TestCase):
    """Проверяет локальное сохранение, сравнение и защиту run-all."""

    def test_main_help_mentions_llm_comparison(self) -> None:
        output = StringIO()

        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            parse_arguments(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("compare --with-llm", output.getvalue())
        self.assertIn("дополнительный API-вызов", output.getvalue())

    def test_run_all_requires_explicit_confirmation(self) -> None:
        errors = StringIO()

        with redirect_stdout(StringIO()), patch("sys.stderr", errors):
            exit_code = main(["run-all"])

        self.assertEqual(exit_code, 2)
        self.assertIn("5 внешних API-вызовов", errors.getvalue())

    def test_saved_results_can_be_compared_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            with patch("main.RESULTS_DIR", results_dir):
                for method in (
                    METHOD_DIRECT,
                    METHOD_STEP_BY_STEP,
                    METHOD_GENERATED_PROMPT,
                    METHOD_EXPERTS,
                ):
                    answer = "FINAL: 8 / (3 - 8 / 3)"
                    if method == METHOD_EXPERTS:
                        answer = (
                            "ANALYST: решение\nENGINEER: проверка\n"
                            "CRITIC: ошибок нет\nFINAL: 8 / (3 - 8 / 3)"
                        )

                    def requester(prompt: str, api_key: str, final_answer: str = answer) -> CompletionResult:
                        if "Составь эффективный промпт" in prompt:
                            return completion("Реши задачу.\nFINAL: <выражение>")
                        return completion(final_answer)

                    with redirect_stdout(StringIO()):
                        result = execute_method(method, "test-key", requester)
                    save_method_result(result)

                output = StringIO()
                with redirect_stdout(output):
                    passed = compare_saved_results()

                self.assertTrue(passed)
                self.assertIn("СРАВНЕНИЕ ЧЕТЫРЁХ СПОСОБОВ", output.getvalue())
                self.assertEqual(len(list(results_dir.glob("*.json"))), 4)
                payload = json.loads((results_dir / "generated-prompt.json").read_text())
                self.assertEqual(len(payload["calls"]), 2)

    def test_llm_comparison_receives_full_answers_without_reprinting_them(self) -> None:
        payloads = []
        for method in (
            METHOD_DIRECT,
            METHOD_STEP_BY_STEP,
            METHOD_GENERATED_PROMPT,
            METHOD_EXPERTS,
        ):
            answer = f"Полный уникальный ответ {method}\nFINAL: 8 / (3 - 8 / 3)"
            if method == METHOD_EXPERTS:
                answer = (
                    "ANALYST: решение\nENGINEER: проверка\nCRITIC: ошибок нет\n"
                    f"Полный уникальный ответ {method}\nFINAL: 8 / (3 - 8 / 3)"
                )

            def method_requester(
                prompt: str,
                api_key: str,
                final_answer: str = answer,
            ) -> CompletionResult:
                if "Составь эффективный промпт" in prompt:
                    return completion("Созданный полный промпт")
                return completion(final_answer)

            with redirect_stdout(StringIO()):
                method_result = execute_method(method, "test-key", method_requester)
            payloads.append(serialize_method_result(method_result))

        comparison_prompt = build_comparison_prompt(payloads)
        for method in (
            METHOD_DIRECT,
            METHOD_STEP_BY_STEP,
            METHOD_GENERATED_PROMPT,
            METHOD_EXPERTS,
        ):
            self.assertIn(f"Полный уникальный ответ {method}", comparison_prompt)

        received: dict[str, str | None] = {}

        def comparison_requester(
            prompt: str,
            api_key: str,
            system_prompt: str | None,
        ) -> CompletionResult:
            received["prompt"] = prompt
            received["system_prompt"] = system_prompt
            return completion("Точность: ничья.\nРазличия: разные.\nВывод: прямой экономнее.")

        with tempfile.TemporaryDirectory() as directory:
            with patch("main.RESULTS_DIR", Path(directory)):
                output = StringIO()
                with redirect_stdout(output):
                    run_llm_comparison(payloads, "test-key", comparison_requester)

                self.assertEqual(received["prompt"], comparison_prompt)
                self.assertEqual(received["system_prompt"], COMPARISON_SYSTEM_PROMPT.strip())
                self.assertNotIn("Полный уникальный ответ direct", output.getvalue())
                self.assertTrue((Path(directory) / "comparison.json").exists())

    def test_comparison_recomputes_saved_validation(self) -> None:
        with redirect_stdout(StringIO()):
            method_result = execute_method(
                METHOD_DIRECT,
                "test-key",
                lambda prompt, api_key: completion("8 / (3 - 8 / 3)"),
            )
        payload = serialize_method_result(method_result)
        payload["validation"] = {
            "passed": True,
            "expression": "8 / (3 - 8 / 3)",
            "checks": [],
        }

        comparison_prompt = build_comparison_prompt([payload])

        self.assertIn("ЛОКАЛЬНАЯ ПРОВЕРКА: FAIL; выражение=None", comparison_prompt)


if __name__ == "__main__":
    unittest.main()
