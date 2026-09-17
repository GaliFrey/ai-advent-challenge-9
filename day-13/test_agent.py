"""Проверки prompts, переходов и транзакционности оркестратора."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from agent import AgentConfig, AgentError, WorkflowAgent
from task_state import TaskStore


CONFIG = AgentConfig(api_key="fake-key", openrouter_api_key="fake-openrouter-key")
PLAN = json.dumps(
    {
        "steps": [
            {"title": "Основа", "instruction": "Подготовить основу"},
            {"title": "Пример", "instruction": "Добавить пример"},
            {"title": "Проверка", "instruction": "Добавить проверку"},
        ]
    },
    ensure_ascii=False,
)
VALIDATION = json.dumps(
    {
        "summary": "Все три шага выполнены",
        "issues": [],
        "improvements": [{"area": "Подача", "suggestion": "Можно добавить короткий вывод"}],
        "revision_instruction": "",
        "final_result": "Готовый цельный результат",
    },
    ensure_ascii=False,
)
FAILED_VALIDATION = json.dumps(
    {
        "summary": "Нужна доработка",
        "issues": [
            {
                "criterion": "Конкретный пример",
                "evidence": "В результате приведено только общее описание",
                "problem": "Не хватает конкретного примера",
                "required_change": "Добавить запускаемый пример",
            }
        ],
        "improvements": [{"area": "Структура", "suggestion": "Добавить подзаголовок"}],
        "revision_instruction": "Добавь конкретный пример и сохрани остальной результат",
        "final_result": "Предварительный результат",
    },
    ensure_ascii=False,
)
SECOND_FAILED_VALIDATION = json.dumps(
    {
        "summary": "После доработки осталось замечание",
        "issues": [
            {
                "criterion": "Конкретный пример",
                "evidence": "Пример не содержит ожидаемый вывод",
                "problem": "Пример всё ещё недостаточно конкретен",
                "required_change": "Добавить ожидаемый вывод",
            }
        ],
        "improvements": [{"area": "Подача", "suggestion": "Сделать пояснение короче"}],
        "revision_instruction": "Нужна ручная проверка",
        "final_result": "Исправленный, но не прошедший проверку результат",
    },
    ensure_ascii=False,
)


def response(text: str, *, status: int = 200, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        },
    )


class WorkflowAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_machine_uses_stage_profiles_and_reaches_done(self):
        prompts: list[list[dict[str, str]]] = []
        request_bodies: list[dict[str, object]] = []
        request_urls: list[str] = []
        replies = iter((PLAN, "Часть 1", "Часть 2", "Часть 3", VALIDATION))

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            request_bodies.append(body)
            prompts.append(body["messages"])
            request_urls.append(str(request.url))
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Подготовить материал", "explainer")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = WorkflowAgent(CONFIG, client, task, store)
                for _ in range(5):
                    await agent.advance()

            self.assertEqual(task.phase, "done")
            self.assertEqual(task.status, "done")
            self.assertEqual(task.final_result, "Готовый цельный результат")
            self.assertEqual(agent.total_tokens, 125)
            self.assertEqual(len(prompts), 5)
            self.assertIn("Объясняющий материал", prompts[0][0]["content"])
            self.assertIn("Роль стадии planning", prompts[0][0]["content"])
            self.assertIn("Роль стадии execution", prompts[1][0]["content"])
            self.assertIn("Роль стадии validation", prompts[-1][0]["content"])
            self.assertEqual(request_bodies[0]["response_format"], {"type": "json_object"})
            self.assertNotIn("response_format", request_bodies[1])
            self.assertNotIn("response_format", request_bodies[2])
            self.assertNotIn("response_format", request_bodies[3])
            self.assertEqual(request_bodies[0]["model"], "deepseek-v4-flash")
            self.assertEqual(request_bodies[4]["model"], "openai/gpt-5.6-sol")
            self.assertEqual(request_bodies[4]["response_format"]["type"], "json_schema")
            self.assertTrue(request_bodies[4]["response_format"]["json_schema"]["strict"])
            self.assertEqual(request_bodies[4]["provider"], {"require_parameters": True})
            self.assertEqual(request_bodies[4]["max_tokens"], 4000)
            self.assertEqual(request_bodies[4]["reasoning"], {"effort": "low", "exclude": True})
            self.assertNotIn("thinking", request_bodies[4])
            self.assertNotIn("temperature", request_bodies[4])
            self.assertEqual(request_bodies[0]["temperature"], 0.0)
            self.assertEqual(request_bodies[0]["max_tokens"], 1800)
            self.assertEqual(request_urls[:4], ["https://api.deepseek.com/chat/completions"] * 4)
            self.assertEqual(request_urls[4], "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(task.validation_improvements, ["Подача: Можно добавить короткий вывод"])
            summary = store.run_log.summary("task-01")
            self.assertEqual(summary["events"], 6)
            self.assertEqual(summary["calls"], 5)
            self.assertEqual(summary["usage"]["total_tokens"], 125)
            self.assertEqual(summary["by_phase"]["execution"]["calls"], 3)

    async def test_pause_after_turn_and_restore_does_not_repeat_completed_step(self):
        replies = iter((PLAN, "Первый результат", "Второй результат"))
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body["messages"][-1]["content"])
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = TaskStore(root)
            task = store.create("task-01", "Цель без повторного объяснения", "engineering")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                first = WorkflowAgent(CONFIG, client, task, store)
                await first.advance()
                first.request_pause()
                await first.advance()
                self.assertEqual(task.status, "paused")
                self.assertEqual(task.current_step, 1)

                restored = store.load("task-01")
                store.resume(restored)
                second = WorkflowAgent(CONFIG, client, restored, store)
                await second.advance()

            self.assertEqual(restored.current_step, 2)
            self.assertEqual(restored.steps[0].result, "Первый результат")
            self.assertEqual(restored.steps[1].result, "Второй результат")
            self.assertIn("Цель без повторного объяснения", calls[-1])
            self.assertIn("Первый результат", calls[-1])
            self.assertIn("Текущий шаг: Пример", calls[-1])

    async def test_invalid_response_keeps_checkpoint_and_allows_retry(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return response("не JSON" if calls == 1 else PLAN)

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель", "engineering")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = WorkflowAgent(CONFIG, client, task, store)
                with self.assertRaises(AgentError):
                    await agent.advance()
                self.assertEqual((task.phase, task.current_step, task.status), ("planning", 0, "failed"))
                self.assertEqual(task.steps, [])
                self.assertEqual(agent.total_tokens, 25)
                self.assertEqual(task.last_request_messages[0]["role"], "system")
                await agent.advance()

            self.assertEqual(task.phase, "execution")
            self.assertEqual(task.current_step, 0)
            self.assertEqual(task.status, "ready")
            summary = store.run_log.summary("task-01")
            self.assertEqual(summary["calls"], 2)
            self.assertEqual(summary["failures"], 1)
            self.assertEqual(summary["usage"]["total_tokens"], 50)
            failed = store.run_log.read("task-01")[1]
            self.assertEqual(failed["event"], "stage_failed")
            self.assertEqual(failed["response"], "не JSON")
            self.assertTrue(failed["error"])

    async def test_different_profiles_change_stage_system_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            engineering = store.create("task-01", "Одна цель", "engineering")
            explainer = store.create("task-02", "Одна цель", "explainer")
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response(PLAN))) as client:
                first = WorkflowAgent(CONFIG, client, engineering, store).build_messages()
                second = WorkflowAgent(CONFIG, client, explainer, store).build_messages()
            self.assertNotEqual(first[0], second[0])
            self.assertIn("технический планировщик", first[0]["content"])
            self.assertIn("редактор учебных материалов", second[0]["content"])

    async def test_length_response_preserves_partial_text_usage_and_checkpoint(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return response("Частичный результат", finish_reason="length")

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель", "engineering")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = WorkflowAgent(CONFIG, client, task, store)
                with self.assertRaisesRegex(AgentError, "length"):
                    await agent.advance()

            self.assertEqual((task.phase, task.status, task.current_step), ("planning", "failed", 0))
            self.assertEqual(agent.total_tokens, 25)
            failed = store.run_log.read("task-01")[-1]
            self.assertEqual(failed["response"], "Частичный результат")
            self.assertEqual(failed["usage"]["total_tokens"], 25)
            self.assertEqual(failed["error"], "Ответ не завершён: length")
            self.assertEqual(failed["details"]["response_format"], "json_object")
            self.assertEqual(failed["details"]["finish_reason"], "length")

    async def test_http_error_preserves_safe_provider_message(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": {"message": "No endpoints found for this model"}})

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель", "engineering")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = WorkflowAgent(CONFIG, client, task, store)
                with self.assertRaisesRegex(AgentError, "No endpoints found"):
                    await agent.advance()

            failed = store.run_log.read("task-01")[-1]
            self.assertEqual(failed["error"], "API вернул HTTP 404: No endpoints found for this model")
            self.assertEqual((task.phase, task.status), ("planning", "failed"))

    async def test_openrouter_compatible_content_blocks_and_usage_names(self):
        replies = iter((PLAN, "Часть 1", "Часть 2", "Часть 3"))

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body["model"] != "openai/gpt-5.6-sol":
                return response(next(replies))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": [{"type": "output_text", "text": VALIDATION}]},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель", "engineering")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = WorkflowAgent(CONFIG, client, task, store)
                for _ in range(5):
                    await agent.advance()

            self.assertEqual((task.phase, task.status), ("done", "done"))
            self.assertEqual(agent.total_tokens, 125)

    async def test_failed_validation_runs_one_clean_revision_then_passes(self):
        prompts: list[list[dict[str, str]]] = []
        request_bodies: list[dict[str, object]] = []
        replies = iter((PLAN, "Часть 1", "Часть 2", "Часть 3", FAILED_VALIDATION, "Исправленный итог", VALIDATION))

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            request_bodies.append(body)
            prompts.append(body["messages"])
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель с quality gate", "engineering")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                first_agent = WorkflowAgent(CONFIG, client, task, store)
                for _ in range(5):
                    await first_agent.advance()
                self.assertEqual(task.phase, "revision")

                task = store.load("task-01")
                second_agent = WorkflowAgent(CONFIG, client, task, store)
                for _ in range(2):
                    await second_agent.advance()

            self.assertEqual((task.phase, task.status), ("done", "done"))
            self.assertTrue(task.validation_passed)
            self.assertEqual(task.revision_count, 1)
            self.assertEqual(task.revision_result, "Исправленный итог")
            self.assertEqual(len(prompts), 7)
            self.assertIn("Не хватает конкретного примера", prompts[5][-1]["content"])
            self.assertEqual(len(prompts[5]), 2)
            self.assertIn("Исправленный итог", prompts[6][-1]["content"])
            self.assertEqual(request_bodies[4]["response_format"]["type"], "json_schema")
            self.assertNotIn("response_format", request_bodies[5])
            self.assertEqual(request_bodies[5]["model"], "deepseek-v4-pro")
            self.assertEqual(request_bodies[6]["response_format"]["type"], "json_schema")
            self.assertEqual(request_bodies[6]["model"], "openai/gpt-5.6-sol")
            events = store.run_log.read("task-01")
            self.assertEqual([item["event"] for item in events].count("stage_completed"), 7)
            self.assertEqual(events[-2]["phase_before"], "revision")
            self.assertEqual(events[-1]["phase_before"], "validation")

    async def test_two_failed_revisions_then_validation_stops_with_issues(self):
        replies = iter(
            (
                PLAN,
                "Часть 1",
                "Часть 2",
                "Часть 3",
                FAILED_VALIDATION,
                "Исправленный итог 1",
                SECOND_FAILED_VALIDATION,
                "Исправленный итог 2",
                SECOND_FAILED_VALIDATION,
            )
        )

        def handler(_: httpx.Request) -> httpx.Response:
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель", "explainer")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = WorkflowAgent(CONFIG, client, task, store)
                for _ in range(9):
                    await agent.advance()

            self.assertEqual((task.phase, task.status), ("done", "done_with_issues"))
            self.assertFalse(task.validation_passed)
            self.assertEqual(task.revision_count, 2)
            self.assertEqual(task.revision_result, "Исправленный итог 2")
            self.assertEqual(len(task.validation_issues), 1)
            self.assertIn("Пример всё ещё недостаточно конкретен", task.validation_issues[0])
            self.assertEqual(task.final_result, "Исправленный, но не прошедший проверку результат")
            events = store.run_log.read("task-01")
            self.assertEqual([item["phase_before"] for item in events[-4:]], ["revision", "validation", "revision", "validation"])
