from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from agent import AgentConfig, AgentError, StageChatAgent
from task_state import Event, Phase, Status, TaskStore, TransitionController


CONFIG = AgentConfig(api_key="fake-key")


def response(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


class StageChatAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_planning_turns_never_advance_phase(self):
        replies = iter(
            (
                {
                    "reply": "Предлагаю первый план",
                    "steps": [
                        {"title": "Анализ", "instruction": "Изучить требования"},
                        {"title": "Результат", "instruction": "Подготовить ответ"},
                    ],
                },
                {
                    "reply": "Добавил отдельную проверку",
                    "steps": [
                        {"title": "Анализ", "instruction": "Изучить требования"},
                        {"title": "Результат", "instruction": "Подготовить ответ"},
                        {"title": "Проверка", "instruction": "Проверить критерии"},
                    ],
                },
            )
        )
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Подготовить материал")
            controller = TransitionController()
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = StageChatAgent(CONFIG, client, task, store, controller)
                await agent.respond("Составь план")
                self.assertEqual(task.phase, Phase.PLANNING)
                await agent.respond("Добавь отдельный шаг проверки")
                self.assertEqual(task.phase, Phase.PLANNING)
                self.assertEqual(task.active_run.turns, 2)
                self.assertEqual(task.active_run.input_tokens, 20)
                self.assertEqual(task.active_run.output_tokens, 10)
                self.assertEqual(task.active_run.total_tokens, 30)
                self.assertEqual(task.active_run.last_prompt_tokens, 10)
                self.assertEqual(len(task.active_run.last_request_messages), 4)
                self.assertIn("Добавил отдельную проверку", task.active_run.last_response_raw)
                self.assertEqual(task.active_run.last_finish_reason, "stop")
                self.assertEqual(len(task.plan), 3)
                controller.apply(task, Event.PLAN_APPROVED)
                store.save(task)
            self.assertEqual(task.phase, Phase.EXECUTION)
            second_messages = bodies[1]["messages"]
            self.assertTrue(any(item["content"] == "Составь план" for item in second_messages))
            self.assertTrue(any(item["content"] == "Предлагаю первый план" for item in second_messages))
            self.assertIn("Ты не можешь менять стадию", bodies[0]["messages"][0]["content"])
            self.assertNotIn("max_tokens", bodies[0])
            self.assertEqual(bodies[0]["reasoning_effort"], "low")

    async def test_every_stage_keeps_own_conversation_and_needs_user_transition(self):
        replies = iter(
            (
                {
                    "reply": "План готов",
                    "steps": [
                        {"title": "Анализ", "instruction": "Проверить ввод"},
                        {"title": "Ответ", "instruction": "Подготовить ответ"},
                    ],
                },
                {"reply": "Результат подготовлен", "artifact": "Полный результат"},
                {"reply": "Проверка пройдена", "summary": "Нарушений нет", "issues": []},
            )
        )

        def handler(_: httpx.Request) -> httpx.Response:
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель")
            controller = TransitionController()
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = StageChatAgent(CONFIG, client, task, store, controller)
                await agent.respond("Предложи план")
                controller.apply(task, Event.PLAN_APPROVED)
                store.save(task)
                await agent.respond("Выполни утверждённый план")
                self.assertEqual(task.phase, Phase.EXECUTION)
                controller.apply(task, Event.EXECUTION_SUBMITTED)
                store.save(task)
                await agent.respond("Проверь результат")
                self.assertEqual(task.phase, Phase.VALIDATION)
                self.assertTrue(task.validation_passed)
                controller.apply(task, Event.VALIDATION_ACCEPTED)
                store.save(task)
            self.assertEqual((task.phase, task.status), (Phase.DONE, Status.DONE))
            self.assertEqual([(run.phase, run.visit, run.turns) for run in task.runs], [
                (Phase.PLANNING, 1, 1),
                (Phase.EXECUTION, 1, 1),
                (Phase.VALIDATION, 1, 1),
            ])

    async def test_failed_validation_waits_for_user_before_revision(self):
        replies = iter(
            (
                {
                    "reply": "План готов",
                    "steps": [
                        {"title": "Основа", "instruction": "Сделать основу"},
                        {"title": "Проверка", "instruction": "Проверить итог"},
                    ],
                },
                {"reply": "Черновик готов", "artifact": "Черновик"},
                {"reply": "Нашёл проблему", "summary": "Нужна доработка", "issues": ["Нет примера"]},
                {"reply": "Добавил пример", "artifact": "Черновик с примером"},
                {"reply": "Исправление прошло проверку", "summary": "Нарушений нет", "issues": []},
            )
        )

        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return response(next(replies))

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель")
            controller = TransitionController()
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = StageChatAgent(CONFIG, client, task, store, controller)
                await agent.respond("Составь план")
                controller.apply(task, Event.PLAN_APPROVED)
                await agent.respond("Выполни")
                controller.apply(task, Event.EXECUTION_SUBMITTED)
                await agent.respond("Проверь")
                self.assertEqual(task.phase, Phase.VALIDATION)
                self.assertFalse(task.validation_passed)
                controller.apply(task, Event.VALIDATION_SENT_TO_REVISION)
                await agent.respond("Исправь замечания")
                self.assertEqual(task.phase, Phase.REVISION)
                controller.apply(task, Event.REVISION_SUBMITTED)
                await agent.respond("Проверь исправленное")
            self.assertEqual(task.phase, Phase.VALIDATION)
            self.assertEqual(task.revision_count, 1)
            validation_runs = [run for run in task.runs if run.phase == Phase.VALIDATION]
            self.assertEqual([(run.visit, run.turns) for run in validation_runs], [(1, 1), (2, 1)])
            self.assertIn("Нужна доработка", validation_runs[0].artifact)
            repeated_validation_messages = bodies[4]["messages"]
            self.assertTrue(all(item["content"] != "Проверь" for item in repeated_validation_messages))
            self.assertTrue(all(item["content"] != "Нашёл проблему" for item in repeated_validation_messages))
            self.assertIn("Черновик с примером", repeated_validation_messages[0]["content"])

    async def test_length_failure_preserves_partial_exchange_and_usage(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{\"reply\":"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 4000, "total_tokens": 4120},
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory))
            task = store.create("task-01", "Цель")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                agent = StageChatAgent(CONFIG, client, task, store)
                with self.assertRaisesRegex(AgentError, "обрезан лимитом"):
                    await agent.respond("Составь подробный план")

            self.assertEqual(task.status, Status.FAILED)
            self.assertEqual(task.active_run.turns, 0)
            self.assertEqual(task.active_run.last_response_raw, '{"reply":')
            self.assertEqual(task.active_run.last_finish_reason, "length")
            self.assertEqual(task.active_run.last_prompt_tokens, 120)
            self.assertEqual(task.active_run.total_tokens, 4120)


if __name__ == "__main__":
    unittest.main()
