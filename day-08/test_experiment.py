"""Проверки полного эксперимента через MockTransport."""

from __future__ import annotations

import json
import io
import re
import tempfile
import unittest
from pathlib import Path

import httpx
from rich.console import Console

from agent import Agent, AgentConfig
from experiment import (
    CONTROL_FACT,
    ErrorEvent,
    RequestEvent,
    ResponseEvent,
    padding_to_target,
    run_experiment,
    save_report,
)
from main import (
    analysis_lines,
    context_table,
    economics_table,
    metric_legend,
    overflow_consequences,
    prompt_excerpt,
)
from token_counter import TokenCounter


CONFIG = AgentConfig(api_key="fake-key")


class PaddingTests(unittest.TestCase):
    def test_padding_hits_both_sides_of_target(self):
        counter = TokenCounter(CONFIG.model)
        history = ({"role": "system", "content": CONFIG.system_prompt},)
        below = padding_to_target(
            counter,
            history,
            target_prompt_tokens=600,
            step=1,
            expected="OK",
        )
        above = padding_to_target(
            counter,
            history,
            target_prompt_tokens=600,
            step=1,
            expected="OK",
            at_least=True,
        )
        below_count = counter.count_messages([*history, {"role": "user", "content": below}])
        above_count = counter.count_messages([*history, {"role": "user", "content": above}])
        self.assertLess(below_count, 600)
        self.assertGreaterEqual(above_count, 600)
        self.assertLess(600 - below_count, 20)
        self.assertLess(above_count - 600, 20)

        with_fact = padding_to_target(
            counter,
            history,
            target_prompt_tokens=1_000,
            step=2,
            expected="OK-2",
            include_fact=True,
        )
        excerpt = prompt_excerpt(with_fact)
        self.assertIn("record_2_0000", excerpt)
        self.assertIn(CONTROL_FACT, excerpt)
        self.assertIn("пропущено строк", excerpt)
        self.assertIn("Ответь ровно: OK-2", excerpt)


class ExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_experiment_reaches_realistic_context_error(self):
        counter = TokenCounter(CONFIG.model)
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            prompt_tokens = counter.count_messages(payload["messages"])
            if prompt_tokens + payload["max_tokens"] > CONFIG.context_limit:
                return httpx.Response(400, json={
                    "error": {
                        "message": (
                            "This model's maximum context length is 16385 tokens. "
                            f"The request contains {prompt_tokens} prompt tokens."
                        ),
                        "code": "context_length_exceeded",
                    }
                })

            last = payload["messages"][-1]["content"]
            if "Какое значение было указано" in last:
                answer = CONTROL_FACT
            else:
                match = re.search(r"Ответь ровно: ([A-Z0-9-]+)$", last)
                if match is None:
                    raise AssertionError("Тестовый запрос не содержит ожидаемый ответ")
                answer = match.group(1)
            completion_tokens = counter.count_text(answer)
            cost = (prompt_tokens * 3 + completion_tokens * 4) / 1_000_000
            return httpx.Response(200, json={
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "cost": cost,
                },
                "provider": "OpenAI",
                "openrouter_metadata": {"pipeline": []},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            agent = Agent(CONFIG, client, counter)
            events = []
            result = await run_experiment(agent, counter, events.append)

        self.assertTrue(result.passed)
        self.assertTrue(result.fact_recalled)
        self.assertTrue(result.overflow_confirmed)
        self.assertTrue(result.history_unchanged_after_error)
        self.assertEqual(len(result.rows), 7)
        self.assertEqual(len(requests), 7)
        self.assertEqual(requests[-1]["plugins"][0]["enabled"], False)
        self.assertGreater(result.rows[-1].prompt_tokens_local, CONFIG.context_limit)
        self.assertEqual(result.rows[-1].status, "ЛИМИТ")
        self.assertEqual(len(agent.messages), 13)
        self.assertEqual(len(events), 14)
        self.assertEqual(sum(isinstance(event, RequestEvent) for event in events), 7)
        self.assertEqual(sum(isinstance(event, ResponseEvent) for event in events), 6)
        self.assertEqual(sum(isinstance(event, ErrorEvent) for event in events), 1)
        self.assertIn("record_", next(
            event.user_message
            for event in events
            if isinstance(event, RequestEvent) and event.scenario == "длинный"
        ))
        self.assertEqual(
            [event.answer for event in events if isinstance(event, ResponseEvent)][0],
            "OK-1",
        )

        successful_prompt_sizes = [
            row.prompt_tokens_api for row in result.rows if row.prompt_tokens_api is not None
        ]
        self.assertEqual(successful_prompt_sizes, sorted(successful_prompt_sizes))
        self.assertGreater(successful_prompt_sizes[-1], successful_prompt_sizes[0] * 10)
        self.assertGreater(result.rows[-2].cumulative_cost_usd, result.rows[0].cost_usd or 0)

        lines = analysis_lines(result)
        self.assertEqual(len(lines), 6)
        self.assertIn("Prompt вырос", lines[0])
        self.assertIn("история занимала", lines[2])
        self.assertIn("превысил окно", lines[-1])
        legend = metric_legend()
        self.assertIn("Текущий запрос", legend[0])
        self.assertIn("Вся история", legend[1])
        consequences = overflow_consequences(result)
        self.assertIn("не вернула ответ", consequences[0])
        self.assertIn("более короткий запрос", consequences[-1])
        console = Console(record=True, width=160, file=io.StringIO())
        console.print(context_table(result))
        console.print(economics_table(result))
        rendered = console.export_text()
        self.assertIn(CONFIG.model, rendered)
        self.assertIn("Текущий", rendered)
        self.assertIn("Вся", rendered)
        self.assertIn("Доля", rendered)
        self.assertIn("Свободно", rendered)
        self.assertIn("К первому", rendered)
        self.assertIn("Σ вход", rendered)

        with tempfile.TemporaryDirectory() as directory:
            path = save_report(result, Path(directory))
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(saved["passed"])
        self.assertEqual(len(saved["rows"]), 7)
        self.assertIsNone(saved["report_path"])


if __name__ == "__main__":
    unittest.main()
