import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import httpx
from textual.widgets import Button, Static, TabbedContent, TextArea, DataTable

from experiment import Model, Settings, cost_estimate, load_session, payload, run_experiment, usage_metrics, all_records, aggregate_metrics, ROOT, load_settings
from main import CompareApp, clean_text

MODELS = (Model("weak", "openrouter", "qwen/qwen3-4b:free"), Model("medium", "deepseek", "deepseek-v4-flash"), Model("strong", "deepseek", "deepseek-v4-pro"))
PROMPTS = ["Задача", "Другая задача"]
SETTINGS = Settings(MODELS, "fake-deepseek-key", "fake-openrouter-key")


def response(answer="Тестовый ответ", finish="stop", usage=True):
    result = {"choices": [{"message": {"content": answer}, "finish_reason": finish}]}
    if usage:
        result["usage"] = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "prompt_cache_hit_tokens": 20}
    return httpx.Response(200, json=result)


class MetricsTests(unittest.TestCase):
    def test_missing_usage_is_not_zero(self):
        metrics = usage_metrics({})
        self.assertIsNone(metrics["total_tokens"])
        self.assertIsNone(cost_estimate(MODELS[1], metrics, "2026-09-04T12:00:00+00:00", "free")["usd"])

    def test_cached_input_and_peak(self):
        metrics = {"input_tokens": 100, "output_tokens": 50, "cached_tokens": 20}
        low = cost_estimate(MODELS[1], metrics, "2026-09-04T12:00:00+00:00", "free")
        high = cost_estimate(MODELS[1], metrics, "2026-09-04T06:00:00+00:00", "free")
        weekend = cost_estimate(MODELS[1], metrics, "2026-09-05T06:00:00+00:00", "free")
        self.assertAlmostEqual(low["usd"], (20 * .007 + 80 * .22 + 50 * .66) / 1e6)
        self.assertAlmostEqual(high["usd"], low["usd"] * 2)
        self.assertEqual(weekend["usd"], low["usd"])
        self.assertEqual(cost_estimate(MODELS[1], metrics, "2026-09-04T10:00:00+00:00", "free")["usd"], low["usd"])

    def test_free_paid_unknown_models(self):
        metrics = {"input_tokens": 100, "output_tokens": 50, "cached_tokens": None}
        self.assertEqual(cost_estimate(MODELS[0], metrics, "2026-09-04T12:00:00+00:00", "free")["usd"], 0)
        self.assertAlmostEqual(cost_estimate(Model("weak", "groq", "qwen/qwen3.8-27b"), metrics, "2026-09-04T12:00:00+00:00", "paid")["usd"], .00028)
        self.assertIsNone(cost_estimate(Model("weak", "groq", "unknown"), metrics, "2026-09-04T12:00:00+00:00", "paid")["usd"])

    def test_pro_cached_rate_is_not_three_times_flash(self):
        metrics = {"input_tokens": 1_000_000, "output_tokens": 0, "cached_tokens": 1_000_000}
        self.assertEqual(cost_estimate(MODELS[2], metrics, "2026-09-04T12:00:00+00:00", "free")["usd"], .022)
        self.assertEqual(cost_estimate(MODELS[2], metrics, "2026-09-04T06:00:00+00:00", "free")["usd"], .044)

    def test_bad_usage_and_cache_fallback(self):
        self.assertIsNone(usage_metrics({"usage": {"prompt_tokens": True}})["input_tokens"])
        self.assertEqual(usage_metrics({"usage": {"prompt_tokens": 100, "completion_tokens": 2, "prompt_cache_miss_tokens": 80}}), {"input_tokens": 100, "output_tokens": 2, "total_tokens": 102, "cached_tokens": 20})
        self.assertIsNone(usage_metrics({"usage": {"prompt_tokens": 1, "prompt_cache_hit_tokens": 100}})["cached_tokens"])

    def test_provider_modes_are_explicit(self):
        messages = [{"role": "user", "content": "Общий запрос"}]
        ds = payload(MODELS[1], messages)
        groq = payload(MODELS[0], messages)
        self.assertEqual(ds["messages"], groq["messages"])
        self.assertEqual(ds["thinking"], {"type": "disabled"})
        self.assertEqual(groq["reasoning"], {"enabled": False})
        self.assertEqual(groq["provider"], {"require_parameters": True, "max_price": {"prompt": 0, "completion": 0}})
        self.assertNotIn("thinking", groq)
        self.assertEqual(ds["max_tokens"], groq["max_tokens"])
        self.assertTrue(all(payload(model, messages)["max_tokens"] == 4096 for model in MODELS))

    def test_openrouter_configuration_needs_no_groq_key_and_rejects_paid_models(self):
        with patch("experiment.load_dotenv"), patch.dict("os.environ", {
            "DEEPSEEK_API_KEY": "fake-deepseek-key", "OPENROUTER_API_KEY": "fake-openrouter-key",
            "OPENROUTER_MODEL_WEAK": "nvidia/nemotron-3.5-lightning:free",
        }, clear=True):
            settings = load_settings()
            settings.check()
            self.assertEqual(settings.models[0].provider, "openrouter")
            self.assertEqual(settings.models[0].name, "nvidia/nemotron-3.5-lightning:free")
            self.assertNotIn("fake-openrouter-key", repr(settings))
        for name in ("nvidia/paid-model", "openrouter/free"):
            with self.assertRaisesRegex(ValueError, ":free"):
                Settings((Model("weak", "openrouter", name), *MODELS[1:]), "fake", "fake").check()

    def test_terminal_controls_removed(self):
        self.assertEqual(clean_text("\x1b[31mОтвет\x1b[0m\x07\nстрока"), "Ответ\nстрока")


class ExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_six_sequential_requests_then_local_check(self):
        started = []
        completed = []
        calls = []
        snapshots = []

        async def handler(request):
            data = json.loads(request.content)
            calls.append(data)
            if data["model"] == MODELS[0].name:
                self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
                self.assertEqual(request.headers["Authorization"], "Bearer fake-openrouter-key")
            else:
                self.assertEqual(request.url.host, "api.deepseek.com")
                self.assertEqual(request.headers["Authorization"], "Bearer fake-deepseek-key")
            self.assertEqual(len(started), len(completed), "Requests must not overlap")
            started.append(data["model"])
            await asyncio.sleep(0)
            self.assertEqual(len(started), len(completed) + 1)
            completed.append(data["model"])
            return response("Ответ без указания модели")

        def on_change(session, path):
            saved = json.loads(path.read_text())
            self.assertEqual(saved, session)
            snapshots.append((session["status"], len(all_records(session))))

        session, path = await run_experiment(SETTINGS, PROMPTS, on_change, results_dir=self.directory, transport=httpx.MockTransport(handler))
        self.assertEqual(len(calls), 6)
        self.assertEqual([c["model"] for c in calls], [m.name for m in MODELS] * 2)
        self.assertEqual(session["execution_mode"], "sequential")
        self.assertIsNone(session["active_request"])
        self.assertEqual(session["status"], "complete")
        self.assertEqual(load_session(path), session)
        self.assertIn(("checking", 6), snapshots)
        for start in (0, 3):
            self.assertEqual(len({json.dumps(c["messages"]) for c in calls[start:start + 3]}), 1)
        self.assertNotEqual(calls[0]["messages"], calls[3]["messages"])
        self.assertNotIn("analysis", session)
        self.assertNotIn("blind_mapping", session)
        self.assertNotIn("fake-deepseek-key", path.read_text())
        self.assertNotIn("fake-openrouter-key", path.read_text())
        self.assertNotIn("Authorization", path.read_text())

    async def test_partial_failure_no_retry(self):
        calls = []
        async def handler(request):
            data = json.loads(request.content)
            calls.append(data)
            if data["model"] == MODELS[0].name:
                return httpx.Response(429, text="fake-openrouter-key must not leak", headers={
                    "retry-after": "12.5", "x-ratelimit-limit-tokens": "8000",
                    "x-ratelimit-remaining-tokens": "0", "x-ratelimit-remaining-requests": "fake-openrouter-key",
                    "Authorization": "fake-deepseek-key",
                })
            if len(data["messages"][0]["content"]) < 200:
                return response("Частичный ответ", "length")
            return response("Сравнение неполное")
        session, path = await run_experiment(SETTINGS, PROMPTS, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        self.assertEqual(len(calls), 6)
        self.assertEqual(session["status"], "partial")
        self.assertEqual(session["results"]["task-1"]["weak"]["status"], "error")
        self.assertIsNone(session["results"]["task-1"]["weak"]["cost"]["usd"])
        error = session["results"]["task-1"]["weak"]
        self.assertIn("превышен лимит", error["error"])
        self.assertIn("12.5 с", error["error"])
        self.assertEqual(error["rate_limit"], {"retry-after": "12.5", "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "0"})
        self.assertEqual(session["results"]["task-1"]["medium"]["status"], "incomplete")
        self.assertNotIn("analysis", session)
        self.assertEqual(session["comparison_note"], "incomplete_answers")
        self.assertNotIn("fake-openrouter-key", path.read_text())

    async def test_output_limit_reason_excludes_private_error_text(self):
        async def handler(request):
            data = json.loads(request.content)
            if data["model"] == MODELS[0].name:
                return httpx.Response(429, json={"error": {"message":
                    "Request too large for model qwen in organization org_private "
                    "on output tokens per minute (OTPM): Limit 1000, Requested 1237. "
                    "fake-openrouter-key fake-deepseek-key"}})
            return response()
        session, path = await run_experiment(SETTINGS, PROMPTS, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        record = session["results"]["task-2"]["weak"]
        self.assertEqual(record["rate_limit"], {"kind": "OTPM", "limit": 1000, "requested": 1237, "request_too_large": True})
        self.assertIn("Уменьши max_tokens", record["error"])
        self.assertIn("1237", record["error"])
        self.assertEqual(session["status"], "partial")
        for secret in ("org_private", "fake-openrouter-key", "fake-deepseek-key"):
            self.assertNotIn(secret, path.read_text())

    async def test_all_fail_still_makes_only_six_requests(self):
        calls = []
        async def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("fake-deepseek-key", request=request)
        session, path = await run_experiment(SETTINGS, PROMPTS, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        self.assertEqual(len(calls), 6)
        self.assertEqual(session["status"], "failed")
        self.assertNotIn("analysis", session)
        self.assertNotIn("fake-deepseek-key", path.read_text())

    async def test_no_seventh_request(self):
        calls = []
        async def handler(request):
            calls.append(request)
            return response()
        session, path = await run_experiment(SETTINGS, PROMPTS, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        self.assertEqual(session["status"], "complete")
        self.assertEqual(len(all_records(session)), 6)
        self.assertEqual(len(calls), 6)
        self.assertNotIn("analysis", session)
        self.assertEqual(load_session(path), session)

    async def test_preflight_and_storage_before_network(self):
        async def handler(request):
            self.fail("Must not call API")
        with self.assertRaises(ValueError):
            await run_experiment(Settings(MODELS), PROMPTS, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        with self.assertRaises(ValueError):
            await run_experiment(SETTINGS, "  ", lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        for prompts in (["one"], ["one", "  "], ["same", " same "], ["one", "two", "three"]):
            with self.assertRaises(ValueError):
                await run_experiment(SETTINGS, prompts, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        self.assertEqual(list(self.directory.iterdir()), [])
        bad = self.directory / "file"
        bad.write_text("not a directory")
        with self.assertRaises(OSError):
            await run_experiment(SETTINGS, PROMPTS, lambda *_: None, results_dir=bad, transport=httpx.MockTransport(handler))

    async def test_runs_never_overwrite(self):
        async def handler(request):
            return response()
        first, p1 = await run_experiment(SETTINGS, ["Первый", "Другой"], lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        content = p1.read_bytes()
        second, p2 = await run_experiment(SETTINGS, ["Второй", "Другой"], lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        self.assertNotEqual(p1, p2)
        self.assertEqual(p1.read_bytes(), content)

    async def test_malformed_responses(self):
        async def handler(request):
            return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 7}})
        session, _ = await run_experiment(SETTINGS, PROMPTS, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        self.assertEqual(session["status"], "failed")
        self.assertTrue(all(r["status"] == "error" for r in all_records(session)))

    async def test_tui_button_progress_conclusion_and_reopen_without_network(self):
        gate = asyncio.Event()
        called = []
        async def handler(request):
            called.append(request)
            if len(called) <= 6:
                await gate.wait()
            data = json.loads(request.content)
            return response("## Итог\n" + data["messages"][-1]["content"])
        app = CompareApp(SETTINGS, results_dir=self.directory, transport=httpx.MockTransport(handler))
        async with app.run_test(size=(110, 40)) as pilot:
            self.assertEqual(len(called), 0)
            for i, prompt in enumerate(PROMPTS, 1):
                app.query_one(f"#prompt-{i}", TextArea).load_text(prompt)
            await pilot.click("#start")
            await pilot.pause()
            self.assertEqual(len(called), 1)
            self.assertTrue(app.query_one("#start", Button).disabled)
            self.assertFalse(app.query_one("#prompts").display)
            self.assertTrue(all(editor.read_only for editor in app.query(TextArea)))
            table = app.query_one("#metrics", DataTable)
            self.assertIn("Запрос", table.get_row_at(0)[2].plain)
            self.assertTrue(all(table.get_row_at(i)[2].plain == "В очереди" for i in range(1, 6)))
            self.assertTrue(all(table.get_row_at(i)[3] == "—" for i in range(1, 6)))
            app.action_start()  # A second activation must not issue extra calls.
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(len(called), 6)
            self.assertEqual(app.session["status"], "complete")
            self.assertEqual([json.loads(r.content)["messages"][-1]["content"] for r in called[:6]], [PROMPTS[0]] * 3 + [PROMPTS[1]] * 3)
            self.assertEqual(app.query_one("#tabs", TabbedContent).active, "task-1")
            self.assertFalse(app.query_one("#start", Button).disabled)
            self.assertLessEqual(app.query_one("#saved").region.bottom, 39)
            self.assertGreaterEqual(app.query_one("#tabs").region.height, 12)
            for index, prompt in enumerate(PROMPTS, 1):
                app.query_one("#tabs", TabbedContent).active = f"task-{index}"
                self.assertIn(prompt, str(app.query_one(f"#compare-preview-{index}-weak", Static).content))
                self.assertIn("Готово", str(app.query_one(f"#compare-meta-{index}-weak", Static).content))
                for slot in ("weak", "medium", "strong"):
                    app.query_one(f"#answers-{index}", TabbedContent).active = f"tab-{index}-{slot}"
                    await pilot.pause()
                    widget = app.query_one(f"#answer-{index}-{slot}", Static)
                    self.assertIn(prompt, str(widget.render()))
                    self.assertGreater(widget.region.height, 0)
            app.query_one("#tabs", TabbedContent).active = "summary"
            await pilot.pause()
            self.assertGreater(app.query_one("#summary-text").region.height, 0)
            app.action_open_latest()
            await pilot.pause()
            self.assertEqual(len(called), 6)
            self.assertIn("ПРОСМОТР", str(app.query_one("#phase", Static).render()))

    async def test_summary_missing_usage_and_one_failed_task(self):
        async def handler(request):
            data = json.loads(request.content)
            if data["messages"][-1]["content"] == PROMPTS[1]:
                if data["model"] == MODELS[0].name:
                    return httpx.Response(429)
                return response(usage=False)
            return response()
        session, _ = await run_experiment(SETTINGS, PROMPTS, lambda *_: None, results_dir=self.directory, transport=httpx.MockTransport(handler))
        summary = aggregate_metrics(session)
        self.assertEqual(summary["weak"]["successful"], 1)
        self.assertIsNone(summary["weak"]["estimated_cost_usd"])
        self.assertIsNone(summary["medium"]["total_tokens"])
        self.assertIsNone(summary["medium"]["estimated_cost_usd"])
        self.assertEqual(summary["strong"]["completed"], 2)
        self.assertIsNotNone(summary["strong"]["sum_elapsed_seconds"])

    async def test_historical_session_opens_without_network_or_rewrite(self):
        # Historical recordings are local materials, not a test dependency.
        path = self.directory / "legacy.json"
        path.write_text(json.dumps({
            "schema_version": 1, "id": "legacy", "prompt": "Задача", "status": "complete",
            "price_date": "2026-09-04", "analysis": None,
            "models": [{"slot": m.slot, "provider": m.provider, "name": m.name} for m in MODELS],
            "blind_mapping": {"A": "weak", "B": "medium", "C": "strong"},
            "results": {m.slot: {"slot": m.slot, "provider": m.provider, "model": m.name,
                "status": "ok", "answer": "Сохранённый ответ", "finish_reason": "stop", "error": None,
                "elapsed_seconds": 1, "metrics": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "cost": {"usd": 0, "basis": "Тест"}} for m in MODELS},
        }))
        original = path.read_bytes()
        session = load_session(path)
        self.assertEqual(len(session["prompts"]), 1)
        self.assertEqual(len(all_records(session)), 3)
        def no_network(request):
            self.fail("Reading a session must not call API")
        app = CompareApp(Settings(MODELS), session_path=path, results_dir=self.directory, transport=httpx.MockTransport(no_network))
        async with app.run_test(size=(110, 40)) as pilot:
            self.assertEqual(app.session, session)
            self.assertIn("ПРОСМОТР", str(app.query_one("#phase", Static).render()))
            self.assertIn("Ответ не сохранён", str(app.query_one("#answer-2-strong", Static).render()))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [path])

    async def test_tui_incomplete_comparison_keeps_available_answers(self):
        calls = []
        async def handler(request):
            calls.append(request)
            return response("Обрезанный ответ", "length") if len(calls) == 1 else response()
        app = CompareApp(SETTINGS, results_dir=self.directory, transport=httpx.MockTransport(handler))
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.click("#start")
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(len(calls), 6)
            self.assertEqual(app.session["status"], "partial")
            self.assertNotIn("analysis", app.session)
            self.assertIn("Общий победитель автоматически не определяется", app.query_one("#conclusion-text", Static).content.markup)
            app.action_open_latest()
            self.assertEqual(len(calls), 6)
            self.assertIn("Общий победитель автоматически не определяется", app.query_one("#conclusion-text", Static).content.markup)

    async def test_tui_missing_keys_does_not_start(self):
        app = CompareApp(Settings(MODELS), results_dir=self.directory)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.click("#start")
            self.assertFalse(app.busy)
            self.assertIn("DEEPSEEK_API_KEY", str(app.query_one("#phase", Static).render()))
            self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
