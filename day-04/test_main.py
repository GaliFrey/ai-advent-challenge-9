"""Локальные проверки контракта и достоверности сохраняемого эксперимента."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from prompts import EXPECTED_FACTS


def response(experiment="facts", finish="stop", prompt_version=main.PROMPT_VERSION):
    body = {"facts": copy.deepcopy(EXPECTED_FACTS)} if main.fact_count(experiment, prompt_version) else {}
    if experiment == "creative":
        body.update(slogan="Космос рядом", activities=["Идея один", "Идея два", "Идея три"])
    return {
        "id": "synthetic-test-response",
        "model": "synthetic-model",
        "choices": [{"message": {"content": json.dumps(body, ensure_ascii=False)}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
    }


class ContractTests(unittest.TestCase):
    def validate(self, body, experiment="facts", finish="stop"):
        return main.validate_answer(experiment, json.dumps(body, ensure_ascii=False), finish)

    def test_only_temperature_changes_within_experiment(self):
        for version in main.PROMPT_SETS:
            for experiment in main.PROMPTS:
                requests = [main.build_request(experiment, t, version) for t in main.TEMPERATURES]
                for request in requests:
                    self.assertEqual(request["thinking"], {"type": "disabled"})
                    self.assertEqual(len(request["messages"]), 2)
                    request.pop("temperature")
                self.assertTrue(all(request == requests[0] for request in requests))

    def test_balanced_plan_and_independent_calls(self):
        plan = main.build_plan(["facts", "creative"], 3)
        self.assertEqual(len(plan), 18)
        self.assertEqual(len({slot["file"] for slot in plan}), 18)
        for experiment in main.PROMPTS:
            for temperature in main.TEMPERATURES:
                slots = [slot for slot in plan if slot["experiment"] == experiment
                         and slot["temperature"] == temperature]
                self.assertEqual([slot["repeat"] for slot in slots], [1, 2, 3])
        self.assertEqual([slot["temperature"] for slot in plan[:9]], [0, .7, 1.2, .7, 1.2, 0, 1.2, 0, .7])

    def test_expected_arithmetic(self):
        self.assertEqual(EXPECTED_FACTS["duration_minutes"], 17 * 60 - (10 * 60 + 30))
        self.assertEqual(EXPECTED_FACTS["total_before_discount"], 3 * 480 + 2 * 260)
        self.assertEqual(EXPECTED_FACTS["total_after_discount"], 1960 * 85 // 100)

    def test_valid_answers_in_both_experiments(self):
        for experiment in main.PROMPTS:
            answer, finish = main.answer_parts(response(experiment))
            self.assertTrue(main.validate_answer(experiment, answer, finish)["passed"])

    def test_creative_prompt_is_independent_of_source_and_calculations(self):
        request = main.build_request("creative", 0)
        text = "\n".join(message["content"] for message in request["messages"]).casefold()
        for fragment in ("facts", "музей", "выставк", "билет", "скидк", "вектор", "21 ноября", "10:30", "480"):
            self.assertNotIn(fragment, text)
        self.assertEqual(main.PROMPTS["facts"], main.PROMPT_SETS["v1"]["facts"])

    def test_creative_contract_has_no_factual_score(self):
        answer, _ = main.answer_parts(response("creative"))
        body = json.loads(answer)
        validation = self.validate(body, "creative")
        self.assertTrue(validation["passed"])
        self.assertIsNone(validation["facts_correct"])
        self.assertEqual(main.fact_score(validation), "—")
        self.assertFalse(self.validate({**body, "facts": EXPECTED_FACTS}, "creative")["passed"])
        self.assertFalse(self.validate(body, "creative", "length")["passed"])
        self.assertFalse(self.validate({**body, "slogan": " "}, "creative")["passed"])

    def test_invented_guide_wrong_arithmetic_and_boolean_fail(self):
        for key, value in [("guide", "Иван"), ("total_after_discount", 1667), ("duration_minutes", True)]:
            body = {"facts": {**EXPECTED_FACTS, key: value}}
            result = self.validate(body)
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["correct_fields"]), 8)

    def test_missing_null_is_not_unknown_value(self):
        body = {"facts": copy.deepcopy(EXPECTED_FACTS)}
        del body["facts"]["guide"]
        self.assertFalse(self.validate(body)["facts_correct"])

    def test_extra_fields_are_separate_from_factual_accuracy(self):
        body = {"facts": {**EXPECTED_FACTS, "organizer": "Выдуман"}}
        validation = self.validate(body)
        self.assertTrue(validation["facts_correct"])
        self.assertFalse(validation["passed"])

    def test_strict_json_rejects_wrappers_duplicates_and_nan(self):
        answer, _ = main.answer_parts(response())
        for bad in [f"```json\n{answer}\n```", answer + "\nГотово", '{"facts": {}, "facts": {}}',
                    '{"facts": NaN}', '{"facts": Infinity}', "[]", "null", ""]:
            self.assertFalse(main.validate_answer("facts", bad, "stop")["passed"])

    def test_truncation_is_not_a_success_even_with_valid_json(self):
        result = self.validate({"facts": EXPECTED_FACTS}, finish="length")
        self.assertTrue(result["facts_correct"])
        self.assertFalse(result["passed"])

    def test_creative_shape_requires_three_nonempty_strings(self):
        for activities in [["one", "two"], ["one", "two", ""], ["one", "two", 3]]:
            result = self.validate({"slogan": "Слоган", "activities": activities}, "creative")
            self.assertFalse(result["passed"])
            self.assertFalse(result["creative_shape"])
            self.assertIsNone(result["facts_correct"])

    def test_runs_are_bounded(self):
        for value in ("0", "6", "-1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main.parse_arguments(["run", "--runs", value])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fake(self, caller=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return main.run_series(["facts"], 1, "synthetic-test-key", self.root,
                                   caller or (lambda payload, key: response()))

    def test_complete_series_stores_raw_requests_responses_without_key(self):
        directory, completed = self.run_fake()
        self.assertTrue(completed)
        self.assertEqual(len(list(directory.glob("*.json"))), 4)
        for path in directory.iterdir():
            text = path.read_text()
            self.assertNotIn("synthetic-test-key", text)
            self.assertNotIn("Authorization", text)
        manifest, rows = main.load_series(directory)
        self.assertEqual(rows[0]["record"]["response"], response())
        self.assertEqual(rows[0]["record"]["request"], manifest["plan"][0]["request"])
        self.assertIn("Серия полная", (directory / "summary.md").read_text())

    def test_failed_call_stops_without_retry_and_keeps_partial_results(self):
        calls = []

        def caller(payload, key):
            calls.append(payload)
            if len(calls) == 2:
                raise main.ExperimentError("Синтетическая ошибка")
            return response()

        directory, completed = self.run_fake(caller)
        self.assertFalse(completed)
        self.assertEqual(len(calls), 2)
        _, rows = main.load_series(directory)
        self.assertTrue(rows[0]["validation"]["passed"])
        self.assertEqual(rows[1]["record"]["status"], "error")
        self.assertIsNone(rows[2]["record"])
        summary = (directory / "summary.md").read_text()
        self.assertIn("Ошибок вызова: 1. Не выполнено: 1. Серия неполная", summary)
        self.assertIn("н/д", summary)

    def test_repeated_series_does_not_overwrite(self):
        first, _ = self.run_fake()
        snapshot = (first / "manifest.json").read_bytes()
        second, _ = self.run_fake()
        self.assertNotEqual(first, second)
        self.assertEqual((first / "manifest.json").read_bytes(), snapshot)

    def test_compare_revalidates_actual_answer_not_saved_pass(self):
        directory, _ = self.run_fake()
        manifest, _ = main.load_series(directory)
        path = directory / manifest["plan"][0]["file"]
        record = json.loads(path.read_text())
        body = {"facts": {**EXPECTED_FACTS, "total_after_discount": 1}}
        record["response"]["choices"][0]["message"]["content"] = json.dumps(body)
        self.assertTrue(record["validation"]["passed"])
        main.write_json(path, record)
        self.assertIn("FAIL", main.render_reports(directory))

    def test_compare_rejects_mismatched_request(self):
        directory, _ = self.run_fake()
        manifest, _ = main.load_series(directory)
        path = directory / manifest["plan"][0]["file"]
        record = json.loads(path.read_text())
        record["request"]["messages"].append({"role": "assistant", "content": "Предыдущий ответ"})
        main.write_json(path, record)
        with self.assertRaises(main.ExperimentError):
            main.render_reports(directory)

    def test_textual_uniqueness_is_not_json_formatting_difference(self):
        calls = []

        def caller(payload, key):
            result = response()
            body = {"facts": dict(reversed(list(EXPECTED_FACTS.items())))} if calls else {"facts": EXPECTED_FACTS}
            calls.append(payload)
            result["choices"][0]["message"]["content"] = json.dumps(body, indent=len(calls))
            return result

        with contextlib.redirect_stdout(io.StringIO()):
            directory, _ = main.run_series(["facts"], 3, "test-key", self.root, caller)
        table = (directory / "summary.md").read_text()
        self.assertIn("| 3/3 | 27/27 | 3/3 | 1/3 |", table)

    def test_compare_does_not_call_api(self):
        directory, _ = self.run_fake()
        with patch.object(main, "ask_deepseek", side_effect=AssertionError("API called")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(["compare", str(directory)]), 0)

    def test_missing_token_usage_is_not_reported_as_zero(self):
        self.assertEqual(main.sum_metric([{"response": {}}], "total_tokens"), "н/д")
        self.assertEqual(main.sum_metric([], "total_tokens"), "н/д")


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, experiment="facts", temperature=0, caller=None, session="video"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            directory, completed = main.record_sample(
                experiment, temperature, session, "synthetic-test-key", self.root,
                caller or (lambda payload, key: response(experiment)),
            )
        return directory, completed, output.getvalue()

    def test_cli_selects_exact_pair_and_performs_three_calls(self):
        calls = []

        def caller(payload, key):
            calls.append(payload)
            self.assertEqual(payload["temperature"], .7)
            self.assertEqual(payload["messages"][1]["content"], main.PROMPTS["creative"])
            self.assertEqual(output.getvalue().count(main.PROMPTS["creative"]), 1)
            self.assertEqual(output.getvalue().count(main.SYSTEM_PROMPT), 1)
            return response("creative")

        with patch.object(main, "ROOT", self.root), patch.object(main, "load_dotenv"), \
                patch.dict(main.os.environ, {"DEEPSEEK_API_KEY": "synthetic-test-key"}), \
                patch.object(main, "ask_deepseek", side_effect=caller), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = main.main(["record", "--prompt", "creative", "--temperature", "0.7"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(output.getvalue().count("| токены: вход 100, выход 200, всего 300"), 3)
        for index in (1, 2, 3):
            self.assertIn(f"Ответ {index}/3", output.getvalue())
        self.assertEqual(output.getvalue().count("Космос рядом"), 3)
        self.assertEqual(output.getvalue().count("Идея три"), 3)
        self.assertEqual(output.getvalue().count(main.PROMPTS["creative"]), 1)
        self.assertNotIn("synthetic-test-key", output.getvalue())
        self.assertIn("В сессии: 3/18 ответов", output.getvalue())

    def test_six_invocations_accumulate_all_answers_without_overwriting(self):
        snapshot = {}
        for experiment in main.PROMPTS:
            for temperature in main.TEMPERATURES:
                directory, completed, _ = self.record(experiment, temperature)
                self.assertTrue(completed)
                for name, content in snapshot.items():
                    self.assertEqual((directory / name).read_bytes(), content)
                snapshot = {path.name: path.read_bytes() for path in directory.glob("*.json")}
        _, rows = main.load_series(directory)
        self.assertEqual(sum(row["record"] is not None for row in rows), 18)
        self.assertIn("Серия полная", (directory / "summary.md").read_text())
        self.assertIn("Идея три", (directory / "answers.md").read_text())

    def test_duplicate_is_rejected_before_api_and_preserves_data(self):
        directory, _, _ = self.record()
        snapshot = {path.name: path.read_bytes() for path in directory.iterdir()}
        with self.assertRaises(main.ExperimentError), patch.object(main, "ask_deepseek") as api:
            main.record_sample("facts", 0, "video", "test-key", self.root)
        api.assert_not_called()
        self.assertEqual(snapshot, {path.name: path.read_bytes() for path in directory.iterdir()})
        other, _, _ = self.record(session="video2")
        self.assertNotEqual(other, directory)

    def test_failure_is_saved_and_successful_response_is_printed(self):
        calls = []

        def caller(payload, key):
            calls.append(payload)
            if len(calls) == 2:
                raise main.ExperimentError("Синтетическая сетевая ошибка")
            return response()

        directory, completed, output = self.record(caller=caller)
        self.assertFalse(completed)
        self.assertEqual(len(calls), 2)
        self.assertIn("Орбита", output)
        self.assertIn("Ответ 2/3: Синтетическая сетевая ошибка", output)
        _, rows = main.load_series(directory)
        self.assertEqual(sum(row["record"] is not None for row in rows), 2)
        with self.assertRaises(main.ExperimentError), patch.object(main, "ask_deepseek") as api:
            main.record_sample("facts", 0, "video", "test-key", self.root)
        api.assert_not_called()

    def test_invalid_answer_is_printed_verbatim_and_does_not_stop_repeats(self):
        bad = response()
        text = '```json\n{"facts": {"total_after_discount": 1258}}\n```'
        bad["choices"][0]["message"]["content"] = text
        _, completed, output = self.record(caller=lambda payload, key: copy.deepcopy(bad))
        self.assertTrue(completed)
        self.assertEqual(output.count(text), 3)
        self.assertEqual(output.count("Автопроверка: FAIL"), 3)

    def test_compact_output_preserves_all_content_and_does_not_hide_extra_keys(self):
        answer, _ = main.answer_parts(response("creative"))
        body = json.loads(answer)
        body["facts"] = copy.deepcopy(EXPECTED_FACTS)
        body["extra"] = "Дополнительное поле"
        output = main.compact_answer(json.dumps(body, ensure_ascii=False, indent=2))
        for value in body["activities"] + [body["slogan"], body["extra"]]:
            self.assertIn(value, output)
        for label in ("Выставка: Орбита", "Дата: 21 ноября", "Начало: 10:30",
                      "Окончание: 17:00", "Место: музей Вектор", "Длительность, мин: 390",
                      "До скидки, руб: 1960", "После скидки, руб: 1666", "Экскурсовод: не указан (null)"):
            self.assertIn(label, output)
        self.assertNotIn('facts: {', output)
        duplicate = '{"facts": {}, "facts": {"guide": "Иван"}}'
        self.assertEqual(main.compact_answer(duplicate), duplicate)

    def test_record_requires_both_parameters_and_safe_session_name(self):
        for args in [[], ["--prompt", "facts"], ["--temperature", "0"],
                     ["--prompt", "facts", "--temperature", "2"],
                     ["--prompt", "facts", "--temperature", "0", "--session", "../old"]]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main.parse_arguments(["record", *args])

    def test_record_does_not_append_to_legacy_series(self):
        with contextlib.redirect_stdout(io.StringIO()):
            directory, _ = main.run_series(["facts", "creative"], 3, "test-key", self.root,
                                           lambda payload, key: response())
        with self.assertRaises(main.ExperimentError), patch.object(main, "ask_deepseek") as api:
            main.record_sample("facts", 0, directory.name, "test-key", self.root)
        api.assert_not_called()

    def test_compact_compare_keeps_full_reports_on_disk(self):
        directory, _, _ = self.record()
        with patch.object(main, "ask_deepseek", side_effect=AssertionError("API called")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main.main(["compare", str(directory), "--compact"]), 0)
        self.assertIn("ФАКТЫ И РАСЧЁТЫ", output.getvalue())
        self.assertIn("3/3", output.getvalue())
        self.assertIn("27/27", output.getvalue())
        self.assertIn("900", output.getvalue())
        self.assertIn("н/д", output.getvalue())
        self.assertNotIn("|---", output.getvalue())
        self.assertNotIn("## Проверки отдельных ответов", output.getvalue())
        self.assertEqual(output.getvalue(), (directory / "summary.txt").read_text())
        self.assertIn("## Проверки отдельных ответов", (directory / "summary.md").read_text())

    def test_creative_record_and_reports_do_not_claim_factual_accuracy(self):
        directory, completed, output = self.record("creative", 0)
        self.assertTrue(completed)
        self.assertNotIn("факты", output.casefold())
        self.assertEqual(output.count("Автопроверка: PASS"), 3)
        manifest, rows = main.load_series(directory)
        creative_rows = [row for row in rows if row["slot"]["experiment"] == "creative"]
        self.assertEqual(main.group_metrics(creative_rows)["facts"], "—")
        self.assertEqual(main.group_metrics(creative_rows)["passes"], "3/9")
        terminal = main.render_console_summary(directory, manifest, rows)
        table = terminal.split("СВОБОДНОЕ ТВОРЧЕСТВО\n")[1].split("\n\n")[0]
        self.assertNotIn("Факты", table)
        self.assertTrue(all(len(line) <= 78 for line in terminal.splitlines()))
        report = (directory / "summary.md").read_text()
        self.assertIn("| Свободное творчество | 0 | 3/3 | — | 3/3 |", report)
        self.assertIn("| — | PASS | stop |", report)


class PromptVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "legacy"
        self.directory.mkdir()
        self.manifest = {
            "schema_version": main.SCHEMA_VERSION,
            "mode": "recording",
            "started_at": "2026-09-03T00:00:00+00:00",
            "experiments": list(main.PROMPTS),
            "runs_per_temperature": 3,
            "plan": main.build_plan(list(main.PROMPTS), 3, "v1"),
        }
        main.write_json(self.directory / "manifest.json", self.manifest)

    def test_old_manifest_without_version_uses_original_prompt(self):
        manifest, rows = main.load_series(self.directory)
        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[-1]["slot"]["request"]["messages"][1]["content"], main.PROMPT_SETS["v1"]["creative"])
        self.assertNotEqual(main.PROMPT_SETS["v1"]["creative"], main.PROMPTS["creative"])
        self.assertEqual(main.PROMPT_SETS["v1"]["facts"], main.PROMPTS["facts"])
        with patch.object(main, "ask_deepseek", side_effect=AssertionError("API called")):
            report = main.render_reports(self.directory)
        self.assertIn("Версия промптов: `v1`", report)
        self.assertIn(main.PROMPT_SETS["v1"]["creative"], (self.directory / "answers.md").read_text())

    def test_record_rejects_mixing_versions_before_any_api_call(self):
        before = (self.directory / "manifest.json").read_bytes()
        with patch.object(main, "ask_deepseek") as api, self.assertRaisesRegex(main.ExperimentError, "версия промптов"):
            main.record_sample("creative", .7, "legacy", "test-key", self.root)
        api.assert_not_called()
        self.assertEqual((self.directory / "manifest.json").read_bytes(), before)

    def test_version_must_match_saved_prompts(self):
        for version in (main.PROMPT_VERSION, "unknown"):
            with self.subTest(version=version):
                main.write_json(self.directory / "manifest.json", {**self.manifest, "prompt_version": version})
                with self.assertRaises(main.ExperimentError):
                    main.load_series(self.directory)

    def test_legacy_creative_results_still_require_facts(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                plan = main.build_plan(list(main.PROMPTS), 3, version)
                manifest = {**self.manifest, "prompt_version": version, "plan": plan}
                main.write_json(self.directory / "manifest.json", manifest)
                slot = next(slot for slot in plan if slot["experiment"] == "creative")
                saved_response = response("creative", prompt_version=version)
                record = {**slot, "status": "ok", "response": saved_response, "elapsed_seconds": 1}
                main.write_json(self.directory / slot["file"], record)
                _, rows = main.load_series(self.directory)
                validation = next(row["validation"] for row in rows if row["record"])
                self.assertTrue(validation["passed"])
                self.assertTrue(validation["facts_correct"])
                report = main.render_reports(self.directory)
                self.assertIn("| 1/3 | 9/27 | 1/3 |", report)
                body = json.loads(saved_response["choices"][0]["message"]["content"])
                del body["facts"]
                saved_response["choices"][0]["message"]["content"] = json.dumps(body)
                main.write_json(self.directory / slot["file"], record)
                _, rows = main.load_series(self.directory)
                validation = next(row["validation"] for row in rows if row["record"])
                self.assertFalse(validation["passed"])
                self.assertFalse(validation["facts_correct"])


if __name__ == "__main__":
    unittest.main()
