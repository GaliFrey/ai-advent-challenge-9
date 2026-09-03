#!/usr/bin/env python3
"""Эксперименты с температурой; API вызывается только командами run и record."""

import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from prompts import EXPECTED_FACTS, PROMPTS, PROMPT_SETS, PROMPT_VERSION, SYSTEM_PROMPT, TITLES

ROOT = Path(__file__).resolve().parent
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
TEMPERATURES = (0, 0.7, 1.2)
SCHEMA_VERSION = 1


class ExperimentError(RuntimeError):
    """Ошибка транспорта или сохранённых данных с безопасным описанием."""


def now():
    return datetime.now(timezone.utc).isoformat()


def build_request(experiment, temperature, prompt_version=PROMPT_VERSION):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_SETS[prompt_version][experiment]},
        ],
        "thinking": {"type": "disabled"},
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": 1600,
        "stream": False,
    }


def build_plan(experiments, runs, prompt_version=PROMPT_VERSION):
    plan = []
    for experiment in experiments:
        for repeat in range(1, runs + 1):
            offset = (repeat - 1) % len(TEMPERATURES)
            order = TEMPERATURES[offset:] + TEMPERATURES[:offset]
            for temperature in order:
                plan.append({
                    "file": f"{len(plan) + 1:02d}-{experiment}-t{temperature}-r{repeat}.json",
                    "experiment": experiment,
                    "temperature": temperature,
                    "repeat": repeat,
                    "request": build_request(experiment, temperature, prompt_version),
                })
    return plan


def ask_deepseek(payload, api_key):
    request = Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        # Не сохраняем тело ошибки/заголовки: провайдер может отразить секрет.
        raise ExperimentError(f"DeepSeek HTTP {error.code}; автоматического повтора нет") from None
    except (URLError, TimeoutError, OSError, UnicodeError):
        raise ExperimentError("Ошибка соединения с DeepSeek; автоматического повтора нет") from None
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ExperimentError("API вернул некорректный JSON") from None
    if not isinstance(result, dict):
        raise ExperimentError("API вернул неожиданный тип ответа")
    return result


def answer_parts(response):
    try:
        choice = response["choices"][0]
        answer = choice["message"]["content"]
        finish = choice.get("finish_reason", "unknown")
        if isinstance(answer, str):
            return answer, finish
    except (KeyError, TypeError, IndexError, AttributeError):
        pass
    return "", "unknown"


def reject_constant(value):
    raise ValueError("Недопустимая числовая константа JSON")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повторяющийся ключ JSON")
        result[key] = value
    return result


def parse_answer(answer):
    return json.loads(answer, object_pairs_hook=unique_object, parse_constant=reject_constant)


def fact_count(experiment, prompt_version=PROMPT_VERSION):
    return 0 if experiment == "creative" and prompt_version == "v3" else len(EXPECTED_FACTS)


def fact_score(validation):
    total = validation["facts_total"]
    return f"{len(validation['correct_fields'])}/{total}" if total else "—"


def validate_answer(experiment, answer, finish_reason, prompt_version=PROMPT_VERSION):
    errors = []
    parsed = None
    json_valid = False
    try:
        parsed = parse_answer(answer)
        json_valid = True
    except (ValueError, TypeError):
        errors.append("Ответ не является одним строгим JSON без повторяющихся ключей")

    facts_total = fact_count(experiment, prompt_version)
    facts = parsed.get("facts") if isinstance(parsed, dict) else None
    correct_fields = []
    for key, expected in (EXPECTED_FACTS.items() if facts_total else ()):
        present = isinstance(facts, dict) and key in facts
        actual = facts[key] if present else None
        if type(expected) is int:
            matches = present and type(actual) in (int, float) and actual == expected
        else:
            matches = present and type(actual) is type(expected) and actual == expected
        if matches:
            correct_fields.append(key)
        else:
            errors.append(f"Факт {key}: отсутствует или не совпадает с эталоном")

    facts_shape = isinstance(facts, dict) and set(facts) == set(EXPECTED_FACTS)
    if facts_total and not facts_shape:
        errors.append("Объект facts должен содержать ровно девять заданных полей")

    expected_keys = {"facts"} if facts_total else set()
    if experiment == "creative":
        expected_keys.update({"slogan", "activities"})
    shape_ok = isinstance(parsed, dict) and set(parsed) == expected_keys
    if not shape_ok:
        errors.append("Неверный набор полей верхнего уровня")
    creative_shape = None
    if experiment == "creative":
        slogan = parsed.get("slogan") if isinstance(parsed, dict) else None
        activities = parsed.get("activities") if isinstance(parsed, dict) else None
        creative_shape = (
            isinstance(slogan, str) and bool(slogan.strip())
            and isinstance(activities, list) and len(activities) == 3
            and all(isinstance(item, str) and bool(item.strip()) for item in activities)
        )
        if not creative_shape:
            errors.append("Нужны непустой слоган и ровно три непустые строки идей")

    if finish_reason != "stop":
        errors.append("Ответ не завершён с finish_reason=stop")

    return {
        "json_valid": json_valid,
        "correct_fields": correct_fields,
        "facts_total": facts_total,
        "facts_correct": len(correct_fields) == facts_total if facts_total else None,
        "creative_shape": creative_shape,
        "passed": not errors,
        "errors": errors,
    }


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_series(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ExperimentError("Неизвестная версия формата серии")
    plan = manifest["plan"]
    prompt_version = manifest.get("prompt_version", "v1")
    if prompt_version not in PROMPT_SETS:
        raise ExperimentError("Неизвестная версия промптов в серии")
    expected_plan = build_plan(manifest["experiments"], manifest["runs_per_temperature"], prompt_version)
    if plan != expected_plan:
        raise ExperimentError("План серии не совпадает с промптами и настройками этой версии кода")
    planned_names = {slot["file"] for slot in plan}
    actual_names = {p.name for p in directory.glob("*.json")} - {"manifest.json"}
    if actual_names - planned_names:
        raise ExperimentError("В серии есть незапланированные JSON-файлы")

    rows = []
    for slot in plan:
        row = {"slot": slot, "record": None, "validation": None}
        path = directory / slot["file"]
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            for key in ("experiment", "temperature", "repeat", "request"):
                if record.get(key) != slot[key]:
                    raise ExperimentError(f"Запись {slot['file']} не совпадает с планом")
            if record.get("status") not in ("ok", "error"):
                raise ExperimentError(f"Неизвестный статус записи {slot['file']}")
            row["record"] = record
            if record["status"] == "ok":
                answer, finish = answer_parts(record.get("response"))
                row["validation"] = validate_answer(slot["experiment"], answer, finish, prompt_version)
        rows.append(row)
    return manifest, rows


def normalized(text):
    return " ".join(text.casefold().split())


def metric(record, key):
    response = record.get("response", {})
    usage = response.get("usage", {})
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else None


def sum_metric(records, key):
    values = [metric(record, key) for record in records]
    return str(sum(values)) if values and all(value is not None for value in values) else "н/д"


def fenced(text):
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def compact_answer(answer):
    """Меняет только представление JSON; не сокращает значения и не скрывает поля."""
    try:
        parsed = parse_answer(answer)
    except (ValueError, TypeError):
        return answer
    if not isinstance(parsed, dict):
        return answer
    labels = {
        "name": "Выставка", "date": "Дата", "start": "Начало", "end": "Окончание",
        "venue": "Место", "duration_minutes": "Длительность, мин",
        "total_before_discount": "До скидки, руб", "total_after_discount": "После скидки, руб",
        "guide": "Экскурсовод",
    }

    def text_value(value):
        encoded = json.dumps(value, ensure_ascii=False)
        return encoded[1:-1] if isinstance(value, str) else encoded

    lines = []
    for key, value in parsed.items():
        if key == "facts" and isinstance(value, dict) and value:
            cells = []
            for field, item in value.items():
                if field == "guide" and item is None:
                    display = "не указан (null)"
                elif isinstance(item, str) and type(EXPECTED_FACTS.get(field)) is int:
                    display = json.dumps(item, ensure_ascii=False)
                else:
                    display = text_value(item)
                cells.append(f"{labels.get(field, field)}: {display}")
            lines.extend(" | ".join(cells[index:index + 3]) for index in range(0, len(cells), 3))
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            lines.append(f"{'Идеи' if key == 'activities' else key}:")
            lines.extend(f"  {index}. {text_value(item)}"
                         for index, item in enumerate(value, 1))
        else:
            lines.append(f"{'Слоган' if key == 'slogan' else key}: {text_value(value)}")
    return "\n".join(lines) if lines else answer


def perform_call(slot, api_key, caller):
    record = {key: value for key, value in slot.items() if key != "file"}
    record["started_at"] = now()
    started = time.monotonic()
    try:
        response = caller(slot["request"], api_key)
    except (ExperimentError, KeyboardInterrupt) as error:
        record.update(status="error", error=str(error) or "Вызов прерван пользователем")
    else:
        answer, finish = answer_parts(response)
        record.update(
            status="ok", response=response,
            validation=validate_answer(slot["experiment"], answer, finish),
        )
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def record_sample(experiment, temperature, session, api_key, results_dir=None, caller=None):
    """Дополняет общую видеосессию тремя ответами одного выбранного сочетания."""
    if experiment not in PROMPTS or temperature not in TEMPERATURES:
        raise ExperimentError("Выберите промпт facts/creative и температуру 0/0.7/1.2")
    session_name(session)
    caller = caller or ask_deepseek
    directory = (results_dir or ROOT / "results") / session
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "mode": "recording",
            "started_at": now(),
            "api_url": API_URL,
            "experiments": list(PROMPTS),
            "runs_per_temperature": 3,
            "plan": build_plan(list(PROMPTS), 3),
        })
    manifest, rows = load_series(directory)
    if manifest.get("mode") != "recording" or manifest["runs_per_temperature"] != 3:
        raise ExperimentError("Каталог занят другой серией; задайте новое имя через --session")
    if manifest.get("prompt_version", "v1") != PROMPT_VERSION:
        raise ExperimentError("В сессии записана другая версия промптов. "
                              "Для новых промптов задайте отдельную сессию, например --session video-soft")
    selected = [row for row in rows if row["slot"]["experiment"] == experiment
                and row["slot"]["temperature"] == temperature]
    if len(selected) != 3:
        raise ExperimentError("План видеосессии должен содержать три повтора выбранного сочетания")
    if any(row["record"] is not None for row in selected):
        raise ExperimentError("Это сочетание уже запускалось; результаты сохранены. "
                              "Для новой записи задайте другое имя через --session")

    print(f"{TITLES[experiment]} | промпт: {experiment}/{PROMPT_VERSION} | T={temperature:g} | "
          f"3 API-вызова | сессия: {session}", flush=True)
    for message in selected[0]["slot"]["request"]["messages"]:
        title = "Системная инструкция" if message["role"] == "system" else "Промпт"
        print(f"\n{title}:\n{message['content']}", flush=True)
    records = []
    completed = True
    try:
        for index, row in enumerate(selected, 1):
            slot = row["slot"]
            record = perform_call(slot, api_key, caller)
            write_json(directory / slot["file"], record)
            if record["status"] == "error":
                completed = False
                print(f"\nОтвет {index}/3: {record['error']}", flush=True)
                break
            records.append(record)
            answer, finish = answer_parts(record["response"])
            validation = record["validation"]
            usage = [metric(record, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
            incoming, outgoing, total = [str(value) if value is not None else "н/д" for value in usage]
            print(f"\nОтвет {index}/3 | токены: вход {incoming}, выход {outgoing}, всего {total} | "
                  f"{record['elapsed_seconds']:.2f} с", flush=True)
            print(compact_answer(answer), flush=True)
            facts_info = f"факты {fact_score(validation)} | " if validation["facts_total"] else ""
            print(f"Автопроверка: {'PASS' if validation['passed'] else 'FAIL'} | "
                  f"{facts_info}завершение: {finish}", flush=True)
            if validation["errors"]:
                print("; ".join(validation["errors"]), flush=True)
    finally:
        render_reports(directory)
        _, saved_rows = load_series(directory)
        saved = sum(row["record"] is not None and row["record"]["status"] == "ok" for row in saved_rows)
        print(f"\nЗа этот запуск: {len(records)}/3 ответов, токенов {sum_metric(records, 'total_tokens')}. "
              f"В сессии: {saved}/18 ответов.", flush=True)
        print(f"Сохранено: {directory}", flush=True)
    return directory, completed


def group_metrics(group, prompt_version=PROMPT_VERSION):
    """Одна трактовка метрик для Markdown и консольной таблицы."""
    available = [row for row in group if row["validation"] is not None]
    records = [row["record"] for row in available]
    canonical, slogans, ideas = [], [], []
    for row in available:
        answer, _ = answer_parts(row["record"]["response"])
        if row["validation"]["json_valid"]:
            parsed = parse_answer(answer)
            canonical.append(json.dumps(parsed, sort_keys=True, ensure_ascii=False))
            if row["slot"]["experiment"] == "creative" and row["validation"]["creative_shape"]:
                slogans.append(parsed["slogan"])
                ideas.extend(parsed["activities"])
    facts_total = sum(fact_count(row["slot"]["experiment"], prompt_version) for row in group)
    return {
        "received": f"{len(available)}/{len(group)}",
        "facts": f"{sum(len(row['validation']['correct_fields']) for row in available)}/{facts_total}" if facts_total else "—",
        "passes": f"{sum(row['validation']['passed'] for row in available)}/{len(group)}",
        "variants": f"{len(set(canonical))}/{len(canonical)}",
        "slogan_count": f"{len({normalized(item) for item in slogans})}/{len(slogans)}",
        "idea_count": f"{len({normalized(item) for item in ideas})}/{len(ideas)}",
        "slogans": slogans,
        "input_tokens": sum_metric(records, "prompt_tokens"),
        "output_tokens": sum_metric(records, "completion_tokens"),
        "total_tokens": sum_metric(records, "total_tokens"),
        "seconds": f"{mean(record['elapsed_seconds'] for record in records):.2f}" if records else "н/д",
    }


def terminal_table(headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]

    def border(left, join, right):
        return left + join.join("─" * (width + 2) for width in widths) + right

    def line(cells):
        return "│ " + " │ ".join(cell.rjust(width) for cell, width in zip(cells, widths)) + " │"

    return "\n".join([
        border("┌", "┬", "┐"), line(headers), border("├", "┼", "┤"),
        *(line(row) for row in rows), border("└", "┴", "┘"),
    ])


def render_console_summary(directory, manifest, rows):
    """Короткие таблицы и фактические слоганы; без выдуманных оценок творчества."""
    received = sum(row["validation"] is not None for row in rows)
    errors = sum(row["record"] is not None and row["record"]["status"] == "error" for row in rows)
    lines = ["СРАВНЕНИЕ ТЕМПЕРАТУР", f"Сессия: {directory.name} · промпты {manifest.get('prompt_version', 'v1')} · "
             f"ответов {received}/{len(rows)} · ошибок API {errors}", ""]
    creative_groups = []
    prompt_version = manifest.get("prompt_version", "v1")
    for experiment in manifest["experiments"]:
        has_facts = bool(fact_count(experiment, prompt_version))
        headers = ["T", "Ответы", "PASS"]
        if has_facts:
            headers.append("Факты")
        headers.append("Варианты")
        if experiment == "creative":
            headers.extend(["Слоганы", "Идеи"])
        headers.append("Токены")
        table_rows = []
        for temperature in TEMPERATURES:
            group = [row for row in rows if row["slot"]["experiment"] == experiment
                     and row["slot"]["temperature"] == temperature]
            metrics = group_metrics(group, prompt_version)
            cells = [str(temperature), metrics["received"], metrics["passes"]]
            if has_facts:
                cells.append(metrics["facts"])
            cells.append(metrics["variants"])
            if experiment == "creative":
                cells.extend([metrics["slogan_count"], metrics["idea_count"]])
                creative_groups.append((temperature, metrics["slogans"]))
            cells.append(metrics["total_tokens"])
            table_rows.append(cells)
        lines.extend([TITLES[experiment].upper(), terminal_table(headers, table_rows), ""])
    lines.extend([
        "PASS — формат и завершение; факты — когда они предусмотрены.",
        "Варианты — уникальные JSON из полученных ответов.",
        "Слоганы и идеи — уникальные тексты; токены — сумма входа и выхода.",
        "Разные формулировки могут описывать одну идею. Качество не оценено.",
    ])
    if any(slogans for _, slogans in creative_groups):
        lines.extend(["", "СЛОГАНЫ В СОХРАНЁННЫХ ОТВЕТАХ"])
        for temperature, slogans in creative_groups:
            unique = {}
            for slogan in slogans:
                key = normalized(slogan)
                if key not in unique:
                    unique[key] = [slogan, 0]
                unique[key][1] += 1
            for slogan, count in unique.values():
                lines.append(textwrap.fill(
                    f"T={temperature}: «{slogan}» ({count}/{len(slogans)})", width=78,
                    subsequent_indent="       ", break_long_words=False, break_on_hyphens=False,
                ))
    lines.extend(["", "Полные ответы: answers.md · подробные метрики: summary.md"])
    return "\n".join(lines) + "\n"


def render_reports(directory):
    manifest, rows = load_series(directory)
    successes = sum(row["validation"] is not None for row in rows)
    failures = sum(row["record"] is not None and row["record"]["status"] == "error" for row in rows)
    missing = len(rows) - successes - failures
    summary = [
        "# Сводка экспериментов дня 4", "",
        f"Серия: `{directory.name}`. Начало (UTC): {manifest['started_at']}.", "",
        f"Версия промптов: `{manifest.get('prompt_version', 'v1')}`.", "",
        f"План: {len(rows)} вызовов. Ответов API: {successes}. Ошибок вызова: {failures}. "
        f"Не выполнено: {missing}. Серия {'полная' if successes == len(rows) else 'неполная'}.", "",
        "Проверки пересчитаны по сохранённому тексту. Автопроверка включает JSON, "
        "поля, факты (где предусмотрены), форму творческого ответа и finish_reason=stop. "
        "PASS не подтверждает смысловую корректность или качество творческой части.", "",
        "| Задача | T | Ответов / план | Верных фактов / план | Автопроверка / план | "
        "Уникальных JSON / JSON | Слоганов / всего | Идей / всего | Входные токены | "
        "Выходные токены | Всего токенов | Среднее время, с |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in manifest["experiments"]:
        for temperature in TEMPERATURES:
            group = [row for row in rows if row["slot"]["experiment"] == experiment
                     and row["slot"]["temperature"] == temperature]
            metrics = group_metrics(group, manifest.get("prompt_version", "v1"))
            slogan_cell = metrics["slogan_count"] if experiment == "creative" else "—"
            idea_cell = metrics["idea_count"] if experiment == "creative" else "—"
            summary.append(
                f"| {TITLES[experiment]} | {temperature} | {metrics['received']} | "
                f"{metrics['facts']} | {metrics['passes']} | "
                f"{metrics['variants']} | {slogan_cell} | {idea_cell} | "
                f"{metrics['input_tokens']} | {metrics['output_tokens']} | "
                f"{metrics['total_tokens']} | {metrics['seconds']} |"
            )

    summary.extend([
        "", "Уникальность JSON не учитывает порядок ключей и пробелы форматирования. "
        "Текстовая уникальность слоганов и идей не означает смысловой оригинальности. "
        "В знаменателе фактов — все запланированные ответы задачи, включая отсутствующие. "
        "Прочерк означает, что метрика не применяется к задаче. "
        "Токены и время приведены только для полученных ответов; н/д означает отсутствие метрики.", "",
        "## Проверки отдельных ответов", "",
        "| Ответ | Факты | Автопроверка | Завершение | Примечания |",
        "|---|---:|---|---|---|",
    ])
    answers = ["# Промпты и полные ответы", "", f"Серия: `{directory.name}`.", ""]
    for experiment in manifest["experiments"]:
        request = next(slot["request"] for slot in manifest["plan"] if slot["experiment"] == experiment)
        answers.extend([f"## Промпт: {TITLES[experiment]}", ""])
        for message in request["messages"]:
            answers.extend([f"Роль: `{message['role']}`", "", fenced(message["content"]), ""])
    for row in rows:
        slot, record, validation = row["slot"], row["record"], row["validation"]
        label = f"{slot['experiment']} · T={slot['temperature']} · повтор {slot['repeat']}"
        answers.extend([f"## {label}", ""])
        if record is None:
            summary.append(f"| {label} | — | НЕ ВЫПОЛНЕН | — | Нет сохранённого вызова |")
            answers.extend(["Не выполнен.", ""])
            continue
        if record["status"] == "error":
            summary.append(f"| [{label}]({slot['file']}) | — | ОШИБКА API | — | Серия остановлена |")
            answers.extend([record["error"], ""])
            continue
        answer, finish = answer_parts(record["response"])
        state = "PASS" if validation["passed"] else "FAIL"
        notes = "; ".join(validation["errors"]) or "—"
        safe_finish = str(finish).replace("|", "\\|").replace("\n", " ")
        summary.append(f"| [{label}]({slot['file']}) | {fact_score(validation)} | "
                       f"{state} | {safe_finish} | {notes} |")
        answers.extend([
            f"[Исходный JSON]({slot['file']}); проверка: {state}; завершение: `{safe_finish}`.",
            "", fenced(answer), "",
        ])
    summary.extend([
        "", "## Ручная оценка", "",
        "Оценку заполняет пользователь после чтения [полных ответов](answers.md). "
        "Автоматические показатели выше не оценивают качество творчества.", "",
        "| Задача | T | Креативность и уместность | Смысловое разнообразие | Примеры / замечания |",
        "|---|---:|---|---|---|",
    ])
    for experiment in manifest["experiments"]:
        for temperature in TEMPERATURES:
            summary.append(f"| {TITLES[experiment]} | {temperature} | "
                           f"{'Не требуется' if experiment == 'facts' else 'Не оценено'} | Не оценено | — |")
    summary.extend(["", "Ручные выводы следует сохранять в README дня 4. Этот файл генерируется "
                    "повторно командой compare; ручные изменения в нём будут перезаписаны.", ""])
    (directory / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (directory / "answers.md").write_text("\n".join(answers), encoding="utf-8")
    (directory / "summary.txt").write_text(render_console_summary(directory, manifest, rows), encoding="utf-8")
    return "\n".join(summary)


def run_series(experiments, runs, api_key, results_dir=None, caller=None):
    caller = caller or ask_deepseek
    results_dir = results_dir or ROOT / "results"
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = results_dir / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    plan = build_plan(experiments, runs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "started_at": now(),
        "api_url": API_URL,
        "experiments": list(experiments),
        "runs_per_temperature": runs,
        "plan": plan,
    }
    write_json(directory / "manifest.json", manifest)
    print(f"Серия: {directory}\nВнешних API-вызовов по плану: {len(plan)}", flush=True)
    completed = True
    try:
        for index, slot in enumerate(plan, 1):
            print(f"[{index}/{len(plan)}] {TITLES[slot['experiment']]} | "
                  f"T={slot['temperature']} | повтор {slot['repeat']}", flush=True)
            record = perform_call(slot, api_key, caller)
            write_json(directory / slot["file"], record)
            if record["status"] == "error":
                completed = False
                print(record["error"], flush=True)
                break
            validation = record["validation"]
            facts_info = f"фактов {fact_score(validation)} | " if validation["facts_total"] else ""
            print(f"  {'PASS' if validation['passed'] else 'FAIL'} | "
                  f"{facts_info}"
                  f"токенов {metric(record, 'total_tokens')} | {record['elapsed_seconds']:.2f} с", flush=True)
    finally:
        render_reports(directory)
        print(f"Сводка: {directory / 'summary.md'}\nОтветы: {directory / 'answers.md'}", flush=True)
    return directory, completed


def repeat_count(value):
    number = int(value)
    if not 1 <= number <= 5:
        raise argparse.ArgumentTypeError("Число повторов должно быть от 1 до 5")
    return number


def session_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise argparse.ArgumentTypeError("Имя сессии: 1–64 латинских буквы, цифры, - или _; первый символ — буква или цифра")
    return value


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="День 4: два эксперимента с температурой LLM")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Для видео: 3 ответа одного промпта при одной температуре")
    record.add_argument("--prompt", choices=tuple(PROMPTS), required=True, help="facts — факты; creative — свободное творчество")
    record.add_argument("--temperature", type=float, choices=TEMPERATURES, required=True)
    record.add_argument("--session", type=session_name, default="video", help="Общая сессия шести запусков (по умолчанию video)")
    run = commands.add_parser("run", help="Выполнить новую серию внешних API-вызовов")
    run.add_argument("--experiment", choices=("all", "facts", "creative"), default="all")
    run.add_argument("--runs", type=repeat_count, default=3, help="Повторов на температуру (1–5, по умолчанию 3)")
    compare = commands.add_parser("compare", help="Пересчитать сводку сохранённой серии без API")
    compare.add_argument("directory", type=Path, help="Каталог серии с manifest.json")
    compare.add_argument("--compact", action="store_true", help="Таблицы для терминала и фактические слоганы без подробных проверок")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        if args.command == "compare":
            report = render_reports(args.directory.resolve())
            print((args.directory / "summary.txt").read_text(encoding="utf-8").rstrip() if args.compact else report)
            return 0
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key or api_key == "your_api_key_here":
            raise ExperimentError("Укажите DEEPSEEK_API_KEY в day-04/.env или окружении")
        if args.command == "record":
            _, completed = record_sample(args.prompt, args.temperature, args.session, api_key)
            return 0 if completed else 1
        experiments = list(PROMPTS) if args.experiment == "all" else [args.experiment]
        _, completed = run_series(experiments, args.runs, api_key)
        return 0 if completed else 1
    except (ExperimentError, OSError, ValueError, KeyError, TypeError) as error:
        # Не печатаем произвольные исключения с содержимым файлов или секретами.
        message = str(error) if isinstance(error, ExperimentError) else "Не удалось прочитать или сохранить данные серии"
        print(f"Ошибка: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Остановлено пользователем; полученные результаты сохранены", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
