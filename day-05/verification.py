"""Independent checks tied to an exact, versioned exercise, never an LLM rating."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PYTHON_CHECKER = "python-build-v1"
PYTHON_PROMPT_HASH = "64ea0faf90103753753ab2b42a6a23a65b78fe7660b4af33e5e64319ae77bd4e"
RELEASE_CHECKER = "release-plan-v1"
RELEASE_PROMPT_HASH = "80f36065677ba11065662ff3b205e05bf2fcc174fe785e6f0a3e6fbf4266dc41"
ORIGINAL = "[(2, (1, 2, 3)), (2, (1, 2, 3))]\n[(3, (1, 2, 3))]"
FIXED = "[(1, (1,)), (2, (1, 2))]\n[(3, (3,))]"
LABELS = {"original_stdout": "Исходный stdout", "contract": "Контракт build", "fixed_stdout": "Новый stdout", "asserts": "Assert модели"}
RELEASE_LABELS = {"best_15": "Оптимум Б15", "second_15": "Второй Б15", "best_12": "Оптимум Б12"}


def code_sections(answer):
    sections = {}
    current = None
    fence = None
    code = []
    for line in answer.splitlines():
        if fence:
            if re.fullmatch(r"\s*" + re.escape(fence) + r"\s*", line):
                sections.setdefault(current, []).append("\n".join(code))
                fence = None
            else:
                code.append(line)
            continue
        opening = re.fullmatch(r"\s*(`{3,}|~{3,})[^`~]*", line)
        if opening:
            fence, code = opening[1], []
            continue
        heading = re.fullmatch(r"\s*(?:#{1,6}\s+)?(?:\*\*)?(Исходный вывод|Причины|Исправление|Новый вывод|Проверки|Итог)(?:\*\*)?\s*:?\s*", line, re.I)
        if heading:
            current = heading[1].lower()
    return {key: blocks[0] for key, blocks in sections.items() if len(blocks) == 1}


def stdout_check(actual, expected):
    if actual is None:
        return {"status": "unknown", "detail": "Нет единственного блока stdout в нужном разделе", "expected": expected}
    # Exact stdout apart from surrounding blank space; types and values matter.
    actual = actual.strip()
    return {"status": "pass" if actual == expected else "fail", "actual": actual, "expected": expected}


def verify_answer(record):
    if record.get("status") != "ok":
        return {"checks": {key: {"status": "unknown", "detail": "Ответ отсутствует или неполон"} for key in LABELS}, "score": None}
    answer = record.get("answer", "")
    if len(answer) > 64000:
        return {"checks": {key: {"status": "unknown", "detail": "Ответ превышает лимит локальной проверки"} for key in LABELS}, "score": None}
    parts = code_sections(answer)
    checks = {"original_stdout": stdout_check(parts.get("исходный вывод"), ORIGINAL), "fixed_stdout": stdout_check(parts.get("новый вывод"), FIXED)}
    unknown = {"status": "unknown", "detail": "Не найден однозначный блок исправления"}
    checks.update(contract=unknown, asserts=unknown)
    if "исправление" in parts:
        try:
            with tempfile.TemporaryDirectory(prefix="day05-verify-") as directory:
                run = subprocess.run(
                    [sys.executable, "-I", "-S", str(Path(__file__).with_name("verification_worker.py"))],
                    input=json.dumps({"function": parts["исправление"], "asserts": parts.get("проверки", "")}),
                    text=True, capture_output=True, timeout=2, cwd=directory, env={},
                )
            if run.returncode != 0:
                raise ValueError("worker failed")
            checks.update(json.loads(run.stdout))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            checks.update({key: {"status": "unknown", "detail": "Проверка недоступна или превышен лимит ресурсов"} for key in ("contract", "asserts")})
    score = sum(c["status"] == "pass" for c in checks.values()) if all(c["status"] != "unknown" for c in checks.values()) else None
    return {"checks": checks, "score": score}


def release_reference():
    items = {
        "A": (4, 3, 8), "B": (6, 5, 13), "C": (5, 4, 10),
        "D": (3, 2, 7), "E": (4, 3, 9), "F": (2, 2, 5),
    }

    def variants(budget_limit):
        names = tuple(items)
        valid = []
        for mask in range(1 << len(names)):
            chosen = tuple(name for bit, name in enumerate(names) if mask & (1 << bit))
            selected = set(chosen)
            budget = sum(items[name][0] for name in chosen)
            days = sum(items[name][1] for name in chosen)
            value = sum(items[name][2] for name in chosen)
            if budget > budget_limit or days > 12:
                continue
            if "B" in selected and "A" not in selected or "D" in selected and "B" not in selected:
                continue
            if {"C", "E"} <= selected:
                continue
            valid.append({"tasks": list(chosen), "budget": budget, "days": days, "value": value})
        return sorted(valid, key=lambda row: (-row["value"], row["budget"], row["days"], row["tasks"]))

    at_15 = variants(15)
    return {"best_15": at_15[0], "second_15": at_15[1], "best_12": variants(12)[0]}


def verify_release_answer(record):
    checks = {key: {"status": "unknown", "detail": "Ответ отсутствует или неполон"} for key in RELEASE_LABELS}
    if record.get("status") != "ok":
        return {"checks": checks, "score": None}
    parts = code_sections(record.get("answer", ""))
    block = parts.get("итог")
    if block is None:
        return {"checks": {key: {"status": "unknown", "detail": "Нет единственного JSON-блока в разделе Итог"} for key in RELEASE_LABELS}, "score": None}
    try:
        data = json.loads(block)
        actual = {
            "best_15": data["budget_15"]["best"],
            "second_15": data["budget_15"]["second"],
            "best_12": data["budget_12"]["best"],
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"checks": {key: {"status": "unknown", "detail": "JSON Итог не соответствует заданной структуре"} for key in RELEASE_LABELS}, "score": None}
    expected = release_reference()
    checks = {}
    for key in RELEASE_LABELS:
        checks[key] = {
            "status": "pass" if actual[key] == expected[key] else "fail",
            "actual": json.dumps(actual[key], ensure_ascii=False),
            "expected": json.dumps(expected[key], ensure_ascii=False),
        }
    return {"checks": checks, "score": sum(check["status"] == "pass" for check in checks.values())}


def verify_session(session):
    tasks = {}
    for prompt in session["prompts"]:
        digest = hashlib.sha256(prompt["text"].strip().encode()).hexdigest()
        if digest == PYTHON_PROMPT_HASH:
            checker, labels, maximum, verifier = PYTHON_CHECKER, LABELS, 4, verify_answer
        elif digest == RELEASE_PROMPT_HASH:
            checker, labels, maximum, verifier = RELEASE_CHECKER, RELEASE_LABELS, 3, verify_release_answer
        else:
            tasks[prompt["id"]] = {"status": "unsupported", "detail": "Для этого промпта нет независимого проверяющего модуля"}
            continue
        answers = {m["slot"]: verifier(session["results"][prompt["id"]].get(m["slot"], {})) for m in session["models"]}
        scores = [a["score"] for a in answers.values()]
        leaders = [slot for slot, a in answers.items() if a["score"] == max(scores)] if scores and all(s is not None for s in scores) else []
        tasks[prompt["id"]] = {"status": "checked", "checker": checker, "prompt_sha256": digest, "labels": labels, "max_score": maximum, "answers": answers, "leaders": leaders}
    return {"version": 1, "tasks": tasks}


def verification_markdown(session):
    lines = ["## Независимые проверки", "Проверки подтверждают только перечисленные формализованные пункты. Общий победитель автоматически не определяется: объяснения, полноту и итог оцениваем вручную по сохранённым ответам."]
    friendly = {
        "nvidia/nemotron-3.5-lightning:free": "Nemotron Lightning",
        "deepseek-v4-flash": "DeepSeek Flash",
        "deepseek-v4-pro": "DeepSeek Pro",
    }
    names = {m["slot"]: friendly.get(m["name"], m["name"]) for m in session["models"]}
    if any(record.get("status") != "ok" for results in session["results"].values() for record in results.values()):
        lines.append("Сравнение неполное: есть ошибка API или обрезанный ответ.")
    for index, prompt in enumerate(session["prompts"], 1):
        task = session.get("verification", {}).get("tasks", {}).get(prompt["id"], {})
        lines.append(f"### Задача {index}")
        if task.get("status") != "checked":
            lines.append("Независимая проверка отсутствует для этого промпта.")
            continue
        labels = task["labels"]
        maximum = task["max_score"]
        table = ["| Модель | " + " | ".join(labels.values()) + " | Итог |", "|---|" + "---|" * (len(labels) + 1)]
        for slot, answer in task["answers"].items():
            cells = []
            for key in labels:
                check = answer["checks"][key]
                label = {"pass": "OK", "fail": "Ошибка", "unknown": "Не проверено"}[check["status"]]
                if key == "asserts" and "total" in check:
                    label += f" ({check['passed']}/{check['total']})"
                cells.append(label)
            table.append("| " + " | ".join([names[slot], *cells, "—" if answer["score"] is None else f"{answer['score']}/{maximum}"]) + " |")
        lines.append("\n".join(table))
        leaders = task["leaders"]
        lines.append((f"Лидеры по {maximum} формализованным проверкам: " + ", ".join(names[s] for s in leaders) + ".") if leaders else "Рейтинг по проверкам недоступен: есть непроверенные ответы.")
        if task["checker"] == PYTHON_CHECKER:
            lines.append("Контракт проверяется на фиксированных случаях; успешные assert не доказывают полноту тестов.")
        else:
            lines.append("Перебор проверяет итоговые наборы и показатели; корректность объяснения оценивается вручную.")
        lines.append("Локальный итог не является общей оценкой качества модели.")
        for slot, answer in task["answers"].items():
            for key, check in answer["checks"].items():
                if check["status"] == "pass":
                    continue
                lines.append(f"**{names[slot]} · {labels[key]}:**")
                if "actual" in check:
                    lines.append(f"Получено:\n```text\n{check['actual']}\n```\nОжидается:\n```text\n{check['expected']}\n```")
                elif "failures" in check:
                    lines.append("Ошибочные проверки:\n```python\n" + "\n".join(check["failures"]) + "\n```")
                elif "cases" in check:
                    lines.extend(f"- {name}: {case['detail']}" for name, case in check["cases"].items() if case["status"] != "pass")
                else:
                    lines.append(check.get("detail", "Не проверено"))
    return "\n\n".join(lines)
