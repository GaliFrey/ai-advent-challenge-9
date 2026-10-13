"""Four explicit paid questions, once each; saves reports without secrets."""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from assistant import ask, save_result
from index import DAY, DEFAULT_PROJECT, index_path, load_model
from main import OVERVIEW

CASES = [
    ("structure", OVERVIEW, True, None),
    ("tests", "Какой командой запускать unit-тесты и какая версия Node.js нужна?", True, "docs/development.md"),
    ("audio", "Как YIN, Worker и AudioWorklet участвуют в распознавании высоты ноты?", True, "docs/audio.md"),
    ("unknown", "Какой точный годовой доход от платных подписок Guitar Trainer за 2025 год?", False, None),
]


def main():
    load_dotenv(DAY / ".env", override=False)
    parser = argparse.ArgumentParser(description="Приёмка: четыре реальных вызова DeepSeek, без повторов")
    parser.add_argument("--project", type=Path, default=Path(os.getenv("DAY31_PROJECT") or DEFAULT_PROJECT))
    args = parser.parse_args()
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("Нужен DEEPSEEK_API_KEY.")
    model = load_model()
    outcomes = []
    for name, question, known, required in CASES:
        result = ask(question, args.project, index_path(args.project), model, key,
                     os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
        path = save_result(result)
        answer = result.get("answer", {})
        used = {citation["source"] for citation in answer.get("citations", [])}
        cited_paths = {source["source"] for source in result["sources"] if source["id"] in used}
        passed = (result["status"] == "accepted" and answer.get("known") is known
                  and (required is None or required in cited_paths))
        outcomes.append({"case": name, "passed": passed, "report": str(path),
                         "status": result["status"], "known": answer.get("known")})
        print(json.dumps(outcomes[-1], ensure_ascii=False), flush=True)
    report = save_result({"cases": outcomes, "passed": all(row["passed"] for row in outcomes)})
    print(f"Сводка: {report}")
    return int(not all(row["passed"] for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
