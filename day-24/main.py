"""Live TUI and offline replay of grounded answers."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from evidence import answer_text
from judge import assessment_text
from pipeline import DAY, Runner, Settings, load_cases
from retrieval import Reranker


def main(argv=None):
    parser = argparse.ArgumentParser(description="День 24: источники, цитаты и отказ при слабом контексте")
    commands = parser.add_subparsers(dest="command", required=True)
    tui = commands.add_parser("tui", help="TUI: до 3 вызовов DeepSeek на вопрос, запуск по кнопке")
    tui.add_argument("report", nargs="?", type=Path)
    show = commands.add_parser("show", help="Просмотр сохранённого результата без сети")
    show.add_argument("report", type=Path)
    commands.add_parser("prepare", help="Подготовить модель reranker без DeepSeek")
    for name in ("ask", "benchmark"):
        live = commands.add_parser(name, help="До 3 вызовов на вопрос; benchmark — до 30")
        if name == "ask":
            live.add_argument("question")
        live.add_argument("--top-k-before", type=int, default=20)
        live.add_argument("--top-k-after", type=int, default=5)
        live.add_argument("--rerank-threshold", type=float, default=.1)
    args = parser.parse_args(argv)
    try:
        if args.command in ("tui", "show"):
            report = json.loads(args.report.read_text(encoding="utf-8")) if args.report else None
            if report and report.get("day") != 24:
                raise ValueError("Ожидался отчёт дня 24")
            if args.command == "tui":
                from tui import RagApp
                RagApp(report=report, report_path=args.report).run()
            else:
                if report.get("validation_only"):
                    print("ЛОКАЛЬНАЯ ПРОВЕРКА: синтетические ответы, качество генерации не оценивалось.")
                for item in report["items"]:
                    print(item["id"], item["question"], item["status"])
                    print(answer_text(item.get("response")))
                    print(json.dumps(item.get("response", {}).get("sources", []) if item.get("response") else [], ensure_ascii=False))
                    if item.get("response"):
                        for claim in item["response"]["answer"]:
                            for citation in claim["citations"]:
                                print(citation["source_id"], citation["quote"])
                    print(json.dumps(item.get("checks", {}), ensure_ascii=False))
                    print(assessment_text(item))
            return 0
        if args.command == "prepare":
            Reranker()
            print("Reranker готов; кэш используется из дня 23.")
            return 0
        load_dotenv(DAY / ".env", override=False)
        settings = Settings(args.top_k_before, args.top_k_after, args.rerank_threshold)
        cases = load_cases() if args.command == "benchmark" else [{"id": "custom", "question": args.question}]
        path = DAY / "resources" / (args.command + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
        print(f"До {len(cases) * 3} вызовов DeepSeek; результат: {path}")
        report = Runner(key=os.getenv("DEEPSEEK_API_KEY", ""), model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), settings=settings).run(cases, path)
        print(report["status"])
        return 0 if report["status"] == "complete" else 1
    except Exception as error:
        print(f"Ошибка: {type(error).__name__}. Проверьте настройки, аргументы и журнал.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
