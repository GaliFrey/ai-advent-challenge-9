"""CLI for live RAG comparisons and offline replay of saved evidence."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag import PARAMETERS, answer_question
from retrieval import DEFAULT_INDEX, STRATEGY, TOP_K, Retriever

DAY = Path(__file__).resolve().parent
QUESTIONS = DAY / "questions.json"


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".rag-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def display_item(item: dict, console: Console, *, chunks: bool = False) -> None:
    console.print(Text(f"{item['id']}: {item['question']}", style="bold"))
    for mode, result in item["results"].items():
        console.print(Panel(Text(result.get("answer", result.get("error", "Нет ответа"))),
                            title="С RAG" if mode == "rag" else "Без RAG"))
        if "answer" not in result:
            continue
        console.print(Text(f"LLM: {result['llm_seconds']} с; поиск: {result['search_seconds']} с; "
                           f"токены: {result['usage'].get('total_tokens', 'нет данных')}"))
        for i, chunk in enumerate(result["chunks"], 1):
            console.print(Text(f"[S{i}] {chunk['source']} | {chunk['section']} | "
                               f"score={chunk['score']} | {chunk['chunk_id']}"))
            if chunks:
                console.print(Panel(Text(chunk["text"])))
        invalid = result["citation_check"]["unknown_references"]
        if invalid:
            console.print(Text(f"Несуществующие ссылки: {', '.join(invalid)}", style="red"))
        review = result.get("review")
        if review:
            console.print(Text(f"Ручная оценка: {review['quality']}/2. {review['note']}"))


def display_summary(report: dict, console: Console) -> None:
    table = Table(title="Сравнение: ручная оценка 0–2; — означает, что ответ ещё не оценён")
    for column in ("Вопрос", "Без RAG", "С RAG", "Нужные файлы в top-5"):
        table.add_column(column)
    totals = {mode: [] for mode in ("plain", "rag")}
    for item in report["items"]:
        cells = []
        for mode in totals:
            result = item["results"].get(mode, {})
            review = result.get("review")
            cells.append(str(review["quality"]) if review else ("ошибка" if "error" in result else "—"))
            if review:
                totals[mode].append(review["quality"])
        expected = set(item.get("sources", []))
        found = {chunk["source"] for chunk in item["results"].get("rag", {}).get("chunks", [])}
        coverage = f"{len(expected & found)}/{len(expected)}" if expected else "н/п"
        table.add_row(item["id"], *cells, coverage)
    console.print(table)
    for mode, scores in totals.items():
        if scores:
            console.print(Text(f"{mode}: {sum(scores)}/{2 * len(scores)} баллов, оценено {len(scores)} ответов"))
    console.print(Text("Попадание файла не доказывает, что найден полный ответ. Проверка номеров ссылок "
                       "не проверяет обоснованность утверждений. Оценки заполняются по сохранённым ответам."))


def run_cases(cases, modes, *, path, model, key, retriever, console) -> dict:
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "model": model, "parameters": PARAMETERS, "strategy": STRATEGY, "top_k": TOP_K,
              "index": retriever.metadata if retriever else None, "status": "running", "items": []}
    save_report(path, report)
    for case in cases:
        item = {**case, "results": {}}
        report["items"].append(item)
        for mode in modes:
            console.print(Text(f"{case['id']} · {mode}: запрос к LLM…"))
            try:
                item["results"][mode] = answer_question(case["question"], mode, key=key,
                                                        model=model, retriever=retriever)
            except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
                item["results"][mode] = {"error": str(error)}
                report["status"] = "failed"
                save_report(path, report)
                raise
            save_report(path, report)
        display_item(item, console)
    report["status"] = "complete"
    save_report(path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Первый RAG: поиск → контекст → ответ LLM")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Один вопрос, 1 или 2 LLM-запроса")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=("plain", "rag", "compare"), default="compare")
    benchmark = sub.add_parser("benchmark", help="10 вопросов × 2 режима = 20 LLM-запросов")
    demo = sub.add_parser("demo", help="Пять экранов для видео по сохранённому прогону, без API")
    demo.add_argument("report", nargs="?", type=Path, help="По умолчанию последний завершённый benchmark с оценками")
    demo.add_argument("--page", type=int, choices=range(1, 6), help="Показать один экран без ожидания Enter")
    tui = sub.add_parser("tui", help="Живое сравнение и демо на 10 вопросов (20 API-запросов)")
    tui.add_argument("report", nargs="?", type=Path, help="Необязательно: открыть сохранённый результат")
    for live in (ask, benchmark):
        live.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    show = sub.add_parser("show", help="Показать сохранённый результат без сети и модели эмбеддингов")
    show.add_argument("report", type=Path)
    show.add_argument("--question", help="Показать один вопрос, например q05")
    show.add_argument("--chunks", action="store_true", help="Показать полные тексты найденных фрагментов")
    show.add_argument("--summary-only", action="store_true", help="Только итоговая таблица")
    review = sub.add_parser("review", help="Сохранить ручную оценку ответа без LLM-вызовов")
    review.add_argument("report", type=Path)
    review.add_argument("--question", required=True)
    review.add_argument("--mode", choices=("plain", "rag"), required=True)
    review.add_argument("--quality", type=int, choices=(0, 1, 2), required=True)
    review.add_argument("--unsupported", choices=("yes", "no"), required=True)
    review.add_argument("--citations", choices=("pass", "fail", "na"), required=True)
    review.add_argument("--retrieval", choices=("full", "partial", "missing", "na"), required=True)
    review.add_argument("--abstention", choices=("correct", "incorrect", "na"), required=True)
    review.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    console = Console()
    try:
        if args.command == "tui":
            from tui import RagApp
            report = json.loads(args.report.read_text(encoding="utf-8")) if args.report else None
            RagApp(report).run()
            return 0
        if args.command == "demo":
            from demo import load_report, run_demo
            _path, report = load_report(args.report, DAY / "resources")
            run_demo(report, console, page=args.page)
            return 0
        if args.command == "review":
            report = json.loads(args.report.read_text(encoding="utf-8"))
            item = next((item for item in report["items"] if item["id"] == args.question), None)
            if item is None or "answer" not in item["results"].get(args.mode, {}):
                raise ValueError("Нет сохранённого ответа для этой оценки")
            if not args.note.strip():
                raise ValueError("Нужно обоснование ручной оценки")
            item["results"][args.mode]["review"] = {
                "quality": args.quality, "unsupported": args.unsupported == "yes",
                "citations": args.citations, "retrieval": args.retrieval,
                "abstention": args.abstention, "note": args.note,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
            save_report(args.report, report)
            display_summary(report, console)
            return 0
        if args.command == "show":
            report = json.loads(args.report.read_text(encoding="utf-8"))
            if args.question and not any(item["id"] == args.question for item in report["items"]):
                raise ValueError("Такого вопроса нет в отчёте")
            console.print(Text(f"Сохранённый прогон: {report['created_at']}; статус: {report['status']}"))
            for item in report["items"]:
                if not args.summary_only and (not args.question or item["id"] == args.question):
                    display_item(item, console, chunks=args.chunks)
            display_summary(report, console)
            return 0
        load_dotenv(DAY / ".env", override=False)
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise ValueError("Задайте DEEPSEEK_API_KEY в day-22/.env или окружении")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        if args.command == "ask" and not args.question.strip():
            raise ValueError("Вопрос не должен быть пустым")
        modes = ["plain", "rag"] if args.command == "benchmark" or args.mode == "compare" else [args.mode]
        cases = json.loads(QUESTIONS.read_text(encoding="utf-8")) if args.command == "benchmark" else [
            {"id": "custom", "question": args.question, "expected": [], "sources": []}]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = DAY / "resources" / f"{args.command}-{stamp}.json"
        console.print(Text(f"Результат: {path}"))
        console.print(Text(f"Модель: {model}; вызовов: {len(cases) * len(modes)}; "
                           "каждый вызов независим, автоматических повторов нет"))
        retriever = Retriever(args.index) if "rag" in modes else None
        report = run_cases(cases, modes, path=path, model=model, key=key,
                           retriever=retriever, console=console)
        display_summary(report, console)
        return 0
    except (RuntimeError, ValueError, OSError, sqlite3.Error, KeyError, TypeError) as error:
        console.print(Text(f"Ошибка: {error}", style="red"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
