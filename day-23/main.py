"""Live TUI, controlled benchmark and saved pipeline comparisons."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pipeline import DAY, LABELS, MODES, Runner, Settings
from retrieval import DEFAULT_INDEX, Reranker
from session_log import SessionLog


def load_cases():
    return json.loads((DAY / "questions.json").read_text(encoding="utf-8"))


def summary(report):
    table = Table(title="Состояние режимов")
    table.add_column("Вопрос")
    for mode in MODES:
        table.add_column(LABELS[mode])
    for item in report["items"]:
        cells = []
        for mode in MODES:
            result = item["results"].get(mode, {})
            status = result.get("status", "—")
            cells.append({'complete': 'Готово', 'running': 'В работе', 'failed': 'Ошибка'}.get(status, 'Не запущен'))
        table.add_row(item["id"], *cells)
    return table


def comparison_table(item):
    table = Table(title="Сравнение выбранного вопроса", expand=True)
    for name in ("Режим", "Статус", "Контекст / кандидаты", "Токены ответа", "LLM, с", "Неверные ссылки"):
        table.add_column(name)
    for mode in MODES:
        result = item['results'].get(mode, {}) if item else {}
        status = {'complete': 'Готово', 'running': 'В работе', 'failed': 'Ошибка'}.get(result.get('status'), 'Не запущен')
        invalid = result.get('citation_check', {}).get('unknown_references')
        table.add_row(LABELS[mode], status,
                      f"{len(result.get('chunks', []))} / {len(result.get('candidates', []))}",
                      str(result.get('usage', {}).get('total_tokens', '—')),
                      str(result.get('llm_seconds', '—')),
                      ', '.join(invalid) if invalid else ('Нет' if 'answer' in result else '—'))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description="День 23: четыре режима RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    tui = sub.add_parser("tui", help="Полноэкранное сравнение; запуск только по кнопке")
    tui.add_argument("report", nargs="?", type=Path)
    for name in ("ask", "benchmark"):
        live = sub.add_parser(name, help="5 LLM-вызовов на вопрос: rewrite + 4 ответа")
        if name == "ask":
            live.add_argument("question")
        live.add_argument("--top-k-before", type=int, default=20)
        live.add_argument("--top-k-after", type=int, default=5)
        live.add_argument("--similarity-threshold", type=float, default=0.3)
        live.add_argument("--rerank-threshold", type=float, default=0.1)
        live.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    sub.add_parser("prepare", help="Загрузить локальную модель reranker, без DeepSeek")
    show = sub.add_parser("show", help="Показать сохранённое сравнение без сети")
    show.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    console = Console()
    log = None
    try:
        if args.command == "tui":
            from tui import RagApp
            report = json.loads(args.report.read_text(encoding="utf-8")) if args.report else None
            RagApp(report=report).run()
            return 0
        if args.command == "show":
            report = json.loads(args.report.read_text(encoding="utf-8"))
            for item in report["items"]:
                console.print(Text(item["question"], style="bold"))
                for mode, result in item["results"].items():
                    console.print(Text(f"{LABELS[mode]}\nПоиск: {result['query']}\n{result.get('answer', result['status'])}\n"))
                console.print(comparison_table(item))
            console.print(summary(report))
            return 0
        log = SessionLog()
        console.print(Text(f"Журнал: {log.path}"))
        if args.command == "prepare":
            with log.stage("reranker_model"):
                Reranker()
            console.print("Модель reranker готова.")
            return 0
        load_dotenv(DAY / ".env", override=False)
        settings = Settings(args.top_k_before, args.top_k_after, args.similarity_threshold, args.rerank_threshold)
        cases = load_cases() if args.command == "benchmark" else [
            {"id": "custom", "question": args.question, "expected": [], "sources": []}]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = DAY / "resources" / f"{args.command}-{stamp}.json"
        console.print(Text(f"Вызовов DeepSeek: {len(cases) * 5}; результат: {path}"))
        runner = Runner(key=os.getenv("DEEPSEEK_API_KEY", ""), model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                        settings=settings, index=args.index)
        report = runner.run(cases, path, log)
        console.print(summary(report))
        if report["status"] == "failed":
            console.print(Text(f"Ошибка: {report['error']}; подробности в {log.path}", style="red"))
            return 1
        return 0
    except Exception as error:
        console.print(Text(f"Ошибка: {type(error).__name__}. " +
                           (f"Журнал: {log.path}" if log else "Проверьте аргументы и настройки."), style="red"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
