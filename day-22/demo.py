"""Five recording screens, rendered only from a saved, reviewed benchmark."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

TITLES = (
    "Что проверяем",
    "Что поиск передал модели",
    "Один вопрос — два ответа",
    "Почему RAG отвечает не всегда",
    "Результат на 10 вопросах",
)
LABELS = ("Лимит и stop-маркер", "Интервал summary", "Слои памяти",
          "Индекс Datex", "Проверка отчёта", "Стратегии контекста",
          "Память дней 9 и 11", "MCP дней 19 и 20", "p95: данных нет", "SLA: данных нет")


def validate_report(report: dict) -> None:
    items = report.get("items", [])
    if report.get("status") != "complete" or {item.get("id") for item in items} != {
            f"q{i:02}" for i in range(1, 11)} or len(items) != 10:
        raise ValueError("Для demo нужен завершённый benchmark из 10 контрольных вопросов")
    for item in items:
        for mode in ("plain", "rag"):
            result = item.get("results", {}).get(mode, {})
            if not result.get("answer") or not result.get("review"):
                raise ValueError("Для demo нужны оба ответа и ручные оценки всех 10 вопросов")
            if result["review"].get("quality") not in (0, 1, 2):
                raise ValueError("Оценки demo должны быть от 0 до 2")


def load_report(path: Path | None, directory: Path) -> tuple[Path, dict]:
    if path is not None:
        report = json.loads(path.read_text(encoding="utf-8"))
        validate_report(report)
        return path, report
    for candidate in sorted(directory.glob("benchmark-*.json"), reverse=True):
        try:
            report = json.loads(candidate.read_text(encoding="utf-8"))
            validate_report(report)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        return candidate, report
    raise ValueError("Нет завершённого benchmark с оценками. Выполните benchmark и review, затем demo")


def pair(left, right):
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(left, right)
    return table


def source_table(chunks: list[dict]):
    table = Table(expand=True, padding=(0, 1), show_edge=False)
    table.add_column("Ссылка", width=7)
    table.add_column("Источник", width=18)
    table.add_column("Сходство", ratio=1)
    for i, chunk in enumerate(chunks, 1):
        table.add_row(Text(f"S{i}"), Text(chunk["source"]), Text(str(chunk["score"])))
    return table


def answer_panel(result: dict, title: str, color: str):
    review = result["review"]
    answer = Text(result["answer"])
    answer.highlight_regex(r"`[^`]+`|\[S\d+\]", f"bold {color}")
    return Panel(Group(answer, Text(""),
                       Text(f"Оценка полноты: {review['quality']}/2", style=f"bold {color}")),
                 title=title, border_style=color, padding=(1, 1))


def pages(report: dict, *, headers: bool = True):
    items = {item["id"]: item for item in report["items"]}
    example, limitation = items["q01"], items["q05"]
    model = report["model"]
    intro = Group(
        Text("RAG даёт модели сведения из наших документов", style="bold cyan"), Text(""),
        pair(Panel("Вопрос → LLM → ответ\n\nМодель не получает README проекта.",
                   title="БЕЗ RAG", border_style="yellow", padding=(1, 1)),
             Panel("Вопрос → поиск → фрагменты + вопрос → LLM\n\nОтвет опирается на найденный контекст.",
                   title="С RAG", border_style="cyan", padding=(1, 1))),
        Text(""), Panel(
            f"День 21: документы → чанки → эмбеддинги → индекс\n"
            f"День 22: поиск в этом индексе → контекст → ответ LLM\n\n"
            f"База: {report['index']['document_count']} README, дни 0–20. "
            f"Поиск: {report['strategy']}, top-{report['top_k']}.", title="Продолжаем предыдущий день"),
        Text(f"Одинаковая модель: {model}. Температура: {report['parameters']['temperature']}. "
             "Истории между запросами нет."),
        Text("10 заранее заданных вопросов: 6 о фактах, 2 на сравнение, 2 без ответа в базе."),
        Text("На экране — сохранённый реальный прогон, без новых обращений к API.", style="dim"),
    )
    rag = example["results"]["rag"]
    cited = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", rag["answer"])
                    if 1 <= int(n) <= len(rag["chunks"])})
    selected = cited[-2:] or list(range(1, min(2, len(rag["chunks"])) + 1))
    excerpts = []
    for number in selected:
        chunk = rag["chunks"][number - 1]
        body = chunk["text"]
        # Show a verbatim paragraph with the relevant fact, not Markdown scaffolding.
        paragraphs = body.split("\n\n")
        focus = next((p for p in paragraphs if "max_tokens" in p or "Поэтому" in p), body)
        if "Поэтому" in focus:
            focus = focus[focus.index("Поэтому"):]
        # The preview is shortened only here; the LLM received the full saved chunk.
        excerpt = focus if len(focus) <= 400 else focus[:400] + "\n… [сокращено для экрана]"
        excerpts.append(Panel(Text(excerpt), title=f"S{number} · {chunk['source']}", border_style="cyan"))
    context = Group(
        Text(example["question"], style="bold"), Text(""), source_table(rag["chunks"]),
        Text(""), Text("Выдержки из фактически переданных фрагментов:", style="bold cyan"),
        *excerpts,
        Text("LLM получает вопрос и все найденные фрагменты с метками S1…S5. "
             "Ожидания и эталонные ответы ей не передаются.", style="dim"),
    )
    comparison = Group(
        Text(example["question"], style="bold"), Text(""),
        pair(answer_panel(example["results"]["plain"], "БЕЗ RAG", "yellow"),
             answer_panel(rag, "С RAG", "cyan")),
        Text(""), Panel(Text(rag["review"]["note"]), title="Что показало сравнение", border_style="green"),
        Text("[S…] в ответе — ссылки на фрагменты предыдущего экрана."),
        Text("0 баллов за полноту не означает выдумку: честное «не знаю» тоже получает 0, "
             "если ответ есть в базе.", style="dim"),
    )
    result = limitation["results"]["rag"]
    limits = Group(
        Text(limitation["question"], style="bold"), Text(""),
        Text("Найденные источники:", style="bold"), source_table(result["chunks"]),
        Text(""), pair(
            Panel(Text("\n".join(f"• {fact}" for fact in limitation["expected"])),
                  title="Что должно быть в ответе", border_style="yellow"),
            answer_panel(result, "ФАКТИЧЕСКИЙ ОТВЕТ С RAG", "cyan")),
        Panel(Text(result["review"]["note"]), title="Проверка найденного контекста", border_style="yellow"),
        Text("Нужный файл в top-5 ≠ нужный фрагмент с ответом.", style="bold yellow"),
    )
    table = Table(expand=True, show_edge=False, padding=(0, 1))
    for name in ("Вопрос", "Без RAG / 2", "С RAG / 2"):
        table.add_column(name)
    scores = {mode: 0 for mode in ("plain", "rag")}
    tokens = {mode: 0 for mode in scores}
    for i, label in enumerate(LABELS, 1):
        item = items[f"q{i:02}"]
        row = []
        for mode in scores:
            value = item["results"][mode]["review"]["quality"]
            scores[mode] += value
            tokens[mode] += item["results"][mode]["usage"]["total_tokens"]
            row.append(str(value))
        table.add_row(f"{i:02}  {label}", *row)
    table.add_row("ИТОГО", f"{scores['plain']}/20", f"{scores['rag']}/20", style="bold")
    full = sum(items[f"q{i:02}"]["results"]["rag"]["review"]["quality"] == 2 for i in range(1, 9))
    partial = sum(items[f"q{i:02}"]["results"]["rag"]["review"]["quality"] == 1 for i in range(1, 9))
    unknown = {mode: sum(items[q]["results"][mode]["review"]["quality"] == 2
                        for q in ("q09", "q10")) for mode in scores}
    summary = Group(
        table, Text(""),
        Text(f"Из 8 вопросов с ответом в базе RAG: полных ответов — {full}, частичных — {partial}.", style="bold cyan"),
        Text(f"2 вопроса без ответа: корректных отказов без RAG — {unknown['plain']}/2, с RAG — {unknown['rag']}/2."),
        Text(f"Токены за 10 ответов: без RAG — {tokens['plain']:,}; с RAG — {tokens['rag']:,}."),
        Text(""), Panel("RAG помогает, когда поиск приносит нужные факты.\n"
                         "Качество ответа зависит от полноты найденных фрагментов.",
                         title="Главный вывод", border_style="green"),
        Text("Оценка 0–2: нет ответа / частичный / полный. Оценки проверены по сохранённым данным."),
        Text("Один прогон, 10 вопросов. Это оценка ассистента, а не независимый тест качества модели.", style="dim"),
    )
    for i, body in enumerate((intro, context, comparison, limits, summary), 1):
        yield Group(Text(f"ДЕНЬ 22 · ПЕРВЫЙ RAG     {i}/5", style="bold cyan"),
                    Text(TITLES[i - 1], style="bold"),
                    Text(f"Сохранённый прогон: {report['created_at'][:19]} UTC · {model}", style="dim"),
                    Text(""), body) if headers else body


def page_height(page, width: int) -> int:
    stream = io.StringIO()
    Console(file=stream, width=width).print(page)
    return len(stream.getvalue().splitlines())


def run_demo(report: dict, console: Console, *, page: int | None = None) -> None:
    validate_report(report)
    screens = list(pages(report))
    if page is not None:
        console.print(screens[page - 1])
        return
    if not console.is_terminal:
        raise ValueError("Для пошагового demo нужен терминал; для текстового просмотра используйте --page 1…5")
    width = min(console.size.width, 120)
    required = max(page_height(screen, width) for screen in screens) + 3
    if width < 100 or console.size.height < required:
        raise ValueError(f"Разверните терминал или уменьшите шрифт: нужно минимум 100 столбцов "
                         f"и {required} строк при текущей ширине. Сейчас {console.size.width}×{console.size.height}")
    screen_console = Console(width=width, file=console.file)
    current = 0
    while True:
        console.clear()
        screen_console.print(screens[current])
        prompt = "Enter — завершить" if current == len(screens) - 1 else "Enter — дальше"
        try:
            command = screen_console.input(f"\n[dim]{prompt} · b — назад · q — выход: [/dim]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            screen_console.print()
            return
        if command == "q":
            return
        if command == "b":
            current = max(0, current - 1)
        elif not command:
            if current == len(screens) - 1:
                return
            current += 1
