"""Wide terminal workspace showing each stage and every retrieval candidate."""
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from multiprocessing import RLock
from threading import Event

from dotenv import load_dotenv
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Input, Label, LoadingIndicator, Select, Static, TabbedContent, TabPane

from main import comparison_table, load_cases
from pipeline import DAY, LABELS, MODES, Runner, Settings
from session_log import SessionLog

STAGE_LABELS = {"index": "Проверка индекса", "embedding_model": "Модель поиска",
                "reranker_model": "Модель reranker", "rewrite": "Перефразирование",
                "search": "Поиск", "filter": "Отбор", "rerank": "Реранкинг", "answer": "Ответ"}
STATUS = {"running": "…", "complete": "✓", "shared": "↗ общий результат", "failed": "✗"}
ACTIVITY = {"index": "Проверяется индекс", "embedding_model": "Загружается модель эмбеддингов",
            "reranker_model": "Загружается локальный reranker", "rewrite": "DeepSeek перефразирует вопрос",
            "search": "Модель эмбеддингов ищет чанки", "filter": "Отбираются чанки",
            "rerank": "Reranker оценивает чанки", "answer": "DeepSeek формирует ответ"}


class RagApp(App):
    TITLE = "День 23 — Поиск, rewrite, фильтрация и reranker"
    CSS_PATH = "tui.tcss"
    BINDINGS = [("ctrl+q", "quit", "Выход")]

    def __init__(self, report=None, *, cases=None, runner_factory=Runner, log_directory=None):
        self._model_lock = RLock()  # Start Python's resource tracker before opening the TUI.
        super().__init__()
        self.cases = cases or load_cases()
        self.report = report
        if report:
            known = {case["id"] for case in self.cases}
            self.cases += [{key: value for key, value in item.items() if key not in ("results", "rewrite")}
                           for item in report["items"] if item["id"] not in known]
        self.selected_id = report["items"][0]["id"] if report and report.get("items") else self.cases[0]["id"]
        self.runner_factory = runner_factory
        self.session_log = SessionLog(log_directory) if log_directory else SessionLog()
        self.running, self.ready = False, False
        self.report_path = None
        self.stop_requested = Event()
        self.status_message = ""
        self.activity_started = None
        self.displayed = {}
        self.selected_chunks = {}
        self.theme = "textual-dark"
        load_dotenv(DAY / ".env", override=False)

    def compose(self) -> ComposeResult:
        yield Static("ДЕНЬ 23  /  ЧЕТЫРЕ РЕЖИМА RAG", id="brand")
        with Horizontal(id="activity"):
            yield LoadingIndicator(id="busy")
            yield Static(id="status")
        with Horizontal(id="controls"):
            yield Select([(f"{case['id']} · {case['question']}", case["id"]) for case in self.cases],
                         value=self.selected_id, allow_blank=False, id="questions")
            yield Button("Сравнить · 5 вызовов", id="ask", variant="primary")
            yield Button("Демо · 50 вызовов", id="demo")
        with Horizontal(id="custom-controls"):
            yield Input(placeholder="Свой вопрос — оставьте пустым для выбранного вопроса", id="custom")
        settings = Settings(**self.report["settings"]) if self.report else Settings()
        with Horizontal(id="settings"):
            for name, label, value in (("before", "Кандидаты K", settings.top_k_before),
                                       ("after", "Итоговый K", settings.top_k_after),
                                       ("similarity", "Порог similarity", settings.similarity_threshold),
                                       ("rerank", "Порог reranker", settings.rerank_threshold)):
                yield Label(label)
                yield Input(str(value), id=name)
        yield Static(id="setup")
        yield Static(id="question")
        with TabbedContent(id="modes"):
            for mode in MODES:
                with TabPane(LABELS[mode], id=f"pane-{mode}"):
                    yield Static(id=f"flow-{mode}", classes="flow")
                    with Horizontal(classes="workspace"):
                        with Vertical(classes="evidence"):
                            with VerticalScroll(classes="queries"):
                                yield Static(id=f"query-{mode}")
                            yield DataTable(id=f"chunks-{mode}", cursor_type="row", zebra_stripes=True)
                            with VerticalScroll(classes="chunk-view"):
                                yield Static(id=f"chunk-text-{mode}")
                        with VerticalScroll(classes="answer-view"):
                            yield Label("КОНЕЧНЫЙ ОТВЕТ", classes="box-title")
                            yield Static(id=f"answer-{mode}")
                            yield Static(id=f"metrics-{mode}", classes="metrics")
            with TabPane("Сравнение", id="pane-summary"):
                with Vertical(id="comparison"):
                    yield Static(id="summary")
                    yield Static(id="comparison-note")
                    with Horizontal(id="comparison-answers"):
                        for mode in MODES:
                            with VerticalScroll(classes="comparison-card"):
                                yield Label(LABELS[mode], classes="box-title")
                                yield Static(id=f"compare-answer-{mode}")
                                yield Static(id=f"compare-sources-{mode}", classes="metrics")
        yield Footer()

    def on_mount(self):
        for mode in MODES:
            table = self.query_one(f"#chunks-{mode}", DataTable)
            for title, width in (("Поиск", 5), ("Ранг", 5), ("Источник / раздел", 32),
                                 ("Similarity", 10), ("Reranker", 8), ("Отбор", 30)):
                table.add_column(title, width=width)
        self.ready = True
        self.query_one("#busy", LoadingIndicator).display = False
        key_ready = bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
        self.set_status("Ключ DeepSeek настроен. Выберите вопрос и запустите сравнение." if key_ready else
                        "НЕТ КЛЮЧА DEEPSEEK_API_KEY: заполните day-23/.env и перезапустите приложение.")
        self.set_interval(0.25, self.refresh_status)
        self.render_results()

    def set_status(self, message):
        self.status_message = message
        self.refresh_status()

    def refresh_status(self):
        elapsed = f" · прошло {time.perf_counter() - self.activity_started:.1f} с" if self.running and self.activity_started else ""
        self.query_one("#status", Static).update(Text(
            f"{self.status_message}{elapsed}\nЖурнал: logs/{self.session_log.path.name}"))

    def launch_error(self, error):
        try:
            with self.session_log.stage("configuration"):
                raise error
        except ValueError:
            pass
        self.set_status(f"ЗАПУСК ОСТАНОВЛЕН: {error}")
        self.notify(str(error), title="Не удалось запустить RAG", severity="error", timeout=15)

    def current_item(self):
        return next((item for item in (self.report or {}).get("items", []) if item["id"] == self.selected_id), None)

    @on(Select.Changed, "#questions")
    def change_question(self, event):
        if self.ready:
            self.selected_id = str(event.value)
            self.render_results()

    @on(DataTable.RowHighlighted)
    def show_chunk(self, event):
        if not self.ready:
            return
        mode = event.data_table.id.removeprefix("chunks-")
        rows = self.displayed.get(mode, [])
        index = event.cursor_row
        targets = self.query(f"#chunk-text-{mode}")
        if targets and 0 <= index < len(rows):
            row = rows[index]
            self.selected_chunks[mode] = row["chunk_id"]
            targets.first(Static).update(Text(
                f"{row.get('source_id') or 'Не передан модели'} · {row['source']} · {row['section']}\n"
                f"ID: {row['chunk_id']}\n\n{row['text']}"))

    def render_results(self):
        if not self.ready:
            return
        item = self.current_item()
        case = next(case for case in self.cases if case["id"] == self.selected_id)
        self.query_one("#question", Static).update(Text(case["question"], style="bold"))
        setup = (self.report or {}).get("setup", [])
        self.query_one("#setup", Static).update(Text(self.flow_text(setup)))
        for mode in MODES:
            result = item["results"].get(mode, {}) if item else {}
            self.query_one(f"#flow-{mode}", Static).update(Text(self.flow_text(result.get("steps", [])) or
                ("Вопрос → поиск → top-K → ответ" if mode == "baseline" else
                 "Вопрос → rewrite → поиск → " + ("reranker → " if mode == "rewrite_rerank" else "") + "отбор → ответ")))
            query = result.get("query", "Появится после запуска")
            rewrite = item.get("rewrite") if item else None
            rewrite_status = ""
            if mode != "baseline" and rewrite:
                changed = " ".join(rewrite["query"].split()).casefold() != " ".join(item["question"].split()).casefold()
                rewrite_status = "\nREWRITE: запрос изменён" if changed else "\nREWRITE: запрос не изменён"
            self.query_one(f"#query-{mode}", Static).update(Text(
                f"ИСХОДНЫЙ ВОПРОС\n{case['question']}\n\nПОИСКОВЫЙ ЗАПРОС\n{query}{rewrite_status}"))
            rows = result.get("candidates", [])
            table = self.query_one(f"#chunks-{mode}", DataTable)
            if rows != self.displayed.get(mode):
                self.displayed[mode] = [dict(row) for row in rows]
                table.clear()
                for row in rows:
                    table.add_row(str(row["rank_before"]), str(row.get("rank_after", "—")),
                                  Text(f"{row['source']} · {row['section']}"),
                                  f"{row['score']:.3f}", f"{row['rerank_score']:.3f}" if "rerank_score" in row else "—",
                                  Text(f"{row.get('source_id') or ''} {row.get('decision', 'Кандидат')}"))
                if rows:
                    index = next((i for i, row in enumerate(rows) if row["chunk_id"] == self.selected_chunks.get(mode)), 0)
                    table.move_cursor(row=index)
                    row = rows[index]
                    self.query_one(f"#chunk-text-{mode}", Static).update(Text(
                        f"{row.get('source_id') or 'Не передан модели'} · {row['source']} · {row['section']}\n"
                        f"ID: {row['chunk_id']}\n\n{row['text']}"))
                else:
                    self.query_one(f"#chunk-text-{mode}", Static).update(Text("Нет чанков. Пустой контекст допустим."))
            answer = result.get("answer", "Ошибка этапа; смотрите журнал." if result.get("status") == "failed" else "Ответ появится после выполнения этапов.")
            self.query_one(f"#answer-{mode}", Static).update(Text(answer))
            self.query_one(f"#compare-answer-{mode}", Static).update(Text(answer))
            selected = result.get("chunks", [])
            sources = "\n".join(f"[S{i}] {row['source']} · {row['section']}"
                                for i, row in enumerate(selected, 1))
            self.query_one(f"#compare-sources-{mode}", Static).update(Text(
                f"ИСТОЧНИКИ ЭТОГО ОТВЕТА\n{sources or 'Нет источников'}"))
            metrics = ""
            if "answer" in result:
                usage = result.get("usage", {}).get("total_tokens", "—")
                metrics = (f"Контекст: {len(result['chunks'])} чанков из {len(rows)}\n"
                           f"Ответ: {usage} токенов · LLM {result['llm_seconds']} с\n"
                           f"Этапы режима: {result['elapsed_seconds']} с")
                if mode != "baseline" and item.get("rewrite"):
                    rewrite = item["rewrite"]
                    metrics += f"\nОбщий rewrite: {rewrite['usage'].get('total_tokens', '—')} токенов · {rewrite['llm_seconds']} с"
                invalid = result["citation_check"]["unknown_references"]
                if invalid:
                    metrics += f"\nНесуществующие ссылки: {', '.join(invalid)}"
            self.query_one(f"#metrics-{mode}", Static).update(Text(metrics))
        self.query_one("#summary", Static).update(comparison_table(item))
        rewrite = item.get("rewrite") if item else None
        note = "Сравните ответы и их источники. Путь поиска и отбор чанков доступны во вкладке каждого режима."
        if rewrite:
            changed = " ".join(rewrite['query'].split()).casefold() != " ".join(item['question'].split()).casefold()
            note += (f"\nОбщий rewrite: {rewrite['usage'].get('total_tokens', '—')} токенов, "
                     f"{rewrite['llm_seconds']} с · запрос {'изменён' if changed else 'не изменён'}. "
                     f"Поисковый запрос: {rewrite['query']}")
        self.query_one("#comparison-note", Static).update(Text(note))

    @staticmethod
    def flow_text(steps):
        return " → ".join(f"{STAGE_LABELS.get(step['stage'], step['stage'])} {STATUS[step['status']]}" +
                           (f" {step['seconds']}с" if step.get("seconds") is not None else "") for step in steps)

    @on(Button.Pressed)
    def launch(self, event):
        if self.running:
            return
        try:
            settings = Settings(int(self.query_one("#before", Input).value), int(self.query_one("#after", Input).value),
                                float(self.query_one("#similarity", Input).value), float(self.query_one("#rerank", Input).value))
        except ValueError as error:
            self.launch_error(error)
            return
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            self.launch_error(ValueError("Задайте DEEPSEEK_API_KEY в day-23/.env и перезапустите приложение."))
            return
        if event.button.id == "demo":
            cases = load_cases()
        else:
            question = self.query_one("#custom", Input).value.strip()
            if question:
                case = {"id": "custom", "question": question, "expected": [], "sources": []}
                self.cases = [case] + [case for case in self.cases if case["id"] != "custom"]
                select = self.query_one("#questions", Select)
                select.set_options([(f"{case['id']} · {case['question']}", case["id"]) for case in self.cases])
                select.value = "custom"
                self.selected_id = "custom"
            cases = [next(case for case in self.cases if case["id"] == self.selected_id)]
        self.running = True
        self.activity_started = time.perf_counter()
        self.query_one("#busy", LoadingIndicator).display = True
        self.stop_requested.clear()
        for widget_id in ("ask", "demo", "custom", "before", "after", "similarity", "rerank"):
            self.query_one(f"#{widget_id}").disabled = True
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.report_path = DAY / "resources" / f"{event.button.id}-{stamp}.json"
        self.session_log = SessionLog(self.session_log.path.parent)
        self.set_status(f"Выполняется сравнение · {len(cases) * 5} вызовов DeepSeek")
        self.run_comparison(cases, settings, key)

    def progress(self, report):
        self.report = report
        self.render_results()
        done = sum(result.get("status") == "complete" for item in report["items"] for result in item["results"].values())
        message = f"Готово ответов: {done}/{len(report['items']) * 4} · {report['status']}"
        current = next(((item, mode, result) for item in reversed(report["items"])
                        for mode, result in reversed(list(item["results"].items()))
                        if result.get("status") == "running"), None)
        if current:
            item, mode, result = current
            pending = [step for step in result["steps"] if step["status"] == "running"]
            if pending:
                message = f"{item['id']} · {LABELS[mode]} · {ACTIVITY[pending[-1]['stage']]} · ответов {done}/{len(report['items']) * 4}"
        else:
            pending = [step for step in report["setup"] if step["status"] == "running"]
            if pending:
                message = ACTIVITY[pending[-1]['stage']]
        if report.get("error"):
            message += f" · {report['error']['stage']} / {report['error']['type']}"
            self.notify(f"Этап: {report['error']['stage']}; тип: {report['error']['type']}. Смотрите журнал.",
                        title="Ошибка RAG", severity="error", timeout=15)
        self.set_status(message)

    @work
    async def run_comparison(self, cases, settings, key):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="day23-rag")
        try:
            runner = self.runner_factory(key=key, model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), settings=settings)
            await asyncio.get_running_loop().run_in_executor(
                executor, runner.run, cases, self.report_path, self.session_log,
                lambda report: None if self.stop_requested.is_set() else self.call_from_thread(self.progress, report),
                self.stop_requested.is_set)
        except Exception as error:
            self.set_status(f"Ошибка {type(error).__name__}; подробности в журнале.")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self.running = False
            if not self.stop_requested.is_set():
                self.query_one("#busy", LoadingIndicator).display = False
                for widget_id in ("ask", "demo", "custom", "before", "after", "similarity", "rerank"):
                    self.query_one(f"#{widget_id}").disabled = False
                self.refresh_status()

    def action_quit(self):
        self.stop_requested.set()
        self.exit()
