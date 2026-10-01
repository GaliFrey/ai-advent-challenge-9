"""Full-screen grounded answers, context inspection and ten-question review."""
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from multiprocessing import RLock
from pathlib import Path
from threading import Event

from dotenv import load_dotenv
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Input, Label, LoadingIndicator, Select, Static, TabbedContent, TabPane

from evidence import answer_text
from pipeline import DAY, Runner, Settings, load_cases
from judge import assessment_text, verdict_text
from session_log import SessionLog

STAGES = {"index": "Проверка индекса", "embedding_model": "Загрузка эмбеддингов",
          "reranker_model": "Загрузка reranker", "rewrite": "DeepSeek: rewrite", "search": "Поиск",
          "rerank": "Реранкинг", "filter": "Отбор по порогу", "answer": "DeepSeek: ответ с цитатами",
          "validate": "Проверка источников и точности цитат", "judge": "DeepSeek: проверка смысла"}
DISABLED = ("ask", "demo", "custom", "before", "after", "threshold")


class RagApp(App):
    TITLE = "День 24 — Цитаты, источники и анти-галлюцинации"
    CSS_PATH = "tui.tcss"
    BINDINGS = [("ctrl+q", "quit", "Выход")]

    def __init__(self, report=None, report_path=None, *, cases=None, runner_factory=Runner, log_directory=None):
        self._model_lock = RLock()
        super().__init__()
        self.cases = list(cases if cases is not None else load_cases())
        self.report, self.report_path = report, Path(report_path) if report_path else None
        if report:
            known = {c["id"] for c in self.cases}
            self.cases += [{"id": i["id"], "question": i["question"]} for i in report["items"] if i["id"] not in known]
        self.selected_id = report["items"][0]["id"] if report and report.get("items") else self.cases[0]["id"]
        self.runner_factory = runner_factory
        self.session_log = SessionLog(log_directory) if log_directory else SessionLog()
        self.running, self.ready = False, False
        self.started = None
        self.status_message = ""
        self.stop_requested = Event()
        self.rows = []
        self.theme = "textual-dark"
        load_dotenv(DAY / ".env", override=False)

    def compose(self) -> ComposeResult:
        yield Static("ДЕНЬ 24  /  ОТВЕТ → ЦИТАТА → ИСТОЧНИК", id="brand")
        with Horizontal(id="activity"):
            yield LoadingIndicator(id="busy")
            yield Static(id="status")
        with Horizontal(id="controls"):
            yield Select([(f"{c['id']} · {c['question']}", c["id"]) for c in self.cases],
                         value=self.selected_id, allow_blank=False, id="questions")
            yield Button("Ответ · до 3 вызовов", id="ask", variant="primary")
            yield Button("10 вопросов · до 30", id="demo")
        yield Input(placeholder="Свой вопрос; пустое поле — выбранный вопрос", id="custom")
        settings = Settings(**self.report["settings"]) if self.report else Settings()
        with Horizontal(id="settings"):
            for name, label, value in (("before", "Кандидаты K", settings.top_k_before),
                                       ("after", "Итоговый K", settings.top_k_after),
                                       ("threshold", "Порог reranker", settings.rerank_threshold)):
                yield Label(label)
                yield Input(str(value), id=name)
        yield Static(id="question")
        yield Static(id="flow")
        with TabbedContent(id="tabs"):
            with TabPane("Ответ и доказательства", id="pane-answer"):
                with Horizontal(classes="workspace"):
                    with VerticalScroll(classes="answer-view"):
                        yield Label("ОТВЕТ", classes="box-title")
                        yield Static(id="answer")
                        yield Label("ТОЧНЫЕ ЦИТАТЫ И ИСТОЧНИКИ", classes="box-title")
                        yield Static(id="quotes")
                    with VerticalScroll(classes="checks-view"):
                        yield Label("ПРОВЕРКИ", classes="box-title")
                        yield Static(id="checks")
                        yield Label("ОЦЕНКА LLM", classes="box-title")
                        yield Static("Отдельный запрос проверяет смысл, полноту и обоснованность отказа. Оценка модели может ошибаться.")
                        yield Static(id="review")
            with TabPane("Поиск и контекст", id="pane-context"):
                yield Static(id="query")
                yield DataTable(id="chunks", cursor_type="row", zebra_stripes=True)
                with VerticalScroll(id="chunk-scroll"):
                    yield Static(id="chunk-text")
            with TabPane("10 вопросов", id="pane-summary"):
                yield Static("Формат и точность цитат проверяются автоматически. Смысл оценивает LLM отдельным запросом; отсутствие и ошибки оценки показаны явно.")
                yield DataTable(id="summary", cursor_type="row", zebra_stripes=True)
                yield Static(id="totals")
            with TabPane("Диагностика", id="pane-debug"):
                with VerticalScroll():
                    yield Static(id="debug")
        yield Footer()

    def on_mount(self):
        self.query_one("#chunks", DataTable).add_columns("Поиск", "Ранг", "Источник / раздел", "Similarity", "Reranker", "Отбор")
        self.query_one("#summary", DataTable).add_columns("Вопрос", "Статус", "Ответ / отказ", "Источники", "Цитаты", "Формат", "Оценка LLM")
        self.ready = True
        self.query_one("#busy").display = False
        self.set_status("Ключ DeepSeek настроен. Запуск по кнопке." if os.getenv("DEEPSEEK_API_KEY", "").strip() else
                        "НЕТ КЛЮЧА: заполните day-24/.env и перезапустите TUI. Просмотр работает без ключа.")
        self.set_interval(.25, self.refresh_status)
        self.render_results()

    def set_status(self, message):
        self.status_message = message
        self.refresh_status()

    def refresh_status(self):
        elapsed = f" · прошло {time.perf_counter() - self.started:.1f} с" if self.running and self.started else ""
        self.query_one("#status", Static).update(Text(self.status_message + elapsed + f"\nЖурнал: logs/{self.session_log.path.name}"))

    def current_item(self):
        return next((i for i in (self.report or {}).get("items", []) if i["id"] == self.selected_id), None)

    @on(Select.Changed, "#questions")
    def select_question(self, event):
        if self.ready:
            self.selected_id = str(event.value)
            self.render_results()

    @on(DataTable.RowHighlighted, "#chunks")
    def show_chunk(self, event):
        if self.ready and 0 <= event.cursor_row < len(self.rows):
            c = self.rows[event.cursor_row]
            self.query_one("#chunk-text", Static).update(Text(
                f"{c.get('source_id') or 'Не передан модели'} · {c['source']} · {c['section']}\nID: {c['chunk_id']}\n\n{c['text']}"))

    @on(DataTable.RowSelected, "#summary")
    def open_summary_question(self, event):
        self.query_one("#questions", Select).value = str(event.row_key.value)
        self.query_one("#tabs", TabbedContent).active = "pane-answer"

    def render_results(self):
        if not self.ready:
            return
        item = self.current_item() or {}
        case = next(c for c in self.cases if c["id"] == self.selected_id)
        validation_only = bool((self.report or {}).get("validation_only"))
        self.query_one("#question", Static).update(Text(("ЛОКАЛЬНАЯ ПРОВЕРКА: синтетические ответы.\n" if validation_only else "") + case["question"], style="bold"))
        steps = (self.report or {}).get("setup", []) + item.get("steps", [])
        self.query_one("#flow", Static).update(Text(" → ".join(
            f"{STAGES.get(s['stage'], s['stage'])} {'✓' if s['status'] == 'complete' else s['status']}" for s in steps)
            or "Вопрос → rewrite → поиск → reranker → порог → ответ → проверка"))
        data = item.get("response")
        self.query_one("#answer", Static).update(Text(answer_text(data) if data or item.get("status") == "invalid" else "Ответ появится после запуска."))
        quotes = []
        if data:
            for n, claim in enumerate(data["answer"], 1):
                for citation in claim["citations"]:
                    source = next(s for s in data["sources"] if s["source_id"] == citation["source_id"])
                    quotes.append(f"Утверждение {n} · [{source['source_id']}]\n«{citation['quote']}»\n{source['source']}\nРаздел: {source['section']}\nchunk_id: {source['chunk_id']}")
        self.query_one("#quotes", Static).update(Text("\n\n".join(quotes) or ("Нет цитат и источников: отказ от ответа." if data else "—")))
        checks = item.get("checks", {})
        check_text = "Формат ещё не проверен."
        if checks:
            check_text = ("Формат и точные цитаты: ПРОШЛИ" if checks["passed"] else "Ответ ОТКЛОНЁН: " + "; ".join(checks["errors"]))
            check_text += f"\nИсточников: {checks.get('sources', 0)} · цитат: {checks['quotes']}"
            if data and data["status"] == "unknown":
                check_text += "\nОтказ: источники/цитаты отсутствуют по контракту."
            if item.get("origin") == "threshold":
                check_text += "\nВсе кандидаты ниже порога; генерация не вызывалась."
            check_text += f"\nТокены генерации: {item.get('usage', {}).get('total_tokens', 0)}; LLM: {item.get('llm_seconds', 0)} с"
        self.query_one("#checks", Static).update(Text(check_text))
        self.query_one("#review", Static).update(Text(assessment_text(item)))
        rewrite = item.get("rewrite", {})
        self.query_one("#query", Static).update(Text(f"Исходный вопрос: {case['question']}\nПоиск: {rewrite.get('query', '—')}\nЗапрос изменён: {rewrite.get('changed', '—')}"))
        table = self.query_one("#chunks", DataTable)
        table.clear()
        self.rows = item.get("candidates", [])
        for c in self.rows:
            table.add_row(str(c["rank_before"]), str(c.get("rank_after", "—")), Text(f"{c['source']} · {c['section']}"),
                          f"{c['score']:.3f}", f"{c.get('rerank_score', 0):.3f}", c.get("decision", "Ожидает отбора"))
        self.query_one("#chunk-text", Static).update(Text(self.rows[0]["text"] if self.rows else "Нет фрагментов."))
        summary = self.query_one("#summary", DataTable)
        summary.clear()
        items = (self.report or {}).get("items", [])
        for row in items:
            c, r = row.get("checks", {}), row.get("response")
            summary.add_row(row["id"], row["status"], ("Отказ" if r["status"] == "unknown" else "Ответ") if r else "—",
                            str(c.get("sources", "—")), str(c.get("quotes", "—")),
                            ("✓" if c["passed"] else "Ошибка") if c else "—",
                            verdict_text(row), key=row["id"])
        self.query_one("#totals", Static).update(Text(f"Готово: {sum(i['status'] == 'complete' for i in items)}/{len(items)} · оценок LLM: {sum(i.get('judge', {}).get('status') == 'complete' for i in items)}/{len(items)}\nРезультат: {self.report_path or 'не сохранён'}"))
        self.query_one("#debug", Static).update(Text("Ошибки: " + "; ".join(checks.get("errors", [])) + "\n\nИсходный ответ модели (непроверенные данные):\n" + item.get("raw_answer", "—") + "\n\nИсходная оценка LLM:\n" + item.get("judge", {}).get("answer", "—")))

    @on(Button.Pressed, "#ask")
    @on(Button.Pressed, "#demo")
    def launch(self, event):
        if self.running:
            return
        try:
            settings = Settings(int(self.query_one("#before", Input).value), int(self.query_one("#after", Input).value), float(self.query_one("#threshold", Input).value))
            key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            if not key:
                raise ValueError("Задайте DEEPSEEK_API_KEY в day-24/.env и перезапустите TUI.")
        except ValueError as error:
            try:
                with self.session_log.stage("configuration"):
                    raise error
            except ValueError:
                pass
            self.set_status(f"ЗАПУСК ОСТАНОВЛЕН: {error}")
            self.notify(str(error), severity="error", timeout=10)
            return
        if event.button.id == "demo":
            cases = load_cases()
            known = {c["id"] for c in self.cases}
            self.cases += [c for c in cases if c["id"] not in known]
            self.query_one("#questions", Select).set_options(
                [(f"{c['id']} · {c['question']}", c["id"]) for c in self.cases])
        else:
            question = self.query_one("#custom", Input).value.strip()
            if question:
                case = {"id": "custom", "question": question}
                self.cases = [case] + [c for c in self.cases if c["id"] != "custom"]
                select = self.query_one("#questions", Select)
                select.set_options([(f"{c['id']} · {c['question']}", c["id"]) for c in self.cases])
                self.selected_id = "custom"
                select.value = "custom"
            cases = [next(c for c in self.cases if c["id"] == self.selected_id)]
        if event.button.id == "demo":
            self.selected_id = cases[0]["id"]
            self.query_one("#questions", Select).value = self.selected_id
        self.report_path = DAY / "resources" / (event.button.id + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
        self.running, self.started = True, time.perf_counter()
        self.stop_requested.clear()
        self.session_log = SessionLog(self.session_log.path.parent)
        self.query_one("#busy").display = True
        for name in DISABLED:
            self.query_one(f"#{name}").disabled = True
        self.set_status(f"Запуск: до {3 * len(cases)} вызовов DeepSeek")
        self.run_pipeline(cases, settings, key)

    def progress(self, report):
        self.report = report
        self.render_results()
        steps = report["setup"] + [s for i in report["items"] for s in i["steps"]]
        pending = next((s for s in reversed(steps) if s["status"] == "running"), None)
        message = STAGES[pending["stage"]] if pending else f"Прогон: {report['status']}"
        self.set_status(message)
        if report.get("error"):
            self.set_status(f"Ошибка: {report['error']['stage']} / {report['error']['type']}. См. журнал.")
            self.notify(self.status_message, severity="error", timeout=10)

    @work
    async def run_pipeline(self, cases, settings, key):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="day24-rag")
        try:
            runner = self.runner_factory(key=key, model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), settings=settings)
            await asyncio.get_running_loop().run_in_executor(executor, runner.run, cases, self.report_path, self.session_log,
                lambda r: None if self.stop_requested.is_set() else self.call_from_thread(self.progress, r), self.stop_requested.is_set)
        except Exception as error:
            self.set_status(f"Ошибка: {type(error).__name__}; проверьте журнал.")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self.running = False
            if not self.stop_requested.is_set():
                self.query_one("#busy").display = False
                for name in DISABLED:
                    self.query_one(f"#{name}").disabled = False
                self.render_results()
                self.refresh_status()

    def action_quit(self):
        self.stop_requested.set()
        self.exit()
