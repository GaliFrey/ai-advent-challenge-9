"""Live, single-screen RAG comparison and ten-question demo."""
from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from multiprocessing import RLock
from pathlib import Path

from dotenv import load_dotenv
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Label, Select, Static

from demo import LABELS
from main import DAY, QUESTIONS, save_report
from rag import PARAMETERS, answer_question
from retrieval import STRATEGY, TOP_K, Retriever
from session_log import SessionLog, error_details, user_error


def new_report(model: str, cases: list[dict]) -> dict:
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model, "parameters": PARAMETERS, "strategy": STRATEGY,
        "top_k": TOP_K, "index": None, "status": "running",
        "items": [{**case, "results": {}} for case in cases],
    }


def source_lines(chunks: list[dict]) -> Text:
    if not chunks:
        return Text("Поиск не нашёл фрагментов.", style="yellow")
    lines = Text()
    for number, chunk in enumerate(chunks, 1):
        lines.append(f"S{number}  ", style="bold cyan")
        lines.append(f"{chunk['source']}  ·  сходство {chunk['score']}\n")
    return lines


class RagApp(App):
    TITLE = "День 22 — Первый RAG"
    CSS_PATH = "tui.tcss"
    BINDINGS = [("ctrl+q", "quit", "Выход")]

    def __init__(self, report: dict | None = None, *, questions: list[dict] | None = None,
                 ask=answer_question, retriever_factory=Retriever, key: str | None = None):
        # sentence-transformers may create a multiprocessing synchronizer when
        # it first loads. Start Python's resource tracker before Textual opens
        # terminal descriptors; spawning it later fails on Python 3.14 here.
        self._model_load_lock = RLock()
        super().__init__()
        self.questions = questions if questions is not None else json.loads(QUESTIONS.read_text(encoding="utf-8"))
        self.cases = {case["id"]: case for case in self.questions}
        self.report = report
        self.report_path: Path | None = None
        self.ask = ask
        self.retriever_factory = retriever_factory
        self.key = key
        load_dotenv(DAY / ".env", override=False)
        self.model = report["model"] if report else os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.session_log = SessionLog(DAY / "resources" / "logs")
        self.session_log.append("session_open")
        self.selected_question = self.questions[0]["id"]
        self.running = False
        self.ready = False
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        yield Static("ДЕНЬ 22  /  ПЕРВЫЙ RAG", id="brand")
        yield Static(id="run-status")
        with Horizontal(id="controls"):
            yield Label("Вопрос", id="question-label")
            yield Select([(f"{case['id']} · {LABELS[i]}", case["id"])
                          for i, case in enumerate(self.questions)],
                         value=self.selected_question, allow_blank=False, id="question-select")
            yield Button("Спросить · 2 запроса", id="ask", variant="primary")
            yield Button("Демо · 20 запросов", id="demo")
        yield Static("Вопрос → LLM  |  Вопрос → поиск 5 чанков → LLM", id="flow")
        with VerticalScroll(id="page"):
            with Vertical(id="content"):
                yield Static(id="question", classes="question")
                with Horizontal(id="answers"):
                    with Vertical(classes="answer-box", id="plain-box"):
                        yield Label("БЕЗ RAG", classes="box-title")
                        yield Static(id="plain-answer", classes="answer")
                        yield Static(id="plain-meta", classes="meta")
                    with Vertical(classes="answer-box", id="rag-box"):
                        yield Label("С RAG", classes="box-title")
                        yield Static(id="rag-answer", classes="answer")
                        yield Static(id="rag-meta", classes="meta")
                with Vertical(id="evidence"):
                    yield Label("ЧТО НАШЛОСЬ ДЛЯ RAG", classes="box-title")
                    yield Static(id="sources")
                    yield Select([("Сначала выполните вопрос", -1)], value=-1,
                                 allow_blank=False, id="chunk-select")
                    yield Static(id="chunk-meta", classes="meta")
                    yield Static(id="chunk-text")
                yield Static(id="result-summary")
        yield Footer()

    def on_mount(self) -> None:
        self.ready = True
        self.update_status("Выберите вопрос и нажмите «Спросить» или запустите полное демо.")
        self.show_question()

    def update_status(self, message: str) -> None:
        prefix = f"Модель: {self.model}"
        if self.report:
            prefix += f"  ·  Запуск: {self.report['created_at'][:19]} UTC"
        self.query_one("#run-status", Static).update(Text(f"{prefix}  ·  {message}"))

    def current_item(self) -> dict | None:
        if self.report:
            return next((item for item in self.report["items"]
                         if item["id"] == self.selected_question), None)
        return None

    @on(Select.Changed, "#question-select")
    def question_changed(self, event: Select.Changed) -> None:
        if self.ready and event.value in self.cases:
            self.selected_question = str(event.value)
            self.show_question()

    @on(Select.Changed, "#chunk-select")
    def chunk_changed(self, event: Select.Changed) -> None:
        if self.ready and isinstance(event.value, int):
            self.show_chunk(event.value)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ask":
            if self.running:
                return
            self.set_running(True)
            self.run_questions([self.selected_question], full_demo=False)
        elif event.button.id == "demo":
            if self.running:
                return
            self.set_running(True)
            self.run_questions([case["id"] for case in self.questions], full_demo=True)

    def set_running(self, value: bool) -> None:
        self.running = value
        for widget_id in ("ask", "demo"):
            self.query_one(f"#{widget_id}", Button).disabled = value

    def show_question(self) -> None:
        self.query_one("#question", Static).update(Text(self.cases[self.selected_question]["question"], style="bold"))
        item = self.current_item()
        results = item["results"] if item else {}
        for mode in ("plain", "rag"):
            result = results.get(mode)
            if result and "answer" in result:
                content = Text(result["answer"])
                content.highlight_regex(r"`[^`]+`|\[S\d+\]", "bold cyan")
                usage = result.get("usage", {})
                meta = (f"Токены: {usage.get('total_tokens', '—')}  ·  LLM: {result['llm_seconds']} с")
                if mode == "rag":
                    meta += f"  ·  поиск: {result['search_seconds']} с"
                    invalid = result["citation_check"]["unknown_references"]
                    if invalid:
                        meta += f"  ·  неверные ссылки: {', '.join(invalid)}"
            elif result and "error" in result:
                content, meta = Text(result["error"], style="red"), "Запрос завершился с ошибкой."
            else:
                content = Text("Ожидание ответа модели…" if self.running and item else
                               "Ответ появится после запуска вопроса.", style="dim")
                meta = ""
            self.query_one(f"#{mode}-answer", Static).update(content)
            self.query_one(f"#{mode}-meta", Static).update(Text(meta))
        chunks = results.get("rag", {}).get("chunks", [])
        self.query_one("#sources", Static).update(source_lines(chunks) if "rag" in results
                   else Text("Чанки появятся после ответа с RAG.", style="dim"))
        select = self.query_one("#chunk-select", Select)
        select.set_options([(f"S{i} · {chunk['source']} · {chunk['section']}", i - 1)
                            for i, chunk in enumerate(chunks, 1)] or [("Нет фрагментов", -1)])
        select.value = 0 if chunks else -1
        self.show_chunk(select.value)
        if item and item["results"]:
            expected = set(item.get("sources", []))
            found = {chunk["source"] for chunk in chunks}
            coverage = f"Ожидаемые файлы в выдаче: {len(expected & found)}/{len(expected)}. " if expected and chunks else ""
            review = results.get("rag", {}).get("review")
            note = f"  ·  Ручная заметка: {review['note']}" if review else ""
            self.query_one("#result-summary", Static).update(
                Text(f"{coverage}Наличие файла не гарантирует наличие нужного фрагмента.{note}"))
        else:
            self.query_one("#result-summary", Static).update(Text(""))

    def show_chunk(self, index: int) -> None:
        item = self.current_item()
        chunks = item["results"].get("rag", {}).get("chunks", []) if item else []
        if 0 <= index < len(chunks):
            chunk = chunks[index]
            meta = (f"S{index + 1}  ·  {chunk['source']}  ·  {chunk['section']}\n"
                    f"ID: {chunk['chunk_id']}  ·  сходство: {chunk['score']}")
            content = chunk["text"]
        else:
            meta, content = "", ""
        self.query_one("#chunk-meta", Static).update(Text(meta))
        self.query_one("#chunk-text", Static).update(Text(content))

    @work
    async def run_questions(self, ids: list[str], *, full_demo: bool) -> None:
        if self.key is None:
            load_dotenv(DAY / ".env", override=False)
            key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        else:
            key = self.key
        if not key:
            self.update_status("Добавьте DEEPSEEK_API_KEY в day-22/.env")
            self.session_log.append("key_missing")
            self.set_running(False)
            return
        self.report = new_report(self.model, [self.cases[item_id] for item_id in ids])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        kind = "benchmark" if full_demo else "ask"
        self.report_path = DAY / "resources" / f"{kind}-{stamp}.json"
        completed = 0
        total = len(ids) * 2
        mode = None
        active_item = None
        stage = "setup"
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-request")
        try:
            self.session_log.append("run_start", total=total, report_file=self.report_path.name)
            save_report(self.report_path, self.report)
            stage = "model"
            self.update_status("Проверка индекса и загрузка модели эмбеддингов…")
            self.session_log.append("model_load_start")
            retriever = await asyncio.get_running_loop().run_in_executor(executor, self.retriever_factory)
            self.report["index"] = retriever.metadata
            save_report(self.report_path, self.report)
            self.session_log.append("model_load_ok")
            for item in self.report["items"]:
                active_item = item
                self.query_one("#question-select", Select).value = item["id"]
                self.selected_question = item["id"]
                self.show_question()
                for mode in ("plain", "rag"):
                    stage = "request"
                    name = "без RAG" if mode == "plain" else "с RAG"
                    self.update_status(f"{item['id']} · запрос {name} · {completed}/{total} готово…")
                    self.session_log.append("request_start", question_id=item["id"], mode=mode,
                                    completed=completed, total=total)
                    call = partial(self.ask, item["question"], mode, key=key,
                                   model=self.model, retriever=retriever)
                    result = await asyncio.get_running_loop().run_in_executor(executor, call)
                    item["results"][mode] = result
                    completed += 1
                    save_report(self.report_path, self.report)
                    self.session_log.append("request_ok", question_id=item["id"], mode=mode,
                                    completed=completed, total=total,
                                    tokens=result.get("usage", {}).get("total_tokens"),
                                    chunks=len(result.get("chunks", [])),
                                    seconds=result.get("llm_seconds"))
                    self.show_question()
            self.report["status"] = "complete"
            save_report(self.report_path, self.report)
            self.session_log.append("run_complete", completed=completed, total=total)
            self.update_status(f"Готово: {completed}/{total} запросов. Выбирайте вопросы в списке сверху.")
        except Exception as error:
            detail = error_details(error, stage)
            self.session_log.append("run_failed", question_id=active_item["id"] if active_item else None,
                            mode=mode, completed=completed, total=total, **detail)
            if active_item and mode and mode not in active_item["results"]:
                active_item["results"][mode] = {"error": user_error(error, stage)}
            self.report["status"] = "failed"
            self.report["error"] = {"stage": stage, **detail}
            save_report(self.report_path, self.report)
            self.show_question()
            self.update_status(f"Остановлено после {completed}/{total}: {user_error(error, stage)} "
                               f"Журнал: {self.session_log.path.name}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self.set_running(False)
