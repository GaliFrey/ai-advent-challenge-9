#!/usr/bin/env python3
"""Терминальный интерфейс сравнения моделей. Запросы — только по кнопке."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Label, Static, TabbedContent, TabPane, TextArea

from experiment import ROOT, RESULTS, Settings, load_session, load_settings, run_experiment, validate_prompts, all_records, ANSWER_MAX_TOKENS
from verification import verification_markdown

SLOTS = ("weak", "medium", "strong")
STATUS = {"ok": "Готово", "incomplete": "Неполный ответ", "error": "Ошибка"}
COLORS = {"ok": "#78dba9", "incomplete": "#edc47b", "error": "#ef8d92"}


def clean_text(value: str) -> str:
    # Model output is text, never terminal control sequences or Rich markup.
    value = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))", "", value)
    return "".join(c for c in value if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) <= 159))


def model_label(name: str) -> str:
    return {
        "nvidia/nemotron-3.5-lightning:free": "Nemotron Lightning",
        "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B A4B",
        "qwen/qwen3-4b:free": "Qwen3 4B",
        "qwen/qwen3.8-27b": "Qwen 3.8 27B",
        "deepseek-v4-flash": "DeepSeek Flash",
        "deepseek-v4-pro": "DeepSeek Pro",
    }.get(name, name)


def number(value) -> str:
    return "—" if value is None else str(value)


def money(record: dict) -> str:
    value = record["cost"]["usd"]
    if value is None:
        return "—"
    if value == 0:
        return "$0 Free" if record["provider"] in {"groq", "openrouter"} else "$0"
    return f"${value:.8f}"


class CompareApp(App):
    TITLE = "MODEL COMPARE · День 05"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("ctrl+r", "start", "Запустить"),
        ("ctrl+o", "open_latest", "Последний результат"),
        ("ctrl+p", "toggle_prompt", "Задача"),
        ("ctrl+q", "quit_safe", "Выйти"),
    ]

    def __init__(self, settings: Settings, *, session_path: Path | None = None, results_dir: Path = RESULTS, transport=None):
        super().__init__()
        self.settings = settings
        self.initial_session = session_path
        self.results_dir = results_dir
        self.transport = transport
        self.session = None
        self.session_path = None
        self.busy = False
        self.active_request = None
        self.request_started = 0.0
        self.theme = "textual-dark"

    def compose(self) -> ComposeResult:
        with Horizontal(id="brand"):
            yield Label("MODEL COMPARE", id="title")
            yield Label("ДЕНЬ 05  /  ТРИ МОДЕЛИ · ДВЕ ЗАДАЧИ", id="subtitle")
        with VerticalScroll(id="workspace"):
            with TabbedContent(id="prompts"):
                for index, filename in enumerate(("prompt.txt", "prompt-02.txt"), 1):
                    with TabPane(f"Промпт {index} · редактировать", id=f"editor-{index}"):
                        yield TextArea((ROOT / filename).read_text(encoding="utf-8"), id=f"prompt-{index}", soft_wrap=True)
            with Horizontal(id="actions"):
                yield Button("▶ Запустить сравнение", variant="primary", id="start")
                yield Button("Последний результат", id="open")
                yield Button("Скрыть промпты", id="toggle")
            yield Static("Готово к запуску · 6 запросов по очереди · без запроса к анализатору", id="phase", markup=False)
            yield DataTable(id="metrics", show_cursor=False, zebra_stripes=True)
            yield Static(f"Лимит ответа: {ANSWER_MAX_TOKENS} ток. Токены: вход / выход. USD ≈ оценка. OpenRouter: Free.", id="legend", markup=False)
            with TabbedContent(id="tabs"):
                with TabPane("Проверки", id="conclusion"):
                    with VerticalScroll(classes="answer-scroll"):
                        yield Static("Начни запись видео и нажми «Запустить сравнение».\n\nЗдесь появятся только локальные проверки. Итоговую оценку делаем вручную по ответам моделей.", id="conclusion-text", markup=False)
                with TabPane("Метрики", id="summary"):
                    with VerticalScroll(classes="answer-scroll"):
                        yield Static("Статистика появится после ответов.", id="summary-text", markup=False)
                for index in (1, 2):
                    with TabPane(f"Задача {index} · ответы", id=f"task-{index}"):
                        with TabbedContent(id=f"answers-{index}"):
                            with TabPane("Сравнение", id=f"compare-tab-{index}"):
                                with VerticalScroll(classes="answer-scroll comparison-scroll"):
                                    for slot in SLOTS:
                                        yield Static(model_label(next(m.name for m in self.settings.models if m.slot == slot)), classes="comparison-title", id=f"compare-title-{index}-{slot}", markup=False)
                                        yield Static("Ожидает запуска", classes="comparison-meta", id=f"compare-meta-{index}-{slot}", markup=False)
                                        yield Static("Ответ появится после завершения запроса.", classes="comparison-preview", id=f"compare-preview-{index}-{slot}", markup=False)
                            for slot in SLOTS:
                                with TabPane(model_label(next(m.name for m in self.settings.models if m.slot == slot)), id=f"tab-{index}-{slot}"):
                                    with VerticalScroll(classes="answer-scroll"):
                                        yield Static("Ответ появится после завершения запроса.", id=f"answer-{index}-{slot}", markup=False)
            yield Static("Результаты сохраняются локально. Открытие сохранённой сессии не вызывает API.", id="saved", markup=False)
        yield Footer()

    def on_mount(self):
        table = self.query_one("#metrics", DataTable)
        table.add_columns("Задача", "Модель", "Состояние", "Сек.", "Вход", "Выход", "USD ≈")
        self.refresh_metrics()
        self.set_interval(.25, self.tick)
        self.query_one("#start", Button).focus()
        if self.initial_session:
            self.open_saved(self.initial_session)

    def refresh_metrics(self):
        table = self.query_one("#metrics", DataTable)
        table.clear()
        models = self.session["models"] if self.session else [{"slot": m.slot, "name": m.name} for m in self.settings.models]
        prompts = self.session["prompts"] if self.session else [{"id": "task-1"}, {"id": "task-2"}]
        for index, prompt in enumerate(prompts, 1):
            for model in models:
                slot = model["slot"]
                record = self.session["results"][prompt["id"]].get(slot) if self.session else None
                if record:
                    status = Text(STATUS.get(record["status"], "Неизвестно"), style=COLORS.get(record["status"], "white"))
                    seconds = f"{record['elapsed_seconds']:.1f}"
                    metrics = record["metrics"]
                    row = (number(metrics["input_tokens"]), number(metrics["output_tokens"]), money(record))
                else:
                    running = self.busy and self.session and self.session["status"] == "running"
                    pending = running and self.session.get("active_request") == {"task_id": prompt["id"], "slot": slot}
                    waiting = running and self.session.get("waiting_request") == {"task_id": prompt["id"], "slot": slot}
                    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                    label = frames[int(time.monotonic() * 4) % len(frames)] + " Запрос" if pending else f"Пауза {self.session['groq_delay_seconds']} с" if waiting else "В очереди" if running else "Нет ответа"
                    status = Text(label, style="#79c5d8" if pending or waiting else "#8190a7")
                    seconds = f"{time.monotonic() - self.request_started:.1f}" if pending else "—"
                    row = ("—", "—", "—")
                table.add_row(str(index), Text(clean_text(model["name"])), status, seconds, *row, key=prompt["id"] + slot)

    def tick(self):
        if self.busy:
            self.refresh_metrics()

    def set_busy(self, value):
        self.busy = value
        self.query_one("#start", Button).disabled = value
        self.query_one("#open", Button).disabled = value
        for editor in self.query(TextArea):
            editor.read_only = value

    @on(Button.Pressed, "#start")
    def action_start(self):
        if self.busy:
            return
        try:
            self.settings.check()
        except ValueError as exc:
            self.query_one("#phase", Static).update(str(exc))
            return
        prompts = [self.query_one(f"#prompt-{i}", TextArea).text.strip() for i in (1, 2)]
        try:
            validate_prompts(prompts)
        except ValueError as exc:
            self.query_one("#phase", Static).update(str(exc))
            return
        self.session = None
        self.set_busy(True)
        self.query_one("#prompts").display = False
        self.query_one("#toggle", Button).label = "Показать промпты"
        self.query_one("#tabs", TabbedContent).active = "conclusion"
        self.query_one("#conclusion-text", Static).update("Три модели решают две задачи по очереди. После шестого ответа откроется сравнение без дополнительного API-запроса.")
        self.query_one("#summary-text", Static).update("Собираем статистику по двум задачам…")
        for index in (1, 2):
            for slot in SLOTS:
                self.query_one(f"#answer-{index}-{slot}", Static).update("Запрос в очереди…")
                self.query_one(f"#compare-meta-{index}-{slot}", Static).update("В очереди")
                self.query_one(f"#compare-preview-{index}-{slot}", Static).update("Ответ ещё не получен.")
        self.execute(prompts)

    @work(exclusive=True)
    async def execute(self, prompts):
        try:
            await run_experiment(self.settings, prompts, self.accept_update, results_dir=self.results_dir, transport=self.transport)
        except OSError:
            self.query_one("#phase", Static).update("Ошибка записи на диск. Данные на экране могут быть сохранены не полностью. Новых запросов не запускаем.")
        except Exception:
            # Do not expose exception bodies/headers, which might contain credentials.
            self.query_one("#phase", Static).update("Внутренняя ошибка. Автоматического повтора нет; доступные результаты остались на экране.")
        finally:
            self.set_busy(False)
            self.query_one("#start", Button).label = "▶ Новый запуск · 6 запросов"
            self.refresh_metrics()

    def accept_update(self, session, path):
        self.session = session
        self.session_path = path
        active = session.get("active_request")
        if active != self.active_request:
            self.active_request = active
            self.request_started = time.monotonic()
        if active and self.busy:
            index = active["task_id"].split("-")[-1]
            self.query_one(f"#answer-{index}-{active['slot']}", Static).update("Запрос выполняется…")
        waiting = session.get("waiting_request")
        if waiting and self.busy:
            index = waiting["task_id"].split("-")[-1]
            self.query_one(f"#answer-{index}-{waiting['slot']}", Static).update(f"Пауза {session['groq_delay_seconds']} с перед отправкой в Groq…")
        for index in (1, 2):
            for model in session["models"]:
                self.query_one(f"#answers-{index}", TabbedContent).get_tab(f"tab-{index}-{model['slot']}").label = model_label(model["name"])
        state = session["status"]
        phases = {
            "running": f"01 / ОТВЕТЫ  ·  завершено {len(all_records(session))} из {len(session['prompts']) * 3}",
            "checking": "02 / ЛОКАЛЬНАЯ ПРОВЕРКА  ·  API не используется",
            "complete": "03 / ГОТОВО  ·  шесть ответов сохранены для ручной оценки",
            "partial": "03 / ЗАВЕРШЕНО С ОГРАНИЧЕНИЯМИ  ·  часть ответов недоступна",
            "failed": "ЗАПРОСЫ НЕ УДАЛИСЬ  ·  ответы не получены",
            "interrupted": "СЕССИЯ ПРЕРВАНА  ·  сохранены доступные результаты",
        }
        self.query_one("#phase", Static).update(phases.get(state, state))
        self.query_one("#saved", Static).update("Сессия: " + session["id"] + "  ·  resources/results/  ·  без ключей API")
        answer_limit = session.get("answer_max_tokens")
        if answer_limit is None:
            answer_limit = next((r.get("request", {}).get("max_tokens") for r in all_records(session)), None)
        free_plan = "OpenRouter: Free" if session.get("free_provider") == "openrouter" else "Groq: " + session.get("groq_plan", "неизвестно")
        self.query_one("#legend", Static).update(f"Лимит ответа: {number(answer_limit)} ток. Вход / выход. USD ≈ тарифы " + session["price_date"] + ". " + free_plan + ".")
        for index, prompt in enumerate(session["prompts"], 1):
            for slot, record in session["results"][prompt["id"]].items():
                info = [record["model"], "Причина завершения: " + str(record["finish_reason"]), record["cost"]["basis"]]
                if record["error"]:
                    info.append(record["error"])
                self.query_one(f"#answer-{index}-{slot}", Static).update(Text(clean_text("\n".join(info) + "\n\n" + (record["answer"] or "Текст ответа не получен."))))
        self.update_summary(session)
        self.update_comparisons(session)
        if session.get("verification"):
            self.query_one("#conclusion-text", Static).update(RichMarkdown(clean_text(verification_markdown(session))))
        if state in {"complete", "partial", "failed"}:
            self.query_one("#tabs", TabbedContent).active = "task-1"
        self.refresh_metrics()

    def update_comparisons(self, session):
        verification = session.get("verification", {}).get("tasks", {})
        for index, prompt in enumerate(session["prompts"], 1):
            task_checks = verification.get(prompt["id"], {})
            for model in session["models"]:
                slot = model["slot"]
                self.query_one(f"#compare-title-{index}-{slot}", Static).update(model_label(model["name"]))
                record = session["results"][prompt["id"]].get(slot)
                if not record:
                    continue
                metrics = record["metrics"]
                check = task_checks.get("answers", {}).get(slot, {})
                score = check.get("score")
                maximum = task_checks.get("max_score")
                check_text = f"локальные проверки {score}/{maximum}" if score is not None else "локальная оценка отсутствует"
                meta = (f"{STATUS.get(record['status'], record['status'])}  ·  {record['elapsed_seconds']:.1f} с  ·  "
                        f"{number(metrics['input_tokens'])} / {number(metrics['output_tokens'])} ток.  ·  {money(record)}  ·  {check_text}")
                self.query_one(f"#compare-meta-{index}-{slot}", Static).update(meta)
                answer = clean_text(record.get("answer") or record.get("error") or "Текст ответа не получен.").strip()
                limit = 900
                preview = answer[:limit].rstrip()
                if len(answer) > limit:
                    preview += f"\n\n… ещё {len(answer) - limit} символов · полный текст во вкладке модели"
                self.query_one(f"#compare-preview-{index}-{slot}", Static).update(Text(preview))

    def update_summary(self, session):
        table = ["| Модель | Готово | Σ с | Вход | Выход | Всего ток. | USD ≈ |",
                 "|---|---|---|---|---|---|---|"]
        for model in session["models"]:
            totals = session["summary"][model["slot"]]
            seconds = totals["sum_elapsed_seconds"]
            cost = totals["estimated_cost_usd"]
            row = [model_label(model["name"]), f"{totals['successful']}/{totals['expected']}",
                   "—" if seconds is None else f"{seconds:.2f}",
                   number(totals["input_tokens"]), number(totals["output_tokens"]), number(totals["total_tokens"]),
                   "—" if cost is None else f"${cost:.8f}"]
            table.append("| " + " | ".join(row) + " |")
        text = "\n".join(table) + "\n\nΣ с — сумма длительностей двух запросов модели без ожидания в очереди. Прочерк: итог ещё не известен или метрика отсутствует. Качество оцениваем вручную по вкладкам задач; токены и скорость не являются оценкой качества."
        self.query_one("#summary-text", Static).update(RichMarkdown(text))

    @on(Button.Pressed, "#toggle")
    def action_toggle_prompt(self):
        prompt = self.query_one("#prompts")
        prompt.display = not prompt.display
        self.query_one("#toggle", Button).label = "Скрыть промпты" if prompt.display else "Показать промпты"

    @on(Button.Pressed, "#open")
    def action_open_latest(self):
        if self.busy:
            return
        paths = sorted(self.results_dir.glob("*.json"), reverse=True)
        if not paths:
            self.query_one("#phase", Static).update("Сохранённых сессий пока нет.")
            return
        self.open_saved(paths[0])

    def open_saved(self, path):
        try:
            session = load_session(path)
            # Missing answers must not leave text from a previous session on screen.
            for index in (1, 2):
                for slot in SLOTS:
                    self.query_one(f"#answer-{index}-{slot}", Static).update("Ответ не сохранён.")
            self.query_one("#conclusion-text", Static).update("В этой сессии локальные проверки не сохранены; при открытии они будут пересчитаны.")
            for index, prompt in enumerate(session["prompts"], 1):
                self.query_one(f"#prompt-{index}", TextArea).load_text(prompt["text"])
            # Historical sessions contain one task; retain a usable second default for a new run.
            if len(session["prompts"]) == 1:
                self.query_one("#prompt-2", TextArea).load_text((ROOT / "prompt-02.txt").read_text(encoding="utf-8"))
            self.accept_update(session, path)
            self.query_one("#phase", Static).update("ПРОСМОТР СОХРАНЁННОЙ СЕССИИ  ·  API не вызывается  ·  статус: " + session["status"])
            self.query_one("#prompts").display = False
            self.query_one("#toggle", Button).label = "Показать промпты"
        except (ValueError, KeyError, TypeError, AttributeError):
            self.query_one("#phase", Static).update("Не удалось открыть файл: неверный формат сессии дня 5.")

    def action_quit_safe(self):
        if self.busy:
            self.notify("Дождись завершения запросов и сохранения результатов.", title="Эксперимент выполняется")
        else:
            self.exit()


def main():
    parser = argparse.ArgumentParser(description="День 5: сравнение ответов трёх моделей в терминале. API вызывается только по кнопке.")
    parser.add_argument("--session", type=Path, help="открыть сохранённый JSON без API-вызовов")
    args = parser.parse_args()
    CompareApp(load_settings(), session_path=args.session).run()


if __name__ == "__main__":
    main()
