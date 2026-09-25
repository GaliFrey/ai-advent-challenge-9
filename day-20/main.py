"""Compact three-server MCP orchestration display for the day 20 video."""
from __future__ import annotations

import json
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from agent import run_turn
from mcp_client import discover_tools
from servers import SERVERS
from trace_log import SessionLog


ROOT = Path(__file__).resolve().parent
VISIBLE = {"USER", "MCP CALL", "MCP RESULT", "MCP ERROR", "ROUTE REJECTED",
           "DOWNLOAD START", "DOWNLOAD ERROR", "REPORT RECEIVED", "ANSWER", "ERROR"}


class MCPAgentApp(App[None]):
    CSS_PATH = ROOT / "app.tcss"
    TITLE = "AI Advent · День 20 · три MCP-сервера"
    BINDINGS = [("ctrl+q", "quit", "Выход")]

    def __init__(self, log_dir: Path | None = None) -> None:
        super().__init__()
        self.session_log = SessionLog(log_dir or ROOT / "sessions")
        self.events: list[tuple[str, object]] = []
        self.busy = False
        self.ready = False
        self.log_error: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("ДЕНЬ 20 · ОРКЕСТРАЦИЯ ТРЁХ MCP-СЕРВЕРОВ", id="brand")
        yield Static("SOURCE → ANALYZE → REPORT · проверяю подключения…", id="servers", markup=False)
        yield Static("Подключение к серверам…", id="status", markup=False)
        with Horizontal(id="workspace"):
            with Vertical(id="timeline-panel"):
                yield Static("ЗАПРОС · ВЫБОР LLM · ВЫЗОВЫ · РЕЗУЛЬТАТ", classes="panel-title")
                yield OptionList(id="steps")
            with Vertical(id="detail-panel"):
                yield Static("ВЫБРАННЫЙ ШАГ", classes="panel-title")
                yield RichLog(id="details", wrap=True, markup=False, highlight=False)
        with Vertical(id="answer-panel"):
            yield Static("ИТОГ НА КЛИЕНТЕ", classes="panel-title")
            yield RichLog(id="answer", wrap=True, markup=False, highlight=False)
        with Horizontal(id="input-row"):
            yield Input(value="Проанализируй SSH-входы за последние сутки и сохрани отчёт на моём компьютере", id="question")
            yield Button("Запустить", id="send", variant="success", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#details", RichLog).write("Выберите шаг слева. Полные запросы и ответы находятся в JSONL-журнале.")
        self.query_one("#answer", RichLog).write(f"Журнал: {self.session_log.path}")
        self.check_servers()

    @work(exclusive=True)
    async def check_servers(self) -> None:
        statuses = []
        try:
            for server in SERVERS:
                discovery = await discover_tools(server)
                names = {tool.name for tool in discovery.tools}
                if server.tool not in names:
                    raise RuntimeError(f"{server.host}: нет инструмента {server.tool}")
                statuses.append(f"{server.role.upper()}: {server.host} ✓ {server.tool}")
                self.session_log.append("TOOLS DISCOVERED", {"server": server.host, "tools": sorted(names)}, server.host)
            self.query_one("#servers", Static).update("\n".join(statuses))
            self.query_one("#status", Static).update("Три сервера готовы. Отправьте запрос.")
            self.ready = True
            self.query_one("#send", Button).disabled = False
        except Exception as error:
            self.query_one("#servers", Static).update("\n".join(statuses))
            self.query_one("#status", Static).update(f"Подключение не удалось: {error}")

    @on(Input.Submitted, "#question")
    @on(Button.Pressed, "#send")
    def send_pressed(self) -> None:
        if not self.ready or self.busy:
            return
        question = self.query_one("#question", Input).value.strip()
        if not question:
            return
        self.busy = True
        self.query_one("#send", Button).disabled = True
        self.query_one("#question", Input).disabled = True
        self.query_one("#steps", OptionList).clear_options()
        self.query_one("#answer", RichLog).clear()
        self.events.clear()
        self.query_one("#status", Static).update("Агент выбирает маршрут…")
        self.run_agent(question)

    def _trace(self, label: str, data: object) -> None:
        server = data.get("server", "client") if isinstance(data, dict) else "client"
        if self.log_error is None:
            try:
                self.session_log.append(label, data, server)
            except OSError as error:
                self.log_error = str(error)
                self.query_one("#status", Static).update(f"Ошибка записи журнала: {error}")
        if label not in VISIBLE:
            return
        if label == "USER":
            title = "ЗАПРОС · " + str(data)
            details = {"question": data}
        elif label == "MCP CALL":
            role = data["server"].rsplit("-", 1)[-1].upper()
            title = f"LLM → {role} / {data['tool']}"
            details = {"Решение LLM": data["model_arguments"], "Маршрут": data["server"],
                       "MCP-инструмент": data["tool"], "Переданный SHA256": data["forwarded_sha256"]}
        elif label == "MCP RESULT":
            summary = data["summary"]
            if data["tool"] == "read_ssh_logins":
                title = f"✓ SOURCE · {summary['event_count']} входов · снимок {summary['source_ref'][:8]}…"
            elif data["tool"] == "analyze_logins":
                title = f"✓ ANALYZE · {summary['total']} входов · {summary['unique_ips']} IP"
            else:
                title = f"✓ REPORT · {summary['bytes']} байт · файл на ВМ"
            details = summary
        elif label == "DOWNLOAD START":
            title = "Клиент скачивает отчёт с ВМ REPORT…"
            details = data
        elif label == "REPORT RECEIVED":
            title = f"✓ ОТЧЁТ ПОЛУЧЕН · {data['local_path']}"
            details = {"Локальный путь": data["local_path"], "Байты": data["bytes"], "SHA256": data["sha256"]}
            self.query_one("#answer", RichLog).write(Text(f"Отчёт получен: {data['local_path']}", style="bold green"))
        elif label == "ANSWER":
            title = "ИТОГОВЫЙ ОТВЕТ"
            details = data
        else:
            title = f"✗ {label} · {data}"
            details = data
        self.events.append((title, details))
        steps = self.query_one("#steps", OptionList)
        follow = steps.highlighted is None or steps.highlighted == steps.option_count - 1
        steps.add_option(Option(Text(title), id=str(len(self.events) - 1)))
        if follow:
            steps.highlighted = len(self.events) - 1
            self.show_step(steps.highlighted)

    @on(OptionList.OptionHighlighted, "#steps")
    def step_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self.show_step(event.option_index)

    def show_step(self, index: int) -> None:
        if not 0 <= index < len(self.events):
            return
        title, data = self.events[index]
        log = self.query_one("#details", RichLog)
        log.clear()
        log.write(Text(title, style="bold yellow"))
        log.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2))

    @work(exclusive=True)
    async def run_agent(self, question: str) -> None:
        try:
            answer = await run_turn(question, self._trace)
            self.query_one("#answer", RichLog).write(Markdown(answer))
            self.query_one("#status", Static).update("Цепочка завершена. Локальный отчёт проверен по размеру и SHA256.")
        except Exception as error:
            self._trace("ERROR", str(error))
            self.query_one("#answer", RichLog).write(Text(f"Ошибка: {error}", style="bold red"))
            self.query_one("#status", Static).update(f"Ошибка: {error}")
        finally:
            self.busy = False
            self.query_one("#send", Button).disabled = False
            self.query_one("#question", Input).disabled = False


if __name__ == "__main__":
    MCPAgentApp().run()
