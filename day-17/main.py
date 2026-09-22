"""Textual MCP browser and agent chat for day 17."""

from __future__ import annotations

import json
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, LoadingIndicator, OptionList, RichLog, Select, Static
from textual.widgets.option_list import Option

from agent import initial_history, run_turn
from mcp_client import Discovery, discover_tools
from servers import SERVERS
from tool_metrics import CHARS_PER_TOKEN, format_metrics, measure_tool, measure_tools
from trace_log import SessionLog


ROOT = Path(__file__).resolve().parent


class MCPAgentApp(App[None]):
    CSS_PATH = Path(__file__).with_name("app.tcss")
    TITLE = "AI Advent · День 17 · MCP агент"
    BINDINGS = [("ctrl+q", "quit", "Выход")]

    def __init__(self, log_dir: Path | None = None) -> None:
        super().__init__()
        self.session_log = SessionLog(log_dir or ROOT / "resources" / "logs")
        self.log_error: str | None = None
        self.busy = False
        self.result: Discovery | None = None
        self.history: list[dict] = []
        self.trace_number = 0

    def compose(self) -> ComposeResult:
        yield Static("ДЕНЬ 17 · MCP И АГЕНТ", id="brand")
        with Horizontal(id="controls"):
            yield Select(
                [(server.name, index) for index, server in enumerate(SERVERS)],
                value=0, allow_blank=False, id="server",
            )
            yield Button("Получить инструменты", id="fetch", variant="primary")
        yield Static(SERVERS[0].address, id="url", markup=False)
        with Horizontal(id="progress"):
            yield LoadingIndicator(id="loading")
            yield Static("Выберите сервер и запросите инструменты.", id="status", markup=False)
        yield Static("Объём tools появится после запроса.", id="metrics", markup=False)
        yield Static(f"Автолог: {self.session_log.path}", id="log-path", markup=False)
        with Horizontal(id="workspace"):
            with Vertical(id="tool-panel"):
                yield Static("Инструменты", id="tool-count", classes="panel-title", markup=False)
                yield OptionList(id="tools")
            with Vertical(id="detail-panel"):
                yield Static("Описание и параметры", classes="panel-title")
                yield RichLog(id="details", wrap=True, markup=False, highlight=False)
        with Horizontal(id="agent-workspace"):
            with Vertical(id="chat-panel"):
                yield Static("ЧАТ С АГЕНТОМ", classes="panel-title")
                yield RichLog(id="chat", wrap=True, markup=False, highlight=False)
            with Vertical(id="trace-panel"):
                yield Static("ЦЕПОЧКА ВЫЗОВОВ · ПОЛНЫЙ JSON", classes="panel-title")
                yield RichLog(id="trace", wrap=True, markup=False, highlight=False)
        with Horizontal(id="input-row"):
            yield Input(placeholder="Спросите об API или документации выбранного сервера…", id="question")
            yield Button("Отправить", id="send", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#loading").display = False
        self._clear_result()
        self._clear_chat()
        self._trace("SESSION START", {"log_path": str(self.session_log.path)})

    def _clear_result(self) -> None:
        self.result = None
        self.query_one("#tools", OptionList).clear_options()
        self.query_one("#tool-count", Static).update("Инструменты")
        self.query_one("#metrics", Static).update("Объём tools появится после запроса.")
        details = self.query_one("#details", RichLog)
        details.clear()
        details.write("После запроса выберите инструмент слева.")
        self.query_one("#send", Button).disabled = True

    def _clear_chat(self) -> None:
        self.history = []
        self.trace_number = 0
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        chat.write("Получите инструменты, затем задайте вопрос.")
        trace = self.query_one("#trace", RichLog)
        trace.clear()
        trace.write("Здесь появятся запросы и ответы LLM, вызовы MCP и результаты.")

    @on(Select.Changed, "#server")
    def server_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, int):
            return
        self.query_one("#url", Static).update(SERVERS[event.value].address)
        self._clear_result()
        self._clear_chat()
        self.query_one("#status", Static).update("Сервер выбран. Нажмите «Получить инструменты».")
        self._trace("SERVER SELECTED", {"server": SERVERS[event.value].name, "address": SERVERS[event.value].address})

    @on(Button.Pressed, "#fetch")
    def fetch_pressed(self) -> None:
        if self.busy:
            return
        index = self.query_one("#server", Select).value
        if not isinstance(index, int):
            return
        self._clear_result()
        self._clear_chat()
        self._trace("TOOLS REQUEST", {"server": SERVERS[index].name, "address": SERVERS[index].address})
        self._set_busy(True)
        self.query_one("#status", Static).update(f"{SERVERS[index].name}: подключение и получение tools…")
        self.fetch_tools(index)

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        self.query_one("#server", Select).disabled = value
        self.query_one("#fetch", Button).disabled = value
        self.query_one("#send", Button).disabled = value or not bool(self.result and self.result.tools)
        self.query_one("#question", Input).disabled = value
        self.query_one("#loading").display = value

    @work(exclusive=True)
    async def fetch_tools(self, index: int) -> None:
        try:
            result = await discover_tools(SERVERS[index])
            self._trace("TOOLS RESPONSE", {
                "server": SERVERS[index].name,
                "protocol_version": result.protocol_version,
                "tools": [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in result.tools],
            })
            self.result = result
            count = len(result.tools)
            metrics = measure_tools(result.tools)
            self.query_one("#tool-count", Static).update(f"Инструменты · {count}")
            self.query_one("#metrics", Static).update(
                f"Объём списка: {format_metrics(metrics)} · оценка {CHARS_PER_TOKEN} символа/токен"
            )
            self.query_one("#status", Static).update(
                f"{SERVERS[index].name}: соединение проверено · "
                f"{result.server_name} {result.server_version} · MCP {result.protocol_version} · tools: {count}"
            )
            options = self.query_one("#tools", OptionList)
            options.add_options([Option(Text(tool.name), id=str(i)) for i, tool in enumerate(result.tools)])
            if count:
                options.highlighted = 0
                self.show_tool(0)
                self.history = initial_history(SERVERS[index])
                chat = self.query_one("#chat", RichLog)
                chat.clear()
                chat.write("Инструменты загружены. Задайте вопрос агенту.")
            else:
                details = self.query_one("#details", RichLog)
                details.clear()
                details.write("Соединение установлено. Сервер вернул пустой список инструментов.")
        except Exception as error:
            self._trace("TOOLS ERROR", str(error))
            self._clear_result()
            self.query_one("#status", Static).update(f"{SERVERS[index].name}: {error}")
        finally:
            self._set_busy(False)

    @on(OptionList.OptionHighlighted, "#tools")
    def tool_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self.show_tool(event.option_index)

    def show_tool(self, index: int) -> None:
        if self.result is None or not 0 <= index < len(self.result.tools):
            return
        tool = self.result.tools[index]
        details = self.query_one("#details", RichLog)
        details.clear()
        details.write(Text(tool.name, style="bold"))
        details.write(f"Объём инструмента: {format_metrics(measure_tool(tool))}")
        if tool.title:
            details.write(tool.title)
        details.write("\n" + (tool.description or "Описание отсутствует."))
        details.write(Text("\nПараметры · inputSchema", style="bold"))
        details.write(json.dumps(tool.input_schema, ensure_ascii=False, indent=2))

    @on(Input.Submitted, "#question")
    @on(Button.Pressed, "#send")
    def send_pressed(self) -> None:
        if self.busy or self.result is None or not self.result.tools:
            return
        field = self.query_one("#question", Input)
        question = field.value.strip()
        if not question:
            return
        field.value = ""
        self.query_one("#chat", RichLog).write(Text("Вы: " + question, style="bold cyan"))
        self._trace("QUESTION SUBMITTED", question)
        self._set_busy(True)
        self.query_one("#status", Static).update("Агент работает · цепочка справа обновляется…")
        index = self.query_one("#server", Select).value
        assert isinstance(index, int)
        self.run_chat(index, question)

    def _trace(self, label: str, data: object) -> None:
        self.trace_number += 1
        if self.log_error is None:
            try:
                selection = self.query_one("#server", Select).value
                server_name = SERVERS[selection].name if isinstance(selection, int) else ""
                self.session_log.append(label, data, server_name)
            except OSError as error:
                self.log_error = str(error)
                self.query_one("#log-path", Static).update(f"Ошибка записи автолога: {error}")
        log = self.query_one("#trace", RichLog)
        log.write(Text(f"#{self.trace_number} · {label}", style="bold yellow"))
        log.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2, default=str))
        log.write("")

    @work(exclusive=True)
    async def run_chat(self, index: int, question: str) -> None:
        try:
            assert self.result is not None
            answer, history = await run_turn(
                SERVERS[index], self.result.tools, self.history, question, self._trace,
            )
            self.history = history
            self._trace("ANSWER DISPLAYED", answer)
            chat = self.query_one("#chat", RichLog)
            chat.write(Text("Агент:", style="bold green"))
            chat.write(Markdown(answer))
            self.query_one("#status", Static).update("Ответ готов · полная цепочка справа.")
        except Exception as error:
            self._trace("ERROR", str(error))
            self.query_one("#chat", RichLog).write(Text(f"Ошибка: {error}", style="bold red"))
            self.query_one("#status", Static).update(f"Ошибка: {error}")
        finally:
            self._set_busy(False)


if __name__ == "__main__":
    MCPAgentApp().run()
