"""Standalone Textual browser for public MCP tool descriptions."""

from __future__ import annotations

import json
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, LoadingIndicator, OptionList, RichLog, Select, Static
from textual.widgets.option_list import Option

from mcp_client import Discovery, DiscoveryError, discover_tools
from servers import SERVERS
from tool_metrics import CHARS_PER_TOKEN, format_metrics, measure_tool, measure_tools


class MCPBrowser(App[None]):
    CSS_PATH = Path(__file__).with_name("app.tcss")
    TITLE = "AI Advent · День 16 · MCP"
    BINDINGS = [("ctrl+q", "quit", "Выход")]

    def __init__(self) -> None:
        super().__init__()
        self.busy = False
        self.result: Discovery | None = None

    def compose(self) -> ComposeResult:
        yield Static("ДЕНЬ 16 · ИНСТРУМЕНТЫ MCP", id="brand")
        with Horizontal(id="controls"):
            yield Select(
                [(server.name, index) for index, server in enumerate(SERVERS)],
                value=0, allow_blank=False, id="server",
            )
            yield Button("Получить инструменты", id="fetch", variant="primary")
        yield Static(SERVERS[0].url, id="url", markup=False)
        with Horizontal(id="progress"):
            yield LoadingIndicator(id="loading")
            yield Static("Выберите сервер и запросите инструменты.", id="status", markup=False)
        yield Static("Объём tools появится после запроса.", id="metrics", markup=False)
        with Horizontal(id="workspace"):
            with Vertical(id="tool-panel"):
                yield Static("Инструменты", id="tool-count", classes="panel-title", markup=False)
                yield OptionList(id="tools")
            with Vertical(id="detail-panel"):
                yield Static("Описание и параметры", classes="panel-title")
                yield RichLog(id="details", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#loading").display = False
        self._clear_result()

    def _clear_result(self) -> None:
        self.result = None
        self.query_one("#tools", OptionList).clear_options()
        self.query_one("#tool-count", Static).update("Инструменты")
        self.query_one("#metrics", Static).update("Объём tools появится после запроса.")
        details = self.query_one("#details", RichLog)
        details.clear()
        details.write("После запроса выберите инструмент слева.")

    @on(Select.Changed, "#server")
    def server_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, int):
            return
        self.query_one("#url", Static).update(SERVERS[event.value].url)
        self._clear_result()
        self.query_one("#status", Static).update("Сервер выбран. Нажмите «Получить инструменты».")

    @on(Button.Pressed, "#fetch")
    def fetch_pressed(self) -> None:
        if self.busy:
            return
        index = self.query_one("#server", Select).value
        if not isinstance(index, int):
            return
        self._clear_result()
        self._set_busy(True)
        self.query_one("#status", Static).update(f"{SERVERS[index].name}: подключение и получение tools…")
        self.fetch_tools(index)

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        self.query_one("#server", Select).disabled = value
        self.query_one("#fetch", Button).disabled = value
        self.query_one("#loading").display = value

    @work(exclusive=True)
    async def fetch_tools(self, index: int) -> None:
        try:
            result = await discover_tools(SERVERS[index])
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
                options.focus()
            else:
                details = self.query_one("#details", RichLog)
                details.clear()
                details.write("Соединение установлено. Сервер вернул пустой список инструментов.")
        except DiscoveryError as error:
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


if __name__ == "__main__":
    MCPBrowser().run()
