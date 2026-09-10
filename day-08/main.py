#!/usr/bin/env python3
"""CLI дня 8: предварительный расчёт и реальный эксперимент."""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import Agent, AgentError, AgentConfig, load_config
from experiment import (
    CONTROL_FACT,
    ErrorEvent,
    ExperimentEvent,
    OVERFLOW_MARGIN,
    RequestEvent,
    ResponseEvent,
    SUCCESS_TARGETS,
    ExperimentResult,
    padding_to_target,
    run_experiment,
    save_report,
)
from token_counter import TokenCounter


console = Console()


def token_value(value: int | None) -> str:
    return "—" if value is None else f"{value:,}".replace(",", " ")


def money(value: float | None) -> str:
    return "—" if value is None else f"${value:.8f}"


def percentage(part: int, whole: int) -> str:
    return "—" if whole <= 0 else f"{part / whole * 100:.1f}%"


def multiplier(value: float | int | None, baseline: float | int | None) -> str:
    if value is None or baseline is None or baseline <= 0:
        return "—"
    return f"×{value / baseline:.1f}"


def metric_legend() -> tuple[str, ...]:
    return (
        "Текущий запрос — только новая реплика пользователя; локальный подсчёт.",
        "Вся история — system и все завершённые предыдущие ходы; локальный подсчёт.",
        "Prompt API — вся история и текущий запрос; точное значение OpenRouter.",
        "Ответ модели — completion_tokens из usage API.",
    )


def prompt_excerpt(text: str, edge_lines: int = 3) -> str:
    """Показывает начало, контрольный факт и конец большой реплики."""

    lines = text.splitlines()
    if len(lines) <= edge_lines * 2 + 3:
        return text
    selected = set(range(edge_lines))
    selected.update(range(len(lines) - edge_lines, len(lines)))
    for index, line in enumerate(lines):
        if CONTROL_FACT in line:
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))

    output: list[str] = []
    previous = -1
    for index in sorted(selected):
        if previous >= 0 and index > previous + 1:
            output.append(f"… пропущено строк: {index - previous - 1} …")
        output.append(lines[index])
        previous = index
    return "\n".join(output)


def render_event(event: ExperimentEvent, prompt_display: str) -> None:
    if isinstance(event, RequestEvent):
        console.rule(f"Ход {event.turn} · {event.scenario}", style="cyan")
        roles = " → ".join(message["role"] for message in event.messages)
        console.print(
            f"[bold cyan]→ LLM[/bold cyan] · messages={len(event.messages)} · "
            f"текущий запрос={token_value(event.user_tokens_local)} · "
            f"вся история={token_value(event.history_tokens_local)} · "
            f"prompt локально={token_value(event.prompt_tokens_local)}"
        )
        console.print(f"Роли: {roles}")
        if prompt_display == "full":
            content = json.dumps(event.messages, ensure_ascii=False, indent=2)
            title = "Точный массив messages, отправляемый OpenRouter"
        elif prompt_display == "preview":
            content = prompt_excerpt(event.user_message)
            line_count = len(event.user_message.splitlines())
            title = (
                f"Текущая реплика: {line_count} строк · "
                f"{len(event.user_message):,} символов".replace(",", " ")
            )
        else:
            return
        console.print(Panel(content, title=title, border_style="cyan", expand=False))
        return

    if isinstance(event, ResponseEvent):
        console.print(
            f"[bold green]← LLM[/bold green] · prompt API={token_value(event.prompt_tokens_api)} · "
            f"ответ={token_value(event.completion_tokens_api)} · "
            f"цена={money(event.cost_usd)} · {event.elapsed_seconds:.2f}s"
        )
        console.print(Panel(event.answer, title="Ответ модели", border_style="green", expand=False))
        return

    if isinstance(event, ErrorEvent):
        status = "—" if event.http_status is None else str(event.http_status)
        console.print(f"[bold red]← OpenRouter · HTTP {status} · ответа модели нет[/bold red]")
        console.print(event.error, style="red", markup=False)


def context_table(result: ExperimentResult) -> Table:
    table = Table(
        title=f"Контекст и поведение · {result.model}",
        header_style="bold cyan",
        collapse_padding=True,
        pad_edge=False,
    )
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Сценарий", no_wrap=True)
    table.add_column("Текущий\nзапрос", justify="right", no_wrap=True)
    table.add_column("Вся\nистория", justify="right", no_wrap=True)
    table.add_column("Доля\nистории", justify="right", no_wrap=True)
    table.add_column("Prompt лок/API", justify="right", no_wrap=True)
    table.add_column("Δ", justify="right", no_wrap=True)
    table.add_column("Окно", justify="right", no_wrap=True)
    table.add_column("Свободно", justify="right", no_wrap=True)
    table.add_column("Ответ\nмодели", justify="right", no_wrap=True)
    table.add_column("Время", justify="right", no_wrap=True)
    table.add_column("Finish", no_wrap=True)
    table.add_column("Статус", no_wrap=True)
    for row in result.rows:
        style = "red" if row.status in {"ЛИМИТ", "ОШИБКА", "НЕОЖИДАННО OK"} else None
        measured_prompt = row.prompt_tokens_api or row.prompt_tokens_local
        delta = (
            "—"
            if row.prompt_tokens_api is None
            else f"{row.prompt_tokens_api - row.prompt_tokens_local:+d}"
        )
        free = result.context_limit - measured_prompt
        table.add_row(
            str(row.turn),
            row.scenario,
            token_value(row.user_tokens_local),
            token_value(row.history_tokens_local),
            percentage(row.history_tokens_local, row.prompt_tokens_local),
            f"{token_value(row.prompt_tokens_local)}/{token_value(row.prompt_tokens_api)}",
            delta,
            percentage(measured_prompt, result.context_limit),
            token_value(free),
            token_value(row.completion_tokens_api),
            "—" if row.elapsed_seconds is None else f"{row.elapsed_seconds:.2f}s",
            row.finish_reason or "—",
            row.status,
            style=style,
        )
    return table


def economics_table(result: ExperimentResult) -> Table:
    table = Table(
        title="Токены и стоимость API",
        header_style="bold cyan",
        collapse_padding=True,
        pad_edge=False,
    )
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Вход API", justify="right", no_wrap=True)
    table.add_column("Выход API", justify="right", no_wrap=True)
    table.add_column("Вызов", justify="right", no_wrap=True)
    table.add_column("Кэш", justify="right", no_wrap=True)
    table.add_column("Цена хода", justify="right", no_wrap=True)
    table.add_column("К первому", justify="right", no_wrap=True)
    table.add_column("Σ вход", justify="right", no_wrap=True)
    table.add_column("Σ выход", justify="right", no_wrap=True)
    table.add_column("Σ токены", justify="right", no_wrap=True)
    table.add_column("Σ USD", justify="right", no_wrap=True)
    first_cost = next((row.cost_usd for row in result.rows if row.cost_usd is not None), None)
    cumulative_prompt = 0
    cumulative_completion = 0
    for row in result.rows:
        if row.prompt_tokens_api is not None:
            cumulative_prompt += row.prompt_tokens_api
        if row.completion_tokens_api is not None:
            cumulative_completion += row.completion_tokens_api
        call_total = (
            None
            if row.prompt_tokens_api is None or row.completion_tokens_api is None
            else row.prompt_tokens_api + row.completion_tokens_api
        )
        style = "red" if row.status in {"ЛИМИТ", "ОШИБКА", "НЕОЖИДАННО OK"} else None
        table.add_row(
            str(row.turn),
            token_value(row.prompt_tokens_api),
            token_value(row.completion_tokens_api),
            token_value(call_total),
            token_value(row.cached_tokens_api),
            money(row.cost_usd),
            multiplier(row.cost_usd, first_cost),
            token_value(cumulative_prompt),
            token_value(cumulative_completion),
            token_value(row.cumulative_tokens),
            money(row.cumulative_cost_usd),
            style=style,
        )
    return table


def analysis_lines(result: ExperimentResult) -> tuple[str, ...]:
    successful = [row for row in result.rows if row.prompt_tokens_api is not None]
    first = successful[0]
    last = successful[-1]
    overflow = result.rows[-1]
    total_prompt = sum(row.prompt_tokens_api or 0 for row in successful)
    total_completion = sum(row.completion_tokens_api or 0 for row in successful)
    maximum_delta = max(
        abs((row.prompt_tokens_api or 0) - row.prompt_tokens_local)
        for row in successful
    )
    overflow_by = overflow.prompt_tokens_local - result.context_limit
    return (
        (
            f"Prompt вырос {multiplier(last.prompt_tokens_api, first.prompt_tokens_api)}: "
            f"с {token_value(first.prompt_tokens_api)} до {token_value(last.prompt_tokens_api)} токенов."
        ),
        (
            f"Цена хода выросла {multiplier(last.cost_usd, first.cost_usd)}: "
            f"с {money(first.cost_usd)} до {money(last.cost_usd)}."
        ),
        (
            f"В последнем успешном запросе история занимала "
            f"{percentage(last.history_tokens_local, last.prompt_tokens_local)}, "
            f"а новая реплика — только {token_value(last.user_tokens_local)} токенов."
        ),
        (
            f"Для контекста из {token_value(last.prompt_tokens_api)} токенов повторно обработано "
            f"{token_value(total_prompt)} входных токенов "
            f"({multiplier(total_prompt, last.prompt_tokens_api)} от его текущего размера)."
        ),
        (
            f"Ответы заняли {token_value(total_completion)} из "
            f"{token_value(last.cumulative_tokens)} успешно обработанных токенов; "
            "расход определял входной контекст."
        ),
        (
            f"Максимальное расхождение локального prompt и API: {maximum_delta} токенов. "
            f"Переполненный prompt превысил окно на {token_value(overflow_by)} токенов."
        ),
    )


def overflow_consequences(result: ExperimentResult) -> tuple[str, ...]:
    overflow = result.rows[-1]
    history_room = result.context_limit - overflow.history_tokens_local
    return (
        "Модель не вернула ответ на переполненный ход.",
        "API не вернул usage, поэтому точные токены и цена неудачного хода неизвестны.",
        "Неудачная реплика не добавлена в историю; последняя успешная история сохранена.",
        (
            f"До новой реплики в окне оставалось около {token_value(history_room)} токенов. "
            f"Запрос из {token_value(overflow.user_tokens_local)} токенов нужно сократить "
            "либо освободить контекст; более короткий запрос ещё может поместиться."
        ),
    )


def print_result(result: ExperimentResult, report_path) -> None:
    console.print("[bold]Как читать метрики:[/bold]")
    for line in metric_legend():
        console.print("• " + line)
    console.print()
    console.print(context_table(result))
    console.print()
    console.print(economics_table(result))
    successful = [row for row in result.rows if row.prompt_tokens_api is not None]
    cached = sum(row.cached_tokens_api or 0 for row in successful)
    providers = ", ".join(sorted({row.provider for row in successful if row.provider})) or "—"
    console.print(
        f"\nУспешно обработано: {token_value(successful[-1].cumulative_tokens)} токенов · "
        f"кэш: {token_value(cached)} · стоимость: {money(successful[-1].cumulative_cost_usd)}"
    )
    sources = ", ".join(sorted({row.cost_source for row in successful if row.cost_source}))
    console.print(f"Модель: [bold]{result.model}[/bold] · провайдер: {providers}")
    console.print(f"Источник цены: {sources or '—'}")
    console.print("\n[bold]Автоматический анализ:[/bold]")
    for line in analysis_lines(result):
        console.print("• " + line)
    overflow = result.rows[-1]
    if overflow.error:
        console.print(f"\n[bold red]Ошибка переполнения:[/bold red] {overflow.error}")
    console.print("\n[bold red]Что сломалось:[/bold red]")
    for line in overflow_consequences(result):
        console.print("• " + line)
    console.print(
        "\nКонтрольный факт: "
        + ("[green]найден[/green]" if result.fact_recalled else "[red]потерян[/red]")
    )
    console.print(
        "Переполнение модели: "
        + ("[green]подтверждено[/green]" if result.overflow_confirmed else "[red]не подтверждено[/red]")
    )
    console.print(
        "История после отказа: "
        + (
            "[green]не изменилась[/green]"
            if result.history_unchanged_after_error
            else "[red]изменилась[/red]"
        )
    )
    console.print(f"Отчёт: {report_path}")
    console.print("Итог: " + ("[bold green]PASS[/bold green]" if result.passed else "[bold red]FAIL[/bold red]"))


def print_preview(config: AgentConfig, counter: TokenCounter) -> None:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": "Это короткий диалог. Ответь ровно: OK-1"},
        {"role": "assistant", "content": "OK-1"},
    ]
    table = Table(title="Предварительный локальный расчёт", header_style="bold cyan")
    table.add_column("Ход", justify="right")
    table.add_column("Сценарий")
    table.add_column("Цель prompt", justify="right")
    table.add_column("Оценка prompt", justify="right")
    table.add_row("1", "короткий", "—", token_value(counter.count_messages(messages[:-1])))

    for index, target in enumerate(SUCCESS_TARGETS, start=2):
        message = padding_to_target(
            counter,
            tuple(messages),
            target_prompt_tokens=target,
            step=index,
            expected=f"OK-{index}",
            include_fact=target == 9_000,
        )
        candidate = [*messages, {"role": "user", "content": message}]
        table.add_row(str(index), "длинный", token_value(target), token_value(counter.count_messages(candidate)))
        messages.extend((
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"OK-{index}"},
        ))

    recall = (
        "Какое значение было указано после слов «КОНТРОЛЬНЫЙ ФАКТ»? "
        f"Ответь ровно: {CONTROL_FACT}"
    )
    messages.append({"role": "user", "content": recall})
    table.add_row("6", "проверка памяти", "—", token_value(counter.count_messages(messages)))
    messages.append({"role": "assistant", "content": CONTROL_FACT})
    overflow_target = config.context_limit + OVERFLOW_MARGIN
    overflow = padding_to_target(
        counter,
        tuple(messages),
        target_prompt_tokens=overflow_target,
        step=99,
        expected="OVERFLOW-SHOULD-NOT-ANSWER",
        at_least=True,
    )
    estimate = counter.count_messages([*messages, {"role": "user", "content": overflow}])
    table.add_row("7", "переполнение", f"≥ {token_value(overflow_target)}", token_value(estimate), style="red")
    console.print(table)
    console.print(
        f"Модель: {config.model} · окно: {config.context_limit} · "
        f"резерв ответа: {config.max_tokens} · реальных API-вызовов: 7"
    )
    console.print("Локальные значения — предварительная оценка; точные токены вернёт usage API.")


async def execute(config: AgentConfig, prompt_display: str) -> int:
    counter = TokenCounter(config.model)
    timeout = httpx.Timeout(120, connect=20)
    console.print(
        f"Тестируемая модель: [bold]{config.model}[/bold] · "
        f"OpenRouter · окно {token_value(config.context_limit)} · "
        f"max_tokens={config.max_tokens} · context compression OFF · fallback OFF"
    )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        agent = Agent(config, client, counter)
        try:
            result = await run_experiment(
                agent,
                counter,
                observer=lambda event: render_event(event, prompt_display),
            )
        except AgentError as error:
            console.print(f"[bold red]Эксперимент остановлен:[/bold red] {error}")
            return 1
    report_path = save_report(result)
    print_result(result, report_path)
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="День 8: рост токенов, стоимости и реальное переполнение контекста."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preview", help="показать локальный план без API-вызовов")
    experiment = subparsers.add_parser(
        "experiment",
        help="выполнить 7 последовательных запросов к OpenRouter",
    )
    experiment.add_argument(
        "--confirm-api-calls",
        action="store_true",
        help="подтвердить платные внешние вызовы",
    )
    experiment.add_argument(
        "--prompt-display",
        choices=("preview", "full", "none"),
        default="preview",
        help=(
            "показывать фрагменты реплик (preview), точный массив messages (full) "
            "или только метрики (none)"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "preview":
        config = AgentConfig(api_key="preview-only")
        counter = TokenCounter(config.model)
        print_preview(config, counter)
        return 0
    if not args.confirm_api_calls:
        parser.error("для внешних вызовов добавь --confirm-api-calls")
    try:
        config = load_config()
        config.check()
    except ValueError as error:
        parser.error(str(error))
    return asyncio.run(execute(config, args.prompt_display))


if __name__ == "__main__":
    raise SystemExit(main())
