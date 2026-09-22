"""Measure the model-facing size of discovered MCP tool definitions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable

from mcp import types


CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ToolMetrics:
    characters: int
    utf8_bytes: int
    estimated_tokens: int


def tool_payload(tool: types.Tool) -> dict:
    """Return the fields actually received, using their MCP wire aliases."""
    return tool.model_dump(mode="json", by_alias=True, exclude_none=True)


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def measure_text(text: str) -> ToolMetrics:
    characters = len(text)
    return ToolMetrics(
        characters=characters,
        utf8_bytes=len(text.encode("utf-8")),
        estimated_tokens=math.ceil(characters / CHARS_PER_TOKEN),
    )


def measure_tool(tool: types.Tool) -> ToolMetrics:
    return measure_text(compact_json(tool_payload(tool)))


def measure_tools(tools: Iterable[types.Tool]) -> ToolMetrics:
    return measure_text(compact_json([tool_payload(tool) for tool in tools]))


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} Б"
    return f"{size / 1024:.1f} КиБ"


def format_metrics(metrics: ToolMetrics) -> str:
    return (
        f"{metrics.characters:,} симв. · {format_bytes(metrics.utf8_bytes)} · "
        f"≈{metrics.estimated_tokens:,} токенов"
    ).replace(",", " ")
