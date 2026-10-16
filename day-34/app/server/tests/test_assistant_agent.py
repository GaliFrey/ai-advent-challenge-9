"""C2/C9/C11: серверный агентный цикл с инструментами (офлайн, подменённый транспорт)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from server import assistant_agent, assistant_staging, assistant_tools
from server.config import AssistantSettings
from server.errors import SupportError

SECRET = "canary-assistant-key-DoNotLeak"


def make_project(tmp_path):
    (tmp_path / "web" / "src").mkdir(parents=True)
    (tmp_path / "web" / "test").mkdir(parents=True)
    (tmp_path / "web" / "index.html").write_text("<html>old</html>\n", encoding="utf-8")
    (tmp_path / "web" / "style.css").write_text("body{}\n", encoding="utf-8")
    for name in ("app.js", "capture.js", "display.js", "pitch.js", "diagnostics.js", "support.js"):
        (tmp_path / "web" / "src" / name).write_text(f"// {name}\n", encoding="utf-8")
    (tmp_path / "web" / "test" / "sample.test.mjs").write_text(
        'import test from "node:test"\ntest("ok", () => {})\n', encoding="utf-8"
    )
    return tmp_path


def tool_response(calls):
    return {
        "model": "deepseek-flash",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                },
            }
        ],
    }


def stop_response(text):
    return {
        "model": "deepseek-flash",
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
    }


def run_script(root, script, settings=None):
    staging = assistant_staging.Staging.create(root)
    tools = assistant_tools.AssistantTools(staging, settings=settings or AssistantSettings())
    payloads = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        index = len(payloads) - 1
        return httpx.Response(200, json=script[min(index, len(script) - 1)])

    async def run():
        return await assistant_agent.run_agent(
            "Поправь подписи в интерфейсе",
            settings=settings or AssistantSettings(),
            api_key=SECRET,
            tools=tools,
            transport=httpx.MockTransport(handler),
        )

    try:
        result = asyncio.run(run())
    finally:
        staging.cleanup()
    return result, payloads


def test_agent_cycle_applies_only_after_check(tmp_path):
    root = make_project(tmp_path)
    script = [
        tool_response([("list_files", {})]),
        tool_response([("read_file", {"path": "web/src/app.js"})]),
        tool_response([("read_file", {"path": "web/style.css"})]),
        tool_response([("propose_edit", {"path": "web/src/app.js", "content": "// new app\n"})]),
        tool_response([("propose_edit", {"path": "web/index.html", "content": "<html>new</html>\n"})]),
        tool_response([("propose_edit", {"path": "web/style.css", "content": "body{color:red}\n"})]),
        tool_response([("run_checks", {})]),
        tool_response([("apply", {})]),
        tool_response([("finish", {"summary": "Обновил интерфейс."})]),
    ]
    result, payloads = run_script(root, script)
    assert result["applied"] is True
    assert result["status"] == "applied"
    assert result["changedPaths"] == ["web/index.html", "web/src/app.js", "web/style.css"]
    assert "web/src/app.js" in result["diff"]
    assert result["checks"]["passed"] is True
    assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// new app\n"
    assert (root / "web" / "index.html").read_text(encoding="utf-8") == "<html>new</html>\n"
    assert (root / "web" / "style.css").read_text(encoding="utf-8") == "body{color:red}\n"
    # Цикл вызвал инструменты, а не заранее заданный сценарий подстановки.
    assert len(payloads) == len(script)
    assert all("tools" in payload for payload in payloads)
    assert all("response_format" not in payload for payload in payloads)
    assert all(payload["thinking"] == {"type": "disabled"} for payload in payloads)
    assert SECRET not in json.dumps(payloads)


def test_different_scripts_change_different_files(tmp_path):
    root = make_project(tmp_path)
    script = [
        tool_response([("read_file", {"path": "web/src/pitch.js"})]),
        tool_response([("propose_edit", {"path": "web/src/pitch.js", "content": "// new pitch\n"})]),
        tool_response([("run_checks", {})]),
        tool_response([("apply", {})]),
        tool_response([("finish", {"summary": "Изменил распознавание."})]),
    ]
    result, _ = run_script(root, script)
    assert result["changedPaths"] == ["web/src/pitch.js"]
    assert (root / "web" / "src" / "pitch.js").read_text(encoding="utf-8") == "// new pitch\n"


def test_short_path_single_small_change(tmp_path):
    root = make_project(tmp_path)
    content = 'import { detectPitch } from "./pitch.js"\nexport const label = "new"\n'
    script = [
        tool_response([("read_file", {"path": "web/src/app.js"})]),
        tool_response([("propose_edit", {"path": "web/src/app.js", "content": content})]),
        tool_response([("run_checks", {})]),
        tool_response([("apply", {})]),
        tool_response([("finish", {"summary": "Внёс небольшое изменение."})]),
    ]
    result, payloads = run_script(root, script)
    assert result["applied"] is True
    assert result["changedPaths"] == ["web/src/app.js"]
    # Целевая проверка реально прогнала node --check по изменённому ES-модулю.
    assert result["checks"]["passed"] is True
    # Короткий путь: одно чтение, одна правка, проверка, применение, итог.
    assert len(payloads) == 5
    assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == content


def test_output_limit_reports_error_without_changes(tmp_path):
    root = make_project(tmp_path)
    settings = AssistantSettings()
    staging = assistant_staging.Staging.create(root)
    tools = assistant_tools.AssistantTools(staging, settings=settings)
    payloads = []

    def length_response():
        return {
            "model": "deepseek-flash",
            "choices": [
                {"finish_reason": "length", "message": {"role": "assistant", "content": "x"}}
            ],
        }

    async def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=length_response())

    async def run():
        return await assistant_agent.run_agent(
            "Поправь app.js",
            settings=settings,
            api_key=SECRET,
            tools=tools,
            transport=httpx.MockTransport(handler),
        )

    try:
        with pytest.raises(SupportError) as error:
            asyncio.run(run())
        assert error.value.code == "provider_output_limit"
        assert tools.applied is False
        assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// app.js\n"
        assert SECRET not in json.dumps(payloads)
    finally:
        staging.cleanup()


def test_request_uses_configured_max_tokens(tmp_path):
    root = make_project(tmp_path)
    settings = AssistantSettings()
    assert settings.max_tokens == 4000
    assert settings.max_steps == 40
    _, payloads = run_script(root, [stop_response("ok")], settings=settings)
    assert payloads[0]["max_tokens"] == settings.max_tokens == 4000
    assert payloads[0]["stream"] is False
    assert payloads[0]["thinking"] == {"type": "disabled"}
    assert "response_format" not in payloads[0]


def test_apply_failure_after_conflict_keeps_working_frontend(tmp_path):
    root = make_project(tmp_path)
    script = [
        tool_response([("propose_edit", {"path": "web/src/app.js", "content": "// staged\n"})]),
        tool_response([("run_checks", {})]),
        tool_response([("apply", {})]),
        tool_response([("finish", {"summary": "Готово."})]),
    ]

    # Внешнее изменение между staging и apply имитируется через вторую staging-копию:
    # сначала готовим, затем меняем рабочий файл до применения.
    staging = assistant_staging.Staging.create(root)
    tools = assistant_tools.AssistantTools(staging, settings=AssistantSettings())
    payloads = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 3:
            (root / "web" / "src" / "app.js").write_text("// external\n", encoding="utf-8")
        index = len(payloads) - 1
        return httpx.Response(200, json=script[min(index, len(script) - 1)])

    async def run():
        return await assistant_agent.run_agent(
            "Поправь app.js",
            settings=AssistantSettings(),
            api_key=SECRET,
            tools=tools,
            transport=httpx.MockTransport(handler),
        )

    try:
        result = asyncio.run(run())
    finally:
        staging.cleanup()
    # apply вернул ошибку в tool-результат, apply не выполнено, finish дал no_changes.
    assert result["applied"] is False
    assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// external\n"
    tool_messages = [p["messages"][-1] for p in payloads]
    assert any("apply_conflict" in json.dumps(m) for m in tool_messages)


def test_step_limit_stops_the_loop(tmp_path):
    root = make_project(tmp_path)
    settings = AssistantSettings(max_steps=2)
    staging = assistant_staging.Staging.create(root)
    tools = assistant_tools.AssistantTools(staging, settings=settings)

    async def handler(request):
        return httpx.Response(200, json=tool_response([("list_files", {})]))

    async def run():
        return await assistant_agent.run_agent(
            "Цель", settings=settings, api_key=SECRET, tools=tools,
            transport=httpx.MockTransport(handler),
        )

    try:
        with pytest.raises(SupportError) as error:
            asyncio.run(run())
        assert error.value.code == "assistant_steps"
        assert tools.applied is False
        assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// app.js\n"
    finally:
        staging.cleanup()


def test_stop_without_finish_reports_no_changes(tmp_path):
    root = make_project(tmp_path)
    result, _ = run_script(root, [stop_response("Изменения не нужны.")])
    assert result["applied"] is False
    assert result["status"] == "no_changes"
    assert result["summary"] == "Изменения не нужны."


def test_read_limit_enforced(tmp_path):
    root = make_project(tmp_path)
    settings = AssistantSettings(max_read_chars=4)
    result, _ = run_script(
        root,
        [tool_response([("read_file", {"path": "web/src/app.js"})]), stop_response("ok")],
        settings=settings,
    )
    assert result["status"] == "no_changes"
