"""C1/C7/C8: панель, API ассистента, защита и персистентность (офлайн)."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from server import config
from server.app import create_app

SECRET = "canary-assistant-api-key"
CSRF = {"X-Support-Request": "1"}


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
        ]
    }


def full_script():
    return [
        tool_response([("read_file", {"path": "web/src/app.js"})]),
        tool_response([("propose_edit", {"path": "web/src/app.js", "content": "// api app\n"})]),
        tool_response([("propose_edit", {"path": "web/index.html", "content": "<html>api</html>\n"})]),
        tool_response([("run_checks", {})]),
        tool_response([("apply", {})]),
        tool_response([("finish", {"summary": "Интерфейс обновлён."})]),
    ]


def app_for(db_path, project_root, script, **kwargs):
    state = {"calls": 0}

    async def handler(request):
        index = state["calls"]
        state["calls"] += 1
        return httpx.Response(200, json=script[min(index, len(script) - 1)])

    app = create_app(
        db_path=db_path,
        deepseek_settings=config.DeepSeekSettings(api_key=SECRET),
        assistant_transport=httpx.MockTransport(handler),
        assistant_project_root=project_root,
        **kwargs,
    )
    return app, state


def new_conversation(client):
    response = client.post("/api/assistant/conversations", headers=CSRF)
    assert response.status_code == 200, response.text
    return response.json()["conversationId"]


def send(client, conversation_id, text="Добавь подпись", key="key-1"):
    return client.post(
        f"/api/assistant/conversations/{conversation_id}/messages",
        headers=CSRF,
        json={"text": text, "idempotencyKey": key},
    )


def test_static_panel_served_from_one_origin(tmp_path):
    root = make_project(tmp_path)
    app, _ = app_for(tmp_path / "db.sqlite", root, full_script())
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="assistant-open"' in page.text
    assert 'id="assistant-panel"' in page.text
    assert 'src="./src/assistant.js"' in page.text
    script = client.get("/src/assistant.js")
    assert script.status_code == 200
    assert script.headers["cache-control"] == "no-store"


def test_mutating_api_requires_header_and_origin(tmp_path):
    root = make_project(tmp_path)
    app, _ = app_for(tmp_path / "db.sqlite", root, full_script())
    client = TestClient(app)
    missing = client.post("/api/assistant/conversations")
    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "missing_request_header"
    wrong = client.post(
        "/api/assistant/conversations",
        headers={**CSRF, "Origin": "http://evil.example"},
    )
    assert wrong.status_code == 403
    assert wrong.json()["error"]["code"] == "invalid_origin"
    assert client.post("/api/assistant/conversations", headers=CSRF).status_code == 200


def test_operation_persists_history_and_result(tmp_path):
    root = make_project(tmp_path)
    db_path = tmp_path / "db.sqlite"
    app, _ = app_for(db_path, root, full_script())
    client = TestClient(app)
    conversation_id = new_conversation(client)
    response = send(client, conversation_id)
    assert response.status_code == 200, response.text
    body = response.json()
    result = body["assistantMessage"]["result"]
    assert result["applied"] is True
    assert result["changedPaths"] == ["web/index.html", "web/src/app.js"]
    assert result["checks"]["passed"] is True
    assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// api app\n"
    assert SECRET not in response.text

    cookie = client.cookies.get("gt_session")
    # Новая страница/сервер читает сохранённую историю без повторного запуска.
    app2, state2 = app_for(db_path, root, full_script())
    restored = TestClient(app2)
    restored.cookies.set("gt_session", cookie)
    history = restored.get(f"/api/assistant/conversations/{conversation_id}/messages")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["result"]["applied"] is True
    assert state2["calls"] == 0

    with sqlite3.connect(str(db_path)) as conn:
        dump = "\n".join(conn.iterdump())
    assert SECRET not in dump


def test_idempotent_repeat_and_conflict(tmp_path):
    root = make_project(tmp_path)
    app, state = app_for(tmp_path / "db.sqlite", root, full_script())
    client = TestClient(app)
    conversation_id = new_conversation(client)
    first = send(client, conversation_id, key="same")
    assert first.status_code == 200
    calls_after_first = state["calls"]

    repeat = send(client, conversation_id, key="same")
    assert repeat.status_code == 200
    assert repeat.json()["userMessage"]["id"] == first.json()["userMessage"]["id"]
    assert state["calls"] == calls_after_first

    conflict = send(client, conversation_id, text="Другая цель", key="same")
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"


def test_missing_provider_key_is_safe_error(tmp_path):
    root = make_project(tmp_path)
    app = create_app(
        db_path=tmp_path / "db.sqlite",
        deepseek_settings=config.DeepSeekSettings(api_key=""),
        assistant_project_root=root,
    )
    client = TestClient(app)
    conversation_id = new_conversation(client)
    response = send(client, conversation_id)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_key_missing"
    history = client.get(f"/api/assistant/conversations/{conversation_id}/messages").json()["messages"]
    assert history[0]["state"] == "error"
    assert history[1]["state"] == "error"


def test_sessions_are_isolated(tmp_path):
    root = make_project(tmp_path)
    app, _ = app_for(tmp_path / "db.sqlite", root, full_script())
    owner = TestClient(app)
    stranger = TestClient(app)
    conversation_id = new_conversation(owner)
    assert send(owner, conversation_id).status_code == 200
    stranger.get("/api/assistant/session")
    assert stranger.get(
        f"/api/assistant/conversations/{conversation_id}/messages"
    ).status_code == 404
    assert send(stranger, conversation_id).status_code == 404
