"""Интеграционные проверки C1–C9 задачи T008 (настоящий MCP через stdio).

Проверки используют официальный Python SDK ``mcp``: backend запускает
``server.mcp_server`` дочерним процессом тем же интерпретатором uv-окружения и
реально проходит ``initialize`` → ``tools/list`` → ``tools/call``. Успешный
путь проверяется настоящим сервером, а не mock-клиентом. Для сбоев, таймаутов и
жизненного цикла используются отдельные тестовые stdio-серверы из
``server/tests/mcp_fixtures``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp import Client, StdioServerParameters

from server.app import create_app
from server.tests.conftest import MOCK_ANSWER
from server.config import PROJECT_ROOT
from server.db import Database, now_iso, open_readonly
from server.mcp_client import McpError, fetch_ticket_context_result
from server.tests.conftest import CSRF, make_client

_FIXTURES = "server.tests.mcp_fixtures"
_ISO_MS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


# --- вспомогательные функции -------------------------------------------------


def valid_snapshot() -> dict:
    return {
        "schemaVersion": 1,
        "collectedAt": "2026-10-14T12:00:00.000Z",
        "capture": {"status": "capturing", "lastError": None},
        "audio": {
            "trackState": "live",
            "trackEnabled": True,
            "trackMuted": False,
            "contextState": "running",
        },
        "device": {"actualLabel": None},
        "measurement": {
            "rms": 0.083,
            "frequencyHz": 109.49,
            "matchedString": "A2",
            "ageMs": 12,
        },
        "display": {"string": "A2", "cents": -8, "direction": "Подтянуть", "tuned": False},
        "commit": {"ageMs": 345},
    }


def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def create_conversation(db_path, session_id: str = "owner") -> int:
    Database(db_path)  # создаёт схему
    conn = _connect(db_path)
    try:
        timestamp = now_iso()
        cursor = conn.execute(
            "INSERT INTO conversations (session_id, created_at, updated_at) "
            "VALUES (?, ?, ?)",
            (session_id, timestamp, timestamp),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def add_message(
    db_path,
    conversation_id: int,
    *,
    role: str,
    text: str,
    state: str,
    created_at: str | None = None,
    error_code: str | None = None,
    reply_message_id: int | None = None,
) -> int:
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO messages "
            "(conversation_id, role, text, state, error_code, reply_message_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                text,
                state,
                error_code,
                reply_message_id,
                created_at or now_iso(),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def add_snapshot(db_path, message_id: int, snapshot: dict, created_at: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO snapshots (message_id, payload, created_at) VALUES (?, ?, ?)",
            (message_id, json.dumps(snapshot, ensure_ascii=False), created_at),
        )
        conn.commit()
    finally:
        conn.close()


def fetch(db_path, conversation_id: int, message_id: int, **kwargs):
    async def main():
        return await fetch_ticket_context_result(
            db_path=db_path,
            conversation_id=conversation_id,
            message_id=message_id,
            **kwargs,
        )

    return anyio.run(main)


def _params(db_path, conversation_id: int, message_id: int) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-B",
            "-m",
            "server.mcp_server",
            "--db",
            str(db_path),
            "--conversation-id",
            str(conversation_id),
            "--message-id",
            str(message_id),
        ],
        cwd=str(PROJECT_ROOT),
    )


def new_conversation(client: TestClient) -> int:
    response = client.post("/api/support/conversations", headers=CSRF)
    assert response.status_code == 200, response.text
    return response.json()["conversationId"]


def send(
    client: TestClient,
    conversation_id: int,
    *,
    text: str = "Привет, тюнер",
    snapshot: dict | None = None,
    key: str = "key-1",
):
    return client.post(
        f"/api/support/conversations/{conversation_id}/messages",
        headers=CSRF,
        json={
            "text": text,
            "snapshot": snapshot if snapshot is not None else valid_snapshot(),
            "idempotencyKey": key,
        },
    )


def _seed_current(db_path, *, session_id: str = "owner"):
    conversation_id = create_conversation(db_path, session_id)
    message_id = add_message(
        db_path,
        conversation_id,
        role="user",
        text="текущий вопрос",
        state="processing",
        created_at="2026-10-14T12:00:00.000Z",
    )
    add_snapshot(db_path, message_id, valid_snapshot(), "2026-10-14T12:00:00.100Z")
    return conversation_id, message_id


def _child_pids(marker: str) -> list[int]:
    if not Path("/proc").exists():  # pragma: no cover - окружение без /proc
        return []
    me = os.getpid()
    found: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            if ppid != me:
                continue
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                "utf-8", "replace"
            )
        except (OSError, IndexError, ValueError):
            continue
        if marker in cmdline:
            found.append(pid)
    return found


def _wait_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"дочерний процесс {pid} не завершился")


# --- C1: настоящий протокольный цикл -----------------------------------------


def test_c1_real_initialize_tools_list_tools_call(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_current(db_path)

    result = fetch(db_path, conversation_id, message_id)

    # initialize завершён (protocol_version доступен только после рукопожатия).
    assert result.protocol_version == "2025-11-25"
    # tools/list вернул ровно ожидаемый инструмент.
    assert result.tool_names == ["get_ticket_context"]
    # tools/call вернул контекст именно этого сообщения.
    assert result.context["messageId"] == message_id
    assert result.context["conversationId"] == conversation_id


def test_real_child_disables_bytecode_even_with_explicit_environment(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_current(db_path)
    result = fetch(db_path, conversation_id, message_id,
                   server_module=_FIXTURES + ".no_bytecode_server",
                   extra_env={"PYTHONDONTWRITEBYTECODE": "0"})
    assert result.context["messageId"] == message_id


# --- C2: контракт результата и снимок ----------------------------------------


def test_c2_contract_fields_and_snapshot_not_recomputed(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_current(db_path)
    snapshot = valid_snapshot()
    snapshot["measurement"]["ageMs"] = 123
    snapshot["commit"]["ageMs"] = 456
    received_at = "2026-10-14T12:00:05.250Z"
    conn = _connect(db_path)
    conn.execute(
        "UPDATE snapshots SET payload = ?, created_at = ? WHERE message_id = ?",
        (json.dumps(snapshot, ensure_ascii=False), received_at, message_id),
    )
    conn.commit()
    conn.close()

    context = fetch(db_path, conversation_id, message_id).context

    assert context["contextVersion"] == 1
    assert context["conversationId"] == conversation_id
    assert context["messageId"] == message_id
    assert context["snapshotSchemaVersion"] == 1
    # Снимок возвращается как есть: пересчёта ageMs и усечения нет.
    assert context["snapshot"] == snapshot
    assert context["snapshot"]["measurement"]["ageMs"] == 123
    assert context["snapshot"]["commit"]["ageMs"] == 456
    assert context["snapshotReceivedAt"] == received_at
    assert _ISO_MS.match(context["snapshotReceivedAt"])
    assert context["history"][-1]["id"] == message_id
    assert context["history"][-1]["role"] == "user"
    assert context["history"][-1]["state"] == "processing"
    assert context["historyIncludedCount"] == 1
    assert context["historyTruncated"] is False
    assert context["historyOmittedCount"] == 0
    assert context["limits"] == {"maxMessages": 20, "maxTextChars": 16000}


# --- C3: область, аргументы, read-only ---------------------------------------


def test_c3_readonly_connection_rejects_writes(tmp_path):
    db_path = tmp_path / "support.db"
    create_conversation(db_path)
    conn = open_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO messages (conversation_id, role, text, state, created_at) "
                "VALUES (1, 'user', 'x', 'processing', '2026-10-14T12:00:00.000Z')"
            )
    finally:
        conn.close()


def test_c3_scope_is_one_conversation_and_message(tmp_path):
    db_path = tmp_path / "support.db"
    conv_a, msg_a = _seed_current(db_path, session_id="a")
    conv_b, msg_b = _seed_current(db_path, session_id="b")

    context_a = fetch(db_path, conv_a, msg_a).context
    context_b = fetch(db_path, conv_b, msg_b).context

    assert context_a["conversationId"] == conv_a
    assert context_a["messageId"] == msg_a
    assert context_a["snapshot"] == valid_snapshot()
    assert context_b["conversationId"] == conv_b
    assert context_b["messageId"] == msg_b

    # Чужое сообщение в разрешённой области обращения — явная ошибка, не пустой успех.
    with pytest.raises(McpError):
        fetch(db_path, conv_a, msg_b)


def test_c3_tool_arguments_cannot_expand_scope(tmp_path):
    db_path = tmp_path / "support.db"
    conv_a, msg_a = _seed_current(db_path, session_id="a")
    conv_b, msg_b = _seed_current(db_path, session_id="b")

    async def call_with_foreign_scope():
        async with Client(_params(db_path, conv_a, msg_a), mode="legacy") as client:
            return await client.call_tool(
                "get_ticket_context",
                {"conversationId": conv_b, "messageId": msg_b, "db": str(db_path)},
            )

    result = anyio.run(call_with_foreign_scope)
    context = result.structured_content
    assert context["conversationId"] == conv_a
    assert context["messageId"] == msg_a


def test_c3_missing_snapshot_is_error(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id = create_conversation(db_path)
    message_id = add_message(
        db_path,
        conversation_id,
        role="user",
        text="без снимка",
        state="processing",
    )
    with pytest.raises(McpError):
        fetch(db_path, conversation_id, message_id)


# --- C4: изоляция сессий на HTTP-границе -------------------------------------


def test_c4_sessions_isolated_around_mcp(db_path):
    with make_client(db_path) as owner, make_client(db_path) as stranger:
        conversation_id = new_conversation(owner)

        # Чужая сессия не получает доступ к обращению владельца.
        assert stranger.get(
            f"/api/support/conversations/{conversation_id}/messages"
        ).status_code == 404
        assert send(stranger, conversation_id, key="stranger").status_code == 404
        assert Database(db_path).message_count(conversation_id) == 0

        # Владелец получает ответ, построенный из MCP-контекста.
        ok = send(owner, conversation_id, key="owner")
        assert ok.status_code == 200, ok.text
        assert ok.json()["assistantMessage"]["text"].startswith(MOCK_ANSWER)

        # Чужая сессия по-прежнему не видит данные.
        assert stranger.get(
            f"/api/support/conversations/{conversation_id}/messages"
        ).status_code == 404


# --- C5: история и совместные лимиты -----------------------------------------


def _seed_history(db_path):
    conversation_id = create_conversation(db_path)
    base = "2026-10-14T12:00:{:02d}.000Z"
    m1 = add_message(db_path, conversation_id, role="user", text="u1", state="answered", created_at=base.format(1))
    m2 = add_message(db_path, conversation_id, role="assistant", text="a1", state="answered", created_at=base.format(2), reply_message_id=m1)
    m3 = add_message(db_path, conversation_id, role="user", text="u2", state="answered", created_at=base.format(3))
    m4 = add_message(db_path, conversation_id, role="assistant", text="a2", state="answered", created_at=base.format(4), reply_message_id=m3)
    m5 = add_message(db_path, conversation_id, role="user", text="u3", state="error", created_at=base.format(5), error_code="mcp_error")
    m6 = add_message(db_path, conversation_id, role="user", text="текущий", state="processing", created_at=base.format(6))
    add_snapshot(db_path, m6, valid_snapshot(), base.format(7))
    return conversation_id, m6


def test_c5_limit_by_message_count_keeps_order_and_current(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_history(db_path)

    context = fetch(
        db_path, conversation_id, message_id, max_messages=3, max_text_chars=100000
    ).context

    assert [item["id"] for item in context["history"]] == [message_id - 2, message_id - 1, message_id]
    assert context["history"][-1]["id"] == message_id
    assert context["history"][-1]["state"] == "processing"
    # user в state=error включён с явными state/errorCode.
    error_items = [item for item in context["history"] if item["state"] == "error"]
    assert error_items and error_items[0]["errorCode"] == "mcp_error"
    # assistant включается только в состоянии answered.
    assert all(
        item["role"] != "assistant" or item["state"] == "answered"
        for item in context["history"]
    )
    assert context["historyIncludedCount"] == 3
    assert context["historyTruncated"] is True
    assert context["historyOmittedCount"] == 3
    assert context["limits"] == {"maxMessages": 3, "maxTextChars": 100000}


def test_c5_limit_by_text_chars_does_not_cut_messages(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_history(db_path)

    context = fetch(
        db_path, conversation_id, message_id, max_messages=100, max_text_chars=15
    ).context

    texts = [item["text"] for item in context["history"]]
    assert texts[-1] == "текущий"
    assert "текущий" in texts
    # Отдельные сообщения не урезаны посередине.
    assert all(text in {"u1", "a1", "u2", "a2", "u3", "текущий"} for text in texts)
    assert context["historyTruncated"] is True
    assert context["historyOmittedCount"] > 0


def test_c5_current_question_too_long_is_explicit_error(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id = create_conversation(db_path)
    message_id = add_message(
        db_path,
        conversation_id,
        role="user",
        text="очень длинный текущий вопрос",
        state="processing",
    )
    add_snapshot(db_path, message_id, valid_snapshot(), now_iso())

    # Текущий вопрос не помещается — явная ошибка, а не тихое усечение.
    with pytest.raises(McpError):
        fetch(db_path, conversation_id, message_id, max_text_chars=5)


# --- C6: тестовый обработчик получает MCP-контекст ---------------------------


def test_c6_test_handler_receives_mcp_context(db_path, monkeypatch):
    from server import responder
    received = []
    original = responder.prepare

    def handler(context, retrieval, settings):
        received.append((context, retrieval))
        return original(context, retrieval, settings)

    monkeypatch.setattr(responder, "prepare", handler)
    with make_client(db_path) as client:
        conversation_id = new_conversation(client)
        response = send(client, conversation_id, key="c6")
        assert response.status_code == 200, response.text
        assert response.json()["assistantMessage"]["text"] == MOCK_ANSWER
        context, retrieval = received[0]
        conn = _connect(db_path)
        snapshot_created = conn.execute(
            "SELECT created_at FROM snapshots ORDER BY message_id DESC LIMIT 1"
        ).fetchone()["created_at"]
        conn.close()
        assert context["snapshotReceivedAt"] == snapshot_created
        assert context["historyIncludedCount"] == 1
        assert retrieval["version"] == 1
        assert "sources" not in context


# --- C7: сбои MCP не превращаются в успех ------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "server.tests.mcp_fixtures.error_server",
        "server.tests.mcp_fixtures.bad_contract_server",
        "server.tests.mcp_fixtures.crash_server",
        "server.tests.mcp_fixtures.does_not_exist",
    ],
)
def test_c7_mcp_failures_return_mcp_error(tmp_path, module):
    db_path = tmp_path / "support.db"
    with make_client(
        db_path,
        mcp_server_module=module,
        mcp_initialize_timeout=2.0,
        mcp_list_tools_timeout=2.0,
        mcp_call_tool_timeout=2.0,
        mcp_total_timeout=6.0,
    ) as client:
        conversation_id = new_conversation(client)
        response = send(client, conversation_id, key="fail")

        assert response.status_code == 502
        assert response.json()["error"]["code"] == "mcp_error"

        database = Database(db_path)
        rows = database.messages(conversation_id)
        assert [row["role"] for row in rows] == ["user"]
        assert rows[0]["state"] == "error"
        assert rows[0]["error_code"] == "mcp_error"
        assert database.reply_for(rows[0]["id"]) is None


def test_c7_timeout_returns_mcp_error(tmp_path):
    db_path = tmp_path / "support.db"
    with make_client(
        db_path,
        mcp_server_module=f"{_FIXTURES}.hanging_server",
        mcp_initialize_timeout=0.5,
        mcp_list_tools_timeout=0.5,
        mcp_call_tool_timeout=0.5,
        mcp_total_timeout=2.0,
    ) as client:
        conversation_id = new_conversation(client)
        response = send(client, conversation_id, key="timeout")

        assert response.status_code == 502
        assert response.json()["error"]["code"] == "mcp_error"

        database = Database(db_path)
        rows = database.messages(conversation_id)
        assert rows[0]["state"] == "error"
        assert database.reply_for(rows[0]["id"]) is None


# --- C8: явный повтор после ошибки -------------------------------------------


def test_c8_retry_same_key_reuses_snapshot_without_duplicates(db_path):
    calls: list[tuple[int, int]] = []
    state = {"fail": True}
    contexts: list[dict] = []

    async def fake_fetch(db_path_arg, conversation_id, message_id):
        calls.append((conversation_id, message_id))
        if state["fail"]:
            raise McpError("смоделированный сбой MCP")
        result = await fetch_ticket_context_result(
            db_path=db_path_arg,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        contexts.append(result.context)
        return result.context

    app = create_app(db_path=db_path, mcp_fetch=fake_fetch)
    with TestClient(app) as client:
        conversation_id = new_conversation(client)
        payload = {
            "text": "вопрос",
            "snapshot": valid_snapshot(),
            "idempotencyKey": "retry-key",
        }
        url = f"/api/support/conversations/{conversation_id}/messages"

        first = client.post(url, headers=CSRF, json=payload)
        assert first.status_code == 502
        assert first.json()["error"]["code"] == "mcp_error"

        conn = _connect(db_path)
        user_id = conn.execute("SELECT id FROM messages WHERE role = 'user'").fetchone()["id"]
        snapshot_created = conn.execute(
            "SELECT created_at FROM snapshots WHERE message_id = ?", (user_id,)
        ).fetchone()["created_at"]
        conn.close()

        state["fail"] = False
        repeat = client.post(url, headers=CSRF, json=payload)
        assert repeat.status_code == 200, repeat.text
        body = repeat.json()
        assert body["userMessage"]["id"] == user_id
        assert body["userMessage"]["state"] == "answered"
        assert body["assistantMessage"]["text"].startswith(MOCK_ANSWER)

        conn = _connect(db_path)
        rows = conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
        snapshots = conn.execute("SELECT * FROM snapshots").fetchall()
        conn.close()

        # Дублей нет: по-прежнему один вопрос и один ответ, снимок один.
        assert [row["role"] for row in rows] == ["user", "assistant"]
        assert len(snapshots) == 1
        assert snapshots[0]["message_id"] == user_id
        assert snapshots[0]["created_at"] == snapshot_created
        assert json.loads(snapshots[0]["payload"]) == payload["snapshot"]
        context = contexts[0]
        assert context["messageId"] == user_id
        assert context["snapshot"] == payload["snapshot"]
        assert context["snapshotReceivedAt"] == snapshot_created
        assert [(item["id"], item["role"], item["text"]) for item in context["history"]] == [
            (user_id, "user", payload["text"]),
        ]
        attempts = Database(db_path).attempts_for(user_id)
        assert [attempt["state"] for attempt in attempts] == ["failed", "succeeded"]
        binding = json.loads(attempts[1]["context_binding"])
        assert binding["messageId"] == user_id
        assert binding["historyIds"] == [user_id]
        assert binding["snapshotReceivedAt"] == snapshot_created
        # Граница MCP вызывается дважды; после сбоя работает настоящий stdio MCP.
        assert calls == [
            (conversation_id, user_id),
            (conversation_id, user_id),
        ]


# --- C9: жизненный цикл и чистый stdout --------------------------------------


def test_c9_process_terminates_after_success(tmp_path):
    if not Path("/proc").exists():  # pragma: no cover
        pytest.skip("нет /proc")
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_current(db_path)

    fetch(db_path, conversation_id, message_id)

    assert _child_pids("server.mcp_server") == []


def test_c9_process_terminated_after_timeout(tmp_path):
    if not Path("/proc").exists():  # pragma: no cover
        pytest.skip("нет /proc")
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_current(db_path)
    pidfile = tmp_path / "mcp.pid"

    with pytest.raises(McpError):
        fetch(
            db_path,
            conversation_id,
            message_id,
            server_module=f"{_FIXTURES}.hanging_server",
            extra_env={"MCP_TEST_PIDFILE": str(pidfile)},
            initialize_timeout=1.0,
            list_tools_timeout=1.0,
            call_tool_timeout=1.0,
            total_timeout=2.0,
        )

    assert pidfile.exists()
    _wait_gone(int(pidfile.read_text(encoding="utf-8")))
    assert _child_pids("hanging_server") == []


def test_c9_stray_stdout_breaks_protocol(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_current(db_path)

    with pytest.raises(McpError):
        fetch(
            db_path,
            conversation_id,
            message_id,
            server_module=f"{_FIXTURES}.garbage_server",
            initialize_timeout=2.0,
            list_tools_timeout=1.0,
            call_tool_timeout=1.0,
            total_timeout=4.0,
        )


def test_c9_stderr_noise_is_allowed(tmp_path):
    db_path = tmp_path / "support.db"
    conversation_id, message_id = _seed_current(db_path)

    result = fetch(
        db_path,
        conversation_id,
        message_id,
        server_module=f"{_FIXTURES}.noisy_stderr_server",
    )

    assert result.context["contextVersion"] == 1
