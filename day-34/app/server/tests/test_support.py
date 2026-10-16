"""Серверные проверки C1–C7 и C11 задачи T007.

Каждый тест назван по критерию, который он подтверждает:
C1 один loopback-origin, C2 атомарное хранение, C3 персистентность и
восстановление, C4 новый разговор, C5 изоляция сессий, C6 идемпотентность,
C7 валидация и лимиты, C11 недоступность служебных файлов.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from server.config import DEFAULT_HOST
from server.tests.conftest import CSRF, make_client, MOCK_ANSWER


# --- вспомогательные функции -------------------------------------------------


def idle_snapshot() -> dict:
    return {
        "schemaVersion": 1,
        "collectedAt": "2026-10-14T12:00:00.000Z",
        "capture": {"status": "idle", "lastError": None},
        "audio": {
            "trackState": None,
            "trackEnabled": None,
            "trackMuted": None,
            "contextState": None,
        },
        "device": {"actualLabel": None},
        "measurement": {
            "rms": None,
            "frequencyHz": None,
            "matchedString": None,
            "ageMs": None,
        },
        "display": {"string": None, "cents": None, "direction": None, "tuned": False},
        "commit": {"ageMs": None},
    }


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


def message_rows(db_path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
    finally:
        conn.close()


# --- C1: один loopback-origin ------------------------------------------------


def test_c1_page_static_and_api_share_one_origin(client):
    page = client.get("/")
    assert page.status_code == 200
    assert 'type="module"' in page.text
    assert 'id="start"' in page.text
    assert 'id="support-panel"' in page.text

    for path in ("/style.css", "/src/app.js", "/src/support.js", "/src/diagnostics.js"):
        assert client.get(path).status_code == 200

    session = client.get("/api/support/session")
    assert session.status_code == 200
    body = session.json()
    assert body["testMode"] is False
    assert "sessionId" in body
    assert "currentConversationId" in body


def test_c1_default_host_is_loopback():
    assert DEFAULT_HOST == "127.0.0.1"


# --- C2: атомарное хранение сообщения, снимка и ответа -----------------------


def test_c2_message_snapshot_and_reply_persisted(client, db_path):
    conversation_id = new_conversation(client)
    snapshot = valid_snapshot()
    response = send(client, conversation_id, snapshot=snapshot)

    assert response.status_code == 200, response.text
    body = response.json()
    # Ответ подменённого провайдера проходит валидацию и сохраняется после MCP.
    assert body["assistantMessage"]["text"].startswith(MOCK_ANSWER)
    assert body["userMessage"]["state"] == "answered"

    rows = message_rows(db_path)
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[0]["state"] == "answered"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        stored = conn.execute("SELECT * FROM snapshots").fetchall()
    finally:
        conn.close()
    assert len(stored) == 1
    assert json.loads(stored[0]["payload"]) == snapshot
    assert stored[0]["message_id"] == rows[0]["id"]


def test_c2_invalid_snapshot_leaves_no_partial_write(client, db_path):
    conversation_id = new_conversation(client)
    broken = valid_snapshot()
    broken["measurement"]["frequencyHz"] = 9999

    response = send(client, conversation_id, snapshot=broken, key="broken")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_snapshot"

    assert message_rows(db_path) == []
    assert client.get(
        f"/api/support/conversations/{conversation_id}/messages"
    ).json()["messages"] == []


# --- C3: персистентность и восстановление ------------------------------------


def test_c3_history_survives_restart(client, db_path):
    conversation_id = new_conversation(client)
    snapshot = valid_snapshot()
    assert send(client, conversation_id, snapshot=snapshot).status_code == 200
    cookie = client.cookies.get("gt_session")

    with make_client(db_path) as restarted:
        restarted.cookies.set("gt_session", cookie)
        response = restarted.get(
            f"/api/support/conversations/{conversation_id}/messages"
        )
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[0]["snapshot"] == snapshot
        assert messages[1]["text"].startswith(MOCK_ANSWER)


def test_c3_processing_becomes_error_after_restart(db_path):
    # Создаём схему, затем оставляем сообщение в состоянии processing.
    make_client(db_path)
    conn = sqlite3.connect(str(db_path))
    timestamp = "2026-10-14T12:00:00.000Z"
    cursor = conn.execute(
        "INSERT INTO conversations (session_id, created_at, updated_at) VALUES (?, ?, ?)",
        ("sess-restart", timestamp, timestamp),
    )
    conversation_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO messages (conversation_id, role, text, state, created_at) "
        "VALUES (?, 'user', 'незавершённое', 'processing', ?)",
        (conversation_id, timestamp),
    )
    conn.commit()
    conn.close()

    with make_client(db_path) as restarted:
        restarted.cookies.set("gt_session", "sess-restart")
        response = restarted.get(
            f"/api/support/conversations/{conversation_id}/messages"
        )
        assert response.status_code == 200
        message = response.json()["messages"][0]
        assert message["state"] == "error"
        assert message["errorCode"] == "interrupted"


# --- C4: новый разговор ------------------------------------------------------


def test_c4_new_conversation_keeps_previous_data(client):
    first = new_conversation(client)
    assert send(client, first, text="первый", key="a").status_code == 200

    second = new_conversation(client)
    assert second != first
    assert send(client, second, text="второй", key="b").status_code == 200

    first_messages = client.get(
        f"/api/support/conversations/{first}/messages"
    ).json()["messages"]
    second_messages = client.get(
        f"/api/support/conversations/{second}/messages"
    ).json()["messages"]

    assert [m["text"] for m in first_messages][0] == "первый"
    assert first_messages[1]["text"].startswith(MOCK_ANSWER)
    assert [m["text"] for m in second_messages][0] == "второй"
    assert second_messages[1]["text"].startswith(MOCK_ANSWER)


# --- C5: изоляция сессий ------------------------------------------------------


def test_c5_sessions_are_isolated(db_path):
    with make_client(db_path) as owner, make_client(db_path) as stranger:
        conversation_id = new_conversation(owner)
        assert send(owner, conversation_id, key="owner").status_code == 200

        stranger.get("/api/support/session")
        assert owner.cookies.get("gt_session") != stranger.cookies.get("gt_session")

        assert stranger.get(
            f"/api/support/conversations/{conversation_id}/messages"
        ).status_code == 404
        assert send(stranger, conversation_id, key="stranger").status_code == 404

        assert owner.get(
            f"/api/support/conversations/{conversation_id}/messages"
        ).status_code == 200


# --- C6: идемпотентность ------------------------------------------------------


def test_c6_idempotent_repeat_and_conflict(client, db_path):
    conversation_id = new_conversation(client)
    payload = {
        "text": "повтор",
        "snapshot": valid_snapshot(),
        "idempotencyKey": "same-key",
    }
    url = f"/api/support/conversations/{conversation_id}/messages"

    first = client.post(url, headers=CSRF, json=payload)
    assert first.status_code == 200
    first_user_id = first.json()["userMessage"]["id"]
    assert len(message_rows(db_path)) == 2

    repeat = client.post(url, headers=CSRF, json=payload)
    assert repeat.status_code == 200
    assert repeat.json()["userMessage"]["id"] == first_user_id
    # Дубль не создаётся: строк по-прежнему две (вопрос и ответ).
    assert len(message_rows(db_path)) == 2

    conflict = client.post(url, headers=CSRF, json={**payload, "text": "другое"})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"
    assert len(message_rows(db_path)) == 2


# --- C7: валидация снимка и лимиты -------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda snapshot: snapshot.update({"schemaVersion": 2}),
        lambda snapshot: snapshot.pop("capture"),
        lambda snapshot: snapshot["measurement"].update({"rms": "громко"}),
        lambda snapshot: snapshot["device"].update({"actualLabel": "Микрофон"}),
        lambda snapshot: snapshot["display"].update({"cents": 0.5}),
        lambda snapshot: snapshot["measurement"].update({"ageMs": None}),
    ],
)
def test_c7_invalid_snapshot_variants_rejected(client, db_path, mutate):
    conversation_id = new_conversation(client)
    broken = valid_snapshot()
    mutate(broken)
    response = send(client, conversation_id, snapshot=broken, key="bad")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_snapshot"
    assert message_rows(db_path) == []


def test_c7_non_finite_number_rejected(client, db_path):
    conversation_id = new_conversation(client)
    # NaN/Infinity не являются корректным JSON, но Python json.loads их принимает;
    # сервер обязан отклонить такое значение.
    raw = json.dumps(
        {"text": "нечисло", "idempotencyKey": "nan-key", "snapshot": valid_snapshot()}
    ).replace('"frequencyHz": 109.49', '"frequencyHz": Infinity')
    response = client.post(
        f"/api/support/conversations/{conversation_id}/messages",
        headers={**CSRF, "content-type": "application/json"},
        content=raw.encode("utf-8"),
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_snapshot"
    assert message_rows(db_path) == []


def test_c7_empty_and_long_text_rejected(client, db_path):
    conversation_id = new_conversation(client)
    empty = send(client, conversation_id, text="   ", key="empty")
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "invalid_text"

    long_text = "x" * 5000
    too_long = send(client, conversation_id, text=long_text, key="long")
    assert too_long.status_code == 413
    assert too_long.json()["error"]["code"] == "text_too_long"

    assert message_rows(db_path) == []


def test_c7_oversized_body_rejected(client):
    conversation_id = new_conversation(client)
    body = b"x" * (70 * 1024)
    response = client.post(
        f"/api/support/conversations/{conversation_id}/messages",
        headers={**CSRF, "content-type": "application/json"},
        content=body,
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_too_large"


def test_c7_oversized_snapshot_rejected(client, db_path):
    conversation_id = new_conversation(client)
    snapshot = valid_snapshot()
    snapshot["padding"] = "y" * (20 * 1024)
    response = send(client, conversation_id, snapshot=snapshot, key="big")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "snapshot_too_large"
    assert message_rows(db_path) == []


def test_c7_history_limit_rejected_without_partial_write(db_path):
    with make_client(db_path, max_history_messages=2) as client:
        conversation_id = new_conversation(client)
        assert send(client, conversation_id, key="first").status_code == 200

        overflow = send(client, conversation_id, key="second")
        assert overflow.status_code == 409
        assert overflow.json()["error"]["code"] == "history_limit"

        messages = client.get(
            f"/api/support/conversations/{conversation_id}/messages"
        ).json()["messages"]
        assert len(messages) == 2


# --- C11: служебные файлы недоступны -----------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/server/app.py",
        "/server/",
        "/tasks/T007/task.json",
        "/.env",
        "/data/support.db",
        "/README.md",
        "/web/README.md",
        "/web/test/page.test.mjs",
        "/pyproject.toml",
        "/uv.lock",
        "/docs/architecture.md",
    ],
)
def test_c11_service_files_not_served(client, path):
    assert client.get(path).status_code == 404
