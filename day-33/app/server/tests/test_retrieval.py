"""Retrieval contracts and failure boundaries; fake encoder only where explicit.

The real CPU model/tokenizer tests and mandatory evaluate complement these
controlled race, corruption and integration cases. No tests download weights.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import struct
import threading

import pytest

from server import retrieval, retrieval_index as r
from server.app import create_app
from server.tests.conftest import MOCK_ANSWER
from server.db import Database
from server.errors import SupportError
from server.tests.conftest import CSRF, make_client
from server.tests.test_support import new_conversation, send, valid_snapshot


class Tokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        return {"input_ids": list(range(len(text) + 2))}


class Encoder:
    tokenizer = Tokenizer()

    def encode(self, texts, **kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["normalize_embeddings"] is True
        return [[1.0] + [0.0] * (r.DIMENSION - 1) for _ in texts]


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    corpus = tmp_path / "docs/support"
    corpus.mkdir(parents=True)
    monkeypatch.setattr(r, "PROJECT_ROOT", tmp_path)
    (corpus / "faq.md").write_text("# Руководство\n\n## Микрофон\n\nРазрешите доступ к микрофону.\n\n## Струны\n\nИграйте по одной струне.\n", encoding="utf-8")
    monkeypatch.setattr(r, "CORPUS", corpus)
    monkeypatch.setattr(r, "INDEX", tmp_path / "retrieval/index.sqlite")
    monkeypatch.setattr(r, "get_model", lambda: Encoder())
    r.build_index()
    return corpus


def context(text="Как включить микрофон?", previous=()):
    return {"messageId": 9, "snapshot": {"secretDiagnostic": "never query"},
            "history": [*previous, {"id": 9, "role": "user", "text": text}]}


def error_code(action, code):
    with pytest.raises(SupportError) as exc:
        action()
    assert exc.value.code == code
    return exc.value


def test_snapshot_coordinates_and_whole_short_sections(indexed):
    documents = r.read_corpus()
    chunks = r.split_document("docs/support/faq.md", documents["docs/support/faq.md"], Encoder())
    assert chunks == r.split_document("docs/support/faq.md", documents["docs/support/faq.md"], Encoder())
    for chunk in chunks:
        text = documents[chunk["article"]]["text"]
        assert chunk["text"] == text[chunk["start"]:chunk["end"]]
        assert chunk["startLine"] == text[:chunk["start"]].count("\n") + 1
        assert r.token_count(Encoder(), chunk["embeddingText"]) <= 512
        assert chunk["embeddingText"].startswith("passage: ")
    assert all(a["end"] <= b["start"] for a, b in zip(chunks, chunks[1:]))
    result = r.search(context())
    assert result["version"] == 1 and result["queryText"] == "Как включить микрофон?"
    assert 1 <= len(result["sources"]) <= 3
    assert "embeddingText" not in result["sources"][0]


def test_long_unicode_and_semantic_blocks():
    text = "# Статья\n\n## FAQ?\n\n" + "ответ 🙂 ё " * 180 + "\n\n- действие\n- ограничение\n\n```md\n## Не заголовок\n```\n"
    doc = {"text": text, "hash": "version"}
    chunks = r.split_document("docs/support/a.md", doc, Encoder())
    assert any("## Не заголовок" in c["text"] for c in chunks)
    assert all("Не заголовок" not in c["headings"] for c in chunks)
    covered = set()
    for chunk in chunks:
        assert chunk["text"] == text[chunk["start"]:chunk["end"]]
        assert r.token_count(Encoder(), chunk["embeddingText"]) <= 512
        covered.update(range(chunk["start"], chunk["end"]))
    assert covered == set(range(len(text)))
    for a, b in zip(chunks, chunks[1:]):
        overlap = a["end"] - b["start"]
        if overlap > 0:
            assert r.token_count(Encoder(), text[b["start"]:a["end"]]) <= 32
    blocks = r.blocks("- a\n\n- b\n\nnext\n", 0, len("- a\n\n- b\n\nnext\n"))
    assert blocks[0] == (0, len("- a\n\n- b\n\n"))


def test_parent_conditions_survive_long_problem():
    text = "# Статья\n\n## Проблема\n\n**Симптом.** Нет звука.\n\n**Что можно сделать:**\n\n" + "Действие и ограничение. " * 80
    chunks = r.split_document("docs/support/a.md", {"text": text, "hash": "v"}, Encoder())
    later = [c for c in chunks if c["start"] > text.index("**Что можно сделать:**")]
    assert later
    assert all("**Симптом.** Нет звука." in c["embeddingText"] for c in later)


@pytest.mark.parametrize("operation", ["change", "add", "delete"])
def test_corpus_changes_before_start_and_after_start(indexed, operation):
    r.validate_startup()
    if operation == "change":
        (indexed / "faq.md").write_text("# Новая статья\nТекст")
    elif operation == "add":
        (indexed / "new.md").write_text("# Ещё\nТекст")
    else:
        (indexed / "faq.md").unlink()
    code = "corpus_missing" if operation == "delete" else "index_stale"
    error_code(r.validate_startup, code)
    error_code(lambda: r.search(context()), code)


def test_corpus_boundaries(indexed, tmp_path):
    (indexed / "ignored.py").write_text("secret")
    (indexed / "nested").mkdir()
    (indexed / "nested/hidden.md").write_text("secret")
    assert set(r.read_corpus()) == {"docs/support/faq.md"}
    (indexed / "alias.md").symlink_to(tmp_path / "outside.md")
    error_code(r.read_corpus, "corpus_invalid")
    (indexed / "alias.md").unlink()
    (indexed / "directory.md").mkdir()
    error_code(r.read_corpus, "corpus_invalid")


def test_symlink_directory(indexed, tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    alias.symlink_to(indexed, target_is_directory=True)
    monkeypatch.setattr(r, "CORPUS", alias)
    error_code(r.read_corpus, "corpus_invalid")


@pytest.mark.parametrize("component", ["docs", "support"])
def test_symlink_each_corpus_parent(indexed, component):
    directory = indexed.parent if component == "docs" else indexed
    moved = directory.with_name(directory.name + "-original")
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    error_code(r.read_corpus, "corpus_invalid")


@pytest.mark.parametrize("moment", ["before_support", "after_listing"])
def test_parent_replacement_never_reads_external_article(indexed, tmp_path, monkeypatch, moment):
    outside = tmp_path / "outside"
    (outside / "support").mkdir(parents=True)
    (outside / "support/secret.md").write_text("EXTERNAL SECRET", encoding="utf-8")
    real_open, real_listdir = r.os.open, r.os.listdir
    opened_articles = []
    def replace_docs():
        indexed.parent.rename(tmp_path / "old-docs")
        (tmp_path / "docs").symlink_to(outside, target_is_directory=True)
    def guarded_open(path, flags, *args, **kwargs):
        if moment == "before_support" and path == "support":
            replace_docs()
        if str(path).endswith(".md"):
            opened_articles.append(path)
        return real_open(path, flags, *args, **kwargs)
    def listing(fd):
        result = real_listdir(fd)
        if moment == "after_listing":
            replace_docs()
        return result
    monkeypatch.setattr(r.os, "open", guarded_open)
    monkeypatch.setattr(r.os, "listdir", listing)
    error_code(r.read_corpus, "corpus_invalid")
    assert "secret.md" not in opened_articles


def test_missing_corrupt_and_incompatible_index(indexed):
    r.INDEX.unlink()
    error_code(lambda: r.search(context()), "index_missing")
    r.INDEX.write_bytes(b"not sqlite")
    error_code(lambda: r.search(context()), "index_corrupt")
    r.build_index()
    with sqlite3.connect(r.INDEX) as conn:
        manifest = json.loads(conn.execute("SELECT payload FROM manifest").fetchone()[0])
        manifest["parameters"]["dimension"] = 2
        conn.execute("UPDATE manifest SET payload=?", (r.canonical(manifest),))
    error_code(lambda: r.search(context()), "index_incompatible")


@pytest.mark.parametrize("vector,code", [(b"bad", "index_incompatible"),
    (struct.pack("<384f", float("nan"), *([0.] * 383)), "index_corrupt"),
    (struct.pack("<384f", *([0.] * 384)), "index_corrupt")])
def test_vector_corruption_rejected_even_with_updated_fingerprint(indexed, vector, code):
    with sqlite3.connect(r.INDEX) as conn:
        conn.execute("UPDATE chunks SET vector=? WHERE id=(SELECT min(id) FROM chunks)", (vector,))
        manifest = json.loads(conn.execute("SELECT payload FROM manifest").fetchone()[0])
        manifest.pop("fingerprint")
        rows = [(json.loads(c), v) for c, v in conn.execute("SELECT payload,vector FROM chunks ORDER BY id")]
        manifest["fingerprint"] = r.fingerprint(manifest, rows)
        conn.execute("UPDATE manifest SET payload=?", (r.canonical(manifest),))
    error_code(lambda: r.search(context()), code)


def test_failed_publication_preserves_old_index(indexed, monkeypatch):
    previous = r.INDEX.read_bytes()
    def interrupted(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(r.os, "replace", interrupted)
    with pytest.raises(OSError):
        r.build_index()
    assert r.INDEX.read_bytes() == previous
    assert list(r.INDEX.parent.glob(".index-*")) == []


def test_corpus_race_during_encode_has_no_result(indexed, monkeypatch):
    old_encode = r.encode
    def racing(model, texts):
        output = old_encode(model, texts)
        (indexed / "new.md").write_text("# Changed\nNew")
        return output
    monkeypatch.setattr(r, "encode", racing)
    error_code(lambda: r.search(context()), "index_stale")


def test_index_replaced_during_search_has_no_result(indexed, monkeypatch):
    # A concurrent publisher swaps a fully built index. The result cannot refer
    # to that version when the query started against a different fingerprint.
    original_load = r.load_index
    calls = 0
    def replacement(documents, model):
        nonlocal calls
        stamp, rows = original_load(documents, model)
        calls += 1
        return ("different" if calls == 2 else stamp), rows
    monkeypatch.setattr(r, "load_index", replacement)
    error_code(lambda: r.search(context()), "index_stale")


@pytest.mark.parametrize("question", ["А теперь?", "Это уже пробовал", "НЕ ПОМОГЛО!", "а дальше"])
def test_short_followups_use_only_two_users_chronologically(question):
    history = [{"id": 1, "role": "user", "text": "oldest"},
               {"id": 2, "role": "user", "text": "older"},
               {"id": 3, "role": "assistant", "text": "never assistant"},
               {"id": 4, "role": "user", "text": "nearest"}]
    assert r.query_from_context(context(question, history), Encoder()) == "older\n\nnearest\n\n" + question


@pytest.mark.parametrize("question", ["Микрофон?", "Как настроить?", "а теперь расскажи про строй", "Это уже пробовал ещё вчера"])
def test_independent_questions_are_unchanged(question):
    assert r.query_from_context(context(question, [{"id": 1, "role": "user", "text": "unrelated"}]), Encoder()) == question


def test_query_budget_nearest_priority_and_no_truncation():
    history = [{"id": 1, "role": "user", "text": "x" * 490}, {"id": 2, "role": "user", "text": "close"}]
    assert r.token_count(Encoder(), "query: " + "\n\n".join([history[0]["text"], "close", "А теперь?"])) == 517
    assert r.token_count(Encoder(), "query: close\n\nА теперь?") <= r.LIMIT
    assert r.query_from_context(context("А теперь?", history), Encoder()) == "close\n\nА теперь?"
    history[1]["text"] = "x" * 510
    assert r.query_from_context(context("А теперь?", history), Encoder()) == "А теперь?"
    assert r.query_from_context(context("x" * 503), Encoder()) == "x" * 503  # 7 prefix + 2 special tokens
    error_code(lambda: r.query_from_context(context("x" * 504), Encoder()), "query_too_long")
    error_code(lambda: r.query_from_context({"messageId": 99, "history": []}, Encoder()), "mcp_error")


def test_candidates_remove_duplicate_text_and_half_overlap():
    blob = struct.pack("<384f", 1., *([0.] * 383))
    def chunk(id, start, end, text, article="a"):
        return {"id": id, "start": start, "end": end, "text": text, "article": article}
    rows = [(chunk("1", 0, 100, "one"), blob), (chunk("2", 50, 110, "two"), blob),
            (chunk("3", 0, 20, "one", "b"), blob), (chunk("4", 100, 200, "four"), blob),
            (chunk("5", 200, 300, "five"), blob), (chunk("6", 300, 400, "six"), blob)]
    assert [c["id"] for c in r.candidates(rows, r.vector_values(blob))] == ["1", "4", "5"]


def test_real_tokenizer_limits_and_cpu_singleton(tmp_path):
    model = retrieval.get_model()
    assert retrieval.get_model() is model
    assert str(model.device) == "cpu"
    text = "# Статья\n\n## Вопрос?\n\n" + "Длинный русский ответ 🙂 с ограничениями. " * 200
    chunks = r.split_document("docs/support/a.md", {"text": text, "hash": "v"}, model)
    assert len(chunks) > 1
    for chunk in chunks:
        assert r.token_count(model, chunk["embeddingText"]) <= 512
        assert chunk["text"] == text[chunk["start"]:chunk["end"]]
    question = "слово " * 600
    error_code(lambda: r.query_from_context(context(question), model), "query_too_long")


def test_singleton_initialization_is_locked(monkeypatch):
    calls = []
    model = object()
    monkeypatch.setattr(retrieval, "_MODEL", None)
    def load(path):
        calls.append(path)
        return model
    monkeypatch.setattr(retrieval, "_load_local_model", load)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: retrieval.get_model(), range(12)))
    assert all(result is model for result in results)
    assert len(calls) == 1


def test_real_mcp_then_retrieval_persistence_and_owner_isolation(indexed, db_path, monkeypatch):
    from server import responder
    observed = []
    original = responder.prepare
    def handler(mcp, found, settings):
        observed.append((mcp, found))
        return original(mcp, found, settings)
    monkeypatch.setattr(responder, "prepare", handler)
    with make_client(db_path) as client:
        conversation = new_conversation(client)
        response = send(client, conversation, text="Как включить микрофон?")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["assistantMessage"]["text"] == MOCK_ANSWER
        mcp, found = observed[0]
        assert mcp["messageId"] == body["userMessage"]["id"]
        assert found["queryText"] == "Как включить микрофон?"
        saved = body["assistantMessage"]["sources"]
        assert saved and all(set(s) == {"id", "title", "headings", "text", "url"} for s in saved)
        original_snapshot = Database(db_path).snapshot_for(body["userMessage"]["id"])["payload"]
        (indexed / "faq.md").write_text("# Новая статья\n\nДругие сведения\n")
        r.build_index()
        history = client.get(f"/api/support/conversations/{conversation}/messages").json()
        assert history["messages"][1]["sources"] == saved
        monkeypatch.setattr(r, "search", lambda _: pytest.fail("Completed result recomputed"))
        repeat = send(client, conversation, text="Как включить микрофон?")
        assert repeat.status_code == 200 and repeat.json()["assistantMessage"]["sources"] == saved
        assert Database(db_path).snapshot_for(body["userMessage"]["id"])["payload"] == original_snapshot
        with make_client(db_path) as other:
            assert other.get(f"/api/support/conversations/{conversation}/messages").status_code == 404
    with make_client(db_path) as restarted:
        restarted.cookies.update(client.cookies)
        assert restarted.get(f"/api/support/conversations/{conversation}/messages").json()["messages"][1]["sources"] == saved


@pytest.mark.parametrize("code", ["corpus_missing", "model_missing", "model_incompatible", "index_missing", "index_stale", "index_corrupt", "index_incompatible", "query_too_long", "retrieval_failed"])
def test_errors_retry_keeps_snapshot_then_records_new_index(indexed, db_path, monkeypatch, code):
    original = r.search
    calls = []
    def fail(context):
        calls.append(context["messageId"])
        raise SupportError(code, "explicit error", 503)
    monkeypatch.setattr(r, "search", fail)
    with make_client(db_path) as client:
        conversation = new_conversation(client)
        response = send(client, conversation)
        assert response.status_code == 503 and response.json()["error"]["code"] == code
        database = Database(db_path)
        user = database.messages(conversation)[0]
        snapshot = dict(database.snapshot_for(user["id"]))
        assert user["state"] == "error" and database.reply_for(user["id"]) is None
        assert calls == [user["id"]]  # no hidden retries
        monkeypatch.setattr(r, "search", original)
        result = send(client, conversation)
        assert result.status_code == 200, result.text
        assert result.json()["userMessage"]["id"] == user["id"]
        assert len(database.messages(conversation)) == 2
        assert dict(database.snapshot_for(user["id"])) == snapshot
        stored = database.retrieval_for(result.json()["assistantMessage"]["id"])
        assert stored["indexFingerprint"] == r.load_index(r.read_corpus(), Encoder())[0]


def test_transaction_rolls_back_reply_and_state_when_retrieval_not_serializable(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    conversation = database.create_conversation("owner")["id"]
    user = database.create_user_message(conversation_id=conversation, text="q", idempotency_key="k", content_hash="h", snapshot_payload="{}", max_history=20)
    with pytest.raises(TypeError):
        database.create_assistant_reply(conversation_id=conversation, user_message_id=user["id"], text="reply", retrieval={"invalid": object()})
    assert database.reply_for(user["id"]) is None
    assert database.get_message(user["id"])["state"] == "processing"


def test_worker_does_not_block_event_loop(indexed, db_path, monkeypatch):
    # Exercise the actual async route with a controlled blocking encoder. A
    # callback on that same event loop releases it; direct execution would time
    # out rather than pass.
    import httpx
    from server import app as app_module
    original_thread = app_module.asyncio.to_thread
    entered = threading.Event()
    release = threading.Event()
    original_encode = r.encode
    def delayed(model, texts):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("Event loop blocked")
        return original_encode(model, texts)
    monkeypatch.setattr(r, "encode", delayed)
    async def run():
        app = create_app(db_path=db_path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            conversation = (await client.post("/api/support/conversations", headers=CSRF)).json()["conversationId"]
            task = asyncio.create_task(client.post(f"/api/support/conversations/{conversation}/messages", headers=CSRF,
                json={"text": "Микрофон?", "snapshot": valid_snapshot(), "idempotencyKey": "loop"}))
            assert await original_thread(entered.wait, 5)
            release.set()
            assert (await task).status_code == 200
    asyncio.run(run())
