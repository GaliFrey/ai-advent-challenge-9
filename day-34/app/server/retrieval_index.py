"""Local retrieval v1. No network, arbitrary corpus paths or generative model.

All entry points are synchronous and must run in a worker thread. Tests may
replace constants/model, never through the public HTTP API.
"""
from __future__ import annotations

from contextlib import closing, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import struct
import tempfile
import threading
import time

from .config import PROJECT_ROOT
from .errors import SupportError
from .retrieval import MODEL_ID, MODEL_REVISION, LIMIT, DIMENSION, get_model

CORPUS = PROJECT_ROOT / "docs/support"
INDEX = PROJECT_ROOT / "data/retrieval/index.sqlite"
COMPUTE_LOCK = threading.Lock()
SPLITTER_VERSION = "markdown-v1"
PARAMETERS = {"schema": 1, "splitter": SPLITTER_VERSION, "model": MODEL_ID,
              "revision": MODEL_REVISION, "dimension": DIMENSION, "limit": LIMIT,
              "queryPrefix": "query: ", "passagePrefix": "passage: ",
              "normalized": True, "device": "cpu", "vectorFormat": "float32-le",
              "overlapTokens": 32}
FOLLOWUPS = {tuple(s.split()) for s in (
    "а теперь", "а дальше", "что дальше", "что ещё", "это уже пробовал",
    "это уже пробовала", "уже пробовал", "уже пробовала", "не помогло",
    "всё равно не работает")}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def failure(code, detail):
    return SupportError(code, detail + " Выполните uv run python -m server.retrieval build-index.", 503)


def read_corpus():
    """Read direct regular Markdown files through a no-follow directory FD."""
    documents = {}
    try:
        if CORPUS != PROJECT_ROOT / "docs/support":
            raise ValueError("Unexpected corpus path")
        with ExitStack() as resources:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            def open_directory(name, parent=None):
                fd = os.open(name, flags, dir_fd=parent)
                resources.callback(os.close, fd)
                if not stat.S_ISDIR(os.fstat(fd).st_mode):
                    raise ValueError("Not a directory")
                return fd
            root = open_directory(PROJECT_ROOT)
            docs = open_directory("docs", root)
            directory = open_directory("support", docs)
            def check_directories():
                for parent, name, fd in ((root, "docs", docs), (docs, "support", directory)):
                    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    opened = os.fstat(fd)
                    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                        raise ValueError("Corpus directory replaced")
            check_directories()
            for name in sorted(os.listdir(directory)):
                if not name.endswith(".md"):
                    continue
                # Symlink Markdown is an error, not a silently ignored article.
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError("Not a regular article")
                    raw = stream.read()
                    after = os.fstat(stream.fileno())
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                        raise ValueError("Article changed during read")
                documents[f"docs/support/{name}"] = {
                    "text": raw.decode("utf-8"), "hash": hashlib.sha256(raw).hexdigest()}
            check_directories()
    except FileNotFoundError as exc:
        raise SupportError("corpus_missing", "Нет корпуса docs/support/*.md.", 503) from exc
    except (OSError, ValueError, UnicodeError) as exc:
        raise SupportError("corpus_invalid", "Корпус недоступен, изменён во время чтения или содержит недопустимые файлы.", 503) from exc
    if not documents or not any(d["text"].strip() for d in documents.values()):
        raise SupportError("corpus_missing", "Нет непустых статей docs/support/*.md.", 503)
    return documents


def versions(documents):
    return {path: document["hash"] for path, document in documents.items()}


def token_count(model, text):
    return len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def embedding_text(headings, text):
    return "passage: " + " > ".join(headings) + "\n\n" + text


def sections(text):
    """ATX headings outside fenced code; coordinates retain exact newlines."""
    boundaries = []
    stack = []
    offset = 0
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker[1]
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not line[marker.end():].strip():
                fence = None
        if fence is None and not marker:
            heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                level = len(heading[1])
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading[2]))
                boundaries.append((offset, [title for _, title in stack]))
        offset += len(line)
    if not boundaries or boundaries[0][0] != 0:
        boundaries.insert(0, (0, []))
    for i, (start, headings) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        if text[start:end].strip():
            yield start, end, headings


def blocks(text, start, end):
    """Blank-line paragraphs; keep a whole list, table or fenced block intact."""
    result = []
    block_start = start
    offset = start
    fence = None
    list_active = False
    pending_blank = False
    for line in text[start:end].splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        is_list = bool(re.match(r"^\s*(?:[-+*]|\d+[.)])\s", line))
        if pending_blank and line.strip() and fence is None:
            if not (list_active and (is_list or line.startswith(("  ", "\t")))):
                result.append((block_start, offset))
                block_start = offset
                list_active = False
            pending_blank = False
        if marker:
            run = marker[1]
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not line[marker.end():].strip():
                fence = None
        list_active = list_active or is_list
        if not line.strip() and fence is None:
            pending_blank = True
        offset += len(line)
    if block_start < end:
        result.append((block_start, end))
    return result


def split_document(path, document, model):
    text = document["text"]
    chunks = []
    for start, end, headings in sections(text):
        # Retain parent introductory conditions for subsections, in the
        # embedding input only. The displayed source remains contiguous.
        context = []
        for parent_start, parent_end, parent_headings in sections(text[:start]):
            if len(parent_headings) >= 2 and headings[:len(parent_headings)] == parent_headings:
                body = text[parent_start:parent_end]
                if body.strip():
                    context.append(body)
        prefix_headings = headings or [Path(path).stem]
        action = text.find("**Что можно сделать:**", start, end)
        condition_end = action if action >= 0 else start
        section_conditions = text[start:condition_end]
        def passage(a, b):
            conditions = [section_conditions] if a >= condition_end and section_conditions else []
            body = "\n\n".join(context + conditions + [text[a:b]])
            return embedding_text(prefix_headings, body)
        def fits(a, b):
            return token_count(model, passage(a, b)) <= LIMIT
        if token_count(model, passage(start, start)) >= LIMIT:
            raise failure("index_build_failed", "Обязательный контекст раздела превышает лимит модели.")
        spans = []
        if fits(start, end):
            spans = [(start, end)]
        else:
            pending = None
            for a, b in blocks(text, start, end):
                if pending is not None and fits(pending[0], b):
                    pending = (pending[0], b)
                    continue
                if pending is not None:
                    spans.append(pending)
                    pending = None
                if fits(a, b):
                    pending = (a, b)
                    continue
                # Deterministic character slicing; verify every final input.
                # Binary search is merely a candidate finder: no assumption
                # about tokenizer monotonicity is used for acceptance.
                cursor = a
                while cursor < b:
                    lo, hi = cursor + 1, b
                    best = cursor
                    while lo <= hi:
                        middle = (lo + hi) // 2
                        if fits(cursor, middle):
                            best = middle
                            lo = middle + 1
                        else:
                            hi = middle - 1
                    if best == cursor or not fits(cursor, best):
                        raise failure("index_build_failed", "Фрагмент не помещается в токеновый бюджет.")
                    spans.append((cursor, best))
                    if best == b:
                        break
                    overlap = best
                    while overlap > cursor + 1 and token_count(model, text[overlap - 1:best]) <= 32:
                        overlap -= 1
                    cursor = max(cursor + 1, overlap)
            if pending is not None:
                spans.append(pending)
        for a, b in spans:
            chunk = {"article": path, "title": prefix_headings[0], "headings": headings,
                     "text": text[a:b], "start": a, "end": b,
                     "startLine": text.count("\n", 0, a) + 1,
                     "endLine": text.count("\n", 0, max(a, b - 1)) + 1,
                     "documentVersion": document["hash"], "embeddingText": passage(a, b)}
            chunk["id"] = digest([path, document["hash"], a, b, SPLITTER_VERSION])
            if token_count(model, chunk["embeddingText"]) > LIMIT:
                raise failure("index_build_failed", "Превышен токеновый бюджет фрагмента.")
            chunks.append(chunk)
    return chunks


def encode(model, texts):
    if any(token_count(model, text) > LIMIT for text in texts):
        raise SupportError("query_too_long", "Сократите вопрос: он превышает лимит 512 токенов модели.", 422)
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False, device="cpu")
    return [struct.pack(f"<{DIMENSION}f", *map(float, vector)) for vector in vectors]


def vector_values(blob):
    if not isinstance(blob, bytes) or len(blob) != DIMENSION * 4:
        raise failure("index_incompatible", "Размерность векторов индекса несовместима.")
    values = struct.unpack(f"<{DIMENSION}f", blob)
    if not all(math.isfinite(v) for v in values) or abs(sum(v * v for v in values) - 1) > 1e-4:
        raise failure("index_corrupt", "Векторы индекса повреждены.")
    return values


def fingerprint(manifest, rows):
    return digest({"manifest": manifest, "chunks": [[c, v.hex()] for c, v in rows]})


def load_index(documents, model):
    if not INDEX.exists():
        raise failure("index_missing", "Локальный индекс отсутствует.")
    try:
        if INDEX.is_symlink() or not INDEX.is_file():
            raise ValueError("Not a regular index")
        with closing(sqlite3.connect(INDEX.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("BEGIN")
            if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite integrity")
            metadata = conn.execute("SELECT payload FROM manifest").fetchall()
            if len(metadata) != 1:
                raise ValueError("Manifest cardinality")
            manifest = json.loads(metadata[0][0])
            if manifest.get("parameters") != PARAMETERS:
                raise failure("index_incompatible", "Версия или параметры индекса несовместимы.")
            if manifest.get("documents") != versions(documents):
                raise failure("index_stale", "Статьи изменились; индекс устарел.")
            rows = [(json.loads(c), v) for c, v in conn.execute("SELECT payload, vector FROM chunks ORDER BY id")]
        actual_fingerprint = manifest.pop("fingerprint")
        if not rows or fingerprint(manifest, rows) != actual_fingerprint:
            raise ValueError("Fingerprint")
        # Reconstruct source metadata AND exact embedding input rather than
        # trusting a re-hashed, edited manifest.
        expected = {c["id"]: c for p, d in documents.items() for c in split_document(p, d, model)}
        if len(rows) != len(expected):
            raise ValueError("Chunk cardinality")
        for chunk, vector in rows:
            if expected.get(chunk["id"]) != chunk:
                raise ValueError("Source/coordinates/embedding mismatch")
            vector_values(vector)
        return actual_fingerprint, rows
    except SupportError:
        raise
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        raise failure("index_corrupt", "Локальный индекс повреждён.") from exc


def assert_current(documents, index_fingerprint, model):
    current = read_corpus()
    if versions(current) != versions(documents):
        raise failure("index_stale", "Корпус изменился во время поиска.")
    if load_index(current, model)[0] != index_fingerprint:
        raise failure("index_stale", "Индекс изменился во время поиска.")
    if versions(read_corpus()) != versions(documents):
        raise failure("index_stale", "Корпус изменился во время проверки индекса.")


def build_index():
    started = time.perf_counter()
    with COMPUTE_LOCK:
        documents = read_corpus()
        model = get_model()
        chunks = [c for p, d in documents.items() for c in split_document(p, d, model)]
        if not chunks:
            raise failure("index_build_failed", "Нет фрагментов для построения индекса.")
        vectors = encode(model, [c["embeddingText"] for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Incomplete encoder output")
        rows = sorted(zip(chunks, vectors), key=lambda row: row[0]["id"])
        for _, vector in rows:
            vector_values(vector)
        manifest = {"parameters": PARAMETERS, "documents": versions(documents)}
        stamp = fingerprint(manifest, rows)
        manifest["fingerprint"] = stamp
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".index-", suffix=".sqlite", dir=INDEX.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            with closing(sqlite3.connect(temporary)) as conn:
                conn.executescript("CREATE TABLE manifest(payload TEXT NOT NULL); CREATE TABLE chunks(id TEXT PRIMARY KEY, payload TEXT NOT NULL, vector BLOB NOT NULL);")
                conn.execute("INSERT INTO manifest VALUES (?)", (canonical(manifest),))
                conn.executemany("INSERT INTO chunks VALUES (?, ?, ?)",
                                 [(c["id"], canonical(c), v) for c, v in rows])
                conn.commit()
                if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ValueError("SQLite integrity")
                stored_rows = [(json.loads(c), v) for c, v in conn.execute("SELECT payload, vector FROM chunks ORDER BY id")]
                base = {k: v for k, v in manifest.items() if k != "fingerprint"}
                if fingerprint(base, stored_rows) != stamp:
                    raise ValueError("Written fingerprint")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            if versions(read_corpus()) != versions(documents):
                raise failure("index_stale", "Корпус изменился при построении индекса.")
            os.replace(temporary, INDEX)
            directory = os.open(INDEX.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return {"articles": len(documents), "chunks": len(chunks), "indexFingerprint": stamp,
                "seconds": time.perf_counter() - started, "model": MODEL_ID, "revision": MODEL_REVISION}


def query_from_context(context, model):
    try:
        history = context["history"]
        current_id = context["messageId"]
        positions = [i for i, item in enumerate(history) if item["id"] == current_id and item["role"] == "user"]
        if len(positions) != 1:
            raise ValueError("Current message missing/duplicated")
        position = positions[0]
        current = history[position]["text"]
        if not isinstance(current, str) or not current.strip():
            raise ValueError("Invalid current question")
        if token_count(model, "query: " + current) > LIMIT:
            raise SupportError("query_too_long", "Сократите вопрос: он превышает лимит 512 токенов модели.", 422)
        words = tuple(re.findall(r"[^\W_]+", current.casefold(), re.UNICODE))
        selected = []
        if len(current) <= 80 and len(words) <= 12 and words in FOLLOWUPS:
            previous = [item["text"] for item in history[:position] if item["role"] == "user" and item["id"] != current_id][-2:]
            for text in reversed(previous):
                candidate = "\n\n".join([text] + selected + [current])
                if token_count(model, "query: " + candidate) > LIMIT:
                    break
                selected.insert(0, text)
        return "\n\n".join(selected + [current])
    except SupportError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SupportError("mcp_error", "MCP не передал пригодную историю текущего вопроса.", 502) from exc


def candidates(rows, query_vector):
    scored = sorted(((sum(a * b for a, b in zip(vector_values(v), query_vector)), c) for c, v in rows),
                    key=lambda pair: (-pair[0], pair[1]["id"]))
    selected = []
    for score, chunk in scored:
        duplicate = False
        for existing in selected:
            intersection = min(existing["end"], chunk["end"]) - max(existing["start"], chunk["start"])
            shorter = min(existing["end"] - existing["start"], chunk["end"] - chunk["start"])
            if (existing["id"] == chunk["id"] or existing["text"] == chunk["text"] or
                (existing["article"] == chunk["article"] and intersection >= shorter / 2)):
                duplicate = True
                break
        if not duplicate:
            selected.append({k: v for k, v in chunk.items() if k != "embeddingText"} | {"similarity": score})
        if len(selected) == 3:
            break
    return selected


def validate_startup():
    with COMPUTE_LOCK:
        documents = read_corpus()
        model = get_model()
        stamp, _ = load_index(documents, model)
        assert_current(documents, stamp, model)


def search(context):
    try:
        with COMPUTE_LOCK:
            documents = read_corpus()
            model = get_model()
            query = query_from_context(context, model)
            stamp, rows = load_index(documents, model)
            vector = vector_values(encode(model, ["query: " + query])[0])
            sources = candidates(rows, vector)
            assert_current(documents, stamp, model)
            return {"version": 1, "indexFingerprint": stamp, "queryText": query, "sources": sources}
    except SupportError:
        raise
    except Exception as exc:
        raise SupportError("retrieval_failed", "Локальный поиск завершился ошибкой. Повторите явно после устранения причины.", 503) from exc


def evaluate():
    """Read only: never prepare weights, rebuild an index or write reports."""
    fixture = json.loads((PROJECT_ROOT / "server/tests/fixtures/retrieval_questions.json").read_text())
    cold_start = time.perf_counter()
    validate_startup()
    cold_seconds = time.perf_counter() - cold_start
    evaluation_stamp = None
    def run(question):
        nonlocal evaluation_stamp
        start = time.perf_counter()
        result = search({"messageId": 1, "history": [{"id": 1, "role": "user", "text": question}]})
        if evaluation_stamp is not None and evaluation_stamp != result["indexFingerprint"]:
            raise failure("index_stale", "Индекс изменился во время оценки качества.")
        evaluation_stamp = result["indexFingerprint"]
        return result, time.perf_counter() - start
    run(fixture["main"][0]["question"])  # explicit warmup
    result = {"model": MODEL_ID, "revision": MODEL_REVISION, "coldValidationSeconds": cold_seconds,
              "threshold": "18/20", "unansweredInterpretation": "Кандидаты не доказывают наличие ответа; успешность не присваивается."}
    for group in ("main", "heldOut"):
        items = []
        for item in fixture[group]:
            found, seconds = run(item["question"])
            hit = any(s["article"] == "docs/support/" + item["article"] and item["section"] in s["headings"] for s in found["sources"])
            items.append({**item, "hit": hit, "seconds": seconds,
                          "candidates": [{"article": s["article"], "headings": s["headings"]} for s in found["sources"]]})
        result[group] = {"hits": sum(i["hit"] for i in items), "total": len(items), "items": items,
                         "misses": [i["question"] for i in items if not i["hit"]]}
    result["unanswered"] = []
    for question in fixture["unanswered"]:
        found, seconds = run(question)
        result["unanswered"].append({"question": question, "seconds": seconds,
                                      "candidates": [{"article": s["article"], "headings": s["headings"]} for s in found["sources"]]})
    documents = read_corpus()
    with COMPUTE_LOCK:
        stamp, rows = load_index(documents, get_model())
        assert_current(documents, stamp, get_model())
    if evaluation_stamp != stamp:
        raise failure("index_stale", "Индекс изменился перед итогом оценки.")
    result.update(articles=len(documents), chunks=len(rows), indexFingerprint=stamp,
                  passed=result["main"]["total"] == 20 and result["main"]["hits"] >= 18,
                  tuningPerformed=False, ambiguities="Проверяются ожидаемые разделы; несколько кандидатов могут быть тематически близки. Ручная оценка не выполнена.")
    return result
