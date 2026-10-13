"""Explicit README/docs corpus, atomic local index, and semantic retrieval."""
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile

from shared import indexer

DAY = Path(__file__).resolve().parent
DEFAULT_PROJECT = DAY.parent.parent / "guitar-trainer"
STRATEGY = "section_windows"
FORMAT_VERSION = "day31-v1"


def documents(project):
    root = Path(project).resolve()
    docs = root / "docs"
    if not docs.is_dir() or docs.is_symlink():
        raise ValueError("Нужна обычная папка docs с Markdown-документацией.")
    paths = [root / "README.md", *sorted(docs.rglob("*.md"))]
    if len(paths) < 2:
        raise ValueError("Нужны README.md и хотя бы один Markdown-файл в docs.")
    result = []
    for path in paths:
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if (not path.resolve().is_relative_to(root)
                or any(parent.is_symlink() for parent in [path, *path.parents]
                       if parent.is_relative_to(root))):
            raise ValueError("Документация не должна содержать символические ссылки.")
        if not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError("Документ отсутствует или больше 1 МБ.")
        result.append((relative.as_posix(), path.read_text(encoding="utf-8")))
    if not any(source.startswith("docs/") for source, _ in result):
        raise ValueError("В docs нет доступных Markdown-документов.")
    return result


def corpus_hash(docs):
    digest = hashlib.sha256()
    for source, body in docs:
        for value in (source, body):
            data = value.encode("utf-8")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def index_path(project):
    identity = hashlib.sha256(str(Path(project).resolve()).encode()).hexdigest()[:16]
    return DAY / "resources" / "indexes" / f"{identity}.sqlite3"


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(indexer.MODEL_ID, revision=indexer.MODEL_REVISION,
                               device="cpu")


def build(project, path, model):
    root = Path(project).resolve()
    docs = documents(root)
    chunks = [chunk for source, body in docs
              for chunk in indexer.make_chunks(source, body, model.tokenizer, STRATEGY)]
    if not chunks:
        raise ValueError("Документация пуста.")
    vectors = model.encode([chunk.text for chunk in chunks], normalize_embeddings=True,
                           batch_size=32, show_progress_bar=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".building-", suffix=".sqlite3", dir=path.parent)
    os.close(fd)
    try:
        with closing(sqlite3.connect(temporary)) as conn:
            indexer.create_schema(conn)
            metadata = {"model_id": indexer.MODEL_ID,
                        "model_revision": indexer.MODEL_REVISION,
                        "project": str(root), "corpus_hash": corpus_hash(docs),
                        "format": FORMAT_VERSION}
            conn.executemany("INSERT INTO meta VALUES (?, ?)", metadata.items())
            for chunk, vector in zip(chunks, vectors, strict=True):
                conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (chunk.chunk_id, chunk.strategy, chunk.source, chunk.title,
                              chunk.section, chunk.text, chunk.token_count,
                              indexer.pack_vector(vector)))
            conn.commit()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"documents": len(docs), "chunks": len(chunks), "corpus_hash": corpus_hash(docs)}


def open_fresh(project, path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Индекс отсутствует. Выполните main.py --index.")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        metadata = dict(conn.execute("SELECT key, value FROM meta"))
        indexer.check_model(conn)
        if metadata.get("project") != str(Path(project).resolve()):
            raise ValueError("Индекс принадлежит другому проекту.")
        if (metadata.get("format") != FORMAT_VERSION
                or metadata.get("corpus_hash") != corpus_hash(documents(project))):
            raise ValueError("Документация изменилась. Выполните main.py --index.")
        return conn
    except sqlite3.Error:
        conn.close()
        raise ValueError("Индекс повреждён или имеет неверный формат. Выполните main.py --index.") from None
    except Exception:
        conn.close()
        raise


def retrieve(project, path, question, model, top_k=8):
    with closing(open_fresh(project, path)) as conn:
        candidates = indexer.search(conn, question, model, STRATEGY, top_k * 2)
    # Embeddings stay within 112 tokens. Generation can see a complete short
    # section so overlapping windows do not cut component names or citations.
    bodies = dict(documents(project))
    results = []
    seen = set()
    for candidate in candidates:
        body = bodies[candidate["source"]]
        row = dict(candidate)
        row["context_expanded"] = False
        for start, end, label in indexer.sections(body):
            section = body[start:end].strip()
            if label == row["section"] and len(section) <= 1800:
                row["retrieval_text"] = row["text"]
                row["text"] = section
                row["context_expanded"] = True
                break
        identity = (row["source"], row["section"], row["text"])
        if identity not in seen:
            seen.add(identity)
            results.append(row)
        if len(results) == top_k:
            break
    return results
