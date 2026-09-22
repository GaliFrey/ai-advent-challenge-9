"""Read-only access to the fixed Datex documentation index."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "corpus" / "datex-index.sqlite3"
SNAPSHOT = ROOT / "corpus" / "snapshot.json"
TOKENS = re.compile(r"[\w]+", re.UNICODE)
MAX_DOCUMENT_CHARS = 6000


def _connect() -> sqlite3.Connection:
    if not INDEX.is_file():
        raise FileNotFoundError(f"Datex index is missing: {INDEX}")
    connection = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def status() -> dict:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    with closing(_connect()) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
    if metadata.get("snapshot_id") != snapshot.get("snapshot_id"):
        raise ValueError("Datex index and snapshot metadata do not match")
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "document_count": count,
        "known_error_count": snapshot.get("error_count", 0),
        "product_version": snapshot.get("product_version"),
        "note": "A fixed, incomplete public documentation snapshot; no target WebSoft build was tested.",
    }


def search(query: str, limit: int = 3) -> dict:
    query = query.strip()
    if not query or len(query) > 120:
        raise ValueError("query must contain 1–120 characters")
    if not 1 <= limit <= 3:
        raise ValueError("limit must be between 1 and 3")
    terms = TOKENS.findall(query)
    if not terms:
        return {"query": query, "results": [], "coverage_note": "No indexed terms."}
    lowered_terms = [term.casefold() for term in terms]
    array_sort_query = (
        any("массив" in term or "array" in term for term in lowered_terms)
        and any("сорт" in term or "sort" in term for term in lowered_terms)
    )
    expression = " OR ".join('"' + term.replace('"', '""') + '"*' for term in terms[:8])
    with closing(_connect()) as connection:
        rows = connection.execute(
            """SELECT d.document_id, d.title, d.url, d.topic_path,
                      snippet(documents_fts, 4, '[', ']', '…', 24) AS snippet
               FROM documents_fts JOIN documents d ON d.rowid = documents_fts.rowid
               WHERE documents_fts MATCH ?
               ORDER BY (lower(d.title) IN (lower(?), lower(?) || '()')) DESC,
                        (? AND d.topic_path LIKE '%Работа с массивами%'
                           AND lower(d.title) LIKE '%sort%') DESC,
                        (? AND d.topic_path LIKE '%Работа с массивами%') DESC,
                        bm25(documents_fts) ASC
               LIMIT ?""",
            (expression, query, query, array_sort_query, array_sort_query, limit),
        ).fetchall()
    return {
        "query": query,
        "results": [
            {
                "document_id": row["document_id"],
                "title": row["title"],
                "url": row["url"],
                "topic_path": json.loads(row["topic_path"]),
                "snippet": row["snippet"],
            }
            for row in rows
        ],
        "coverage_note": "No match does not prove that the API is absent from WebSoft.",
    }


def read(document_id: str) -> dict:
    if not document_id or len(document_id) > 100:
        raise ValueError("document_id must contain 1–100 characters")
    with closing(_connect()) as connection:
        row = connection.execute(
            """SELECT document_id, title, url, snapshot_id, plain_text
               FROM documents WHERE document_id = ?""",
            (document_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"document not found: {document_id}")
    body = row["plain_text"]
    return {
        "document_id": row["document_id"],
        "title": row["title"],
        "url": row["url"],
        "snapshot_id": row["snapshot_id"],
        "text": body[:MAX_DOCUMENT_CHARS],
        "truncated": len(body) > MAX_DOCUMENT_CHARS,
        "total_characters": len(body),
        "applicability_note": "Documentation is evidence, not verification on a specific WebSoft build.",
    }
