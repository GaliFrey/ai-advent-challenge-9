"""Build and query three local embedding indexes of the course README files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


DAY_DIR = Path(__file__).resolve().parent
ROOT = DAY_DIR.parent
DEFAULT_INDEX = DAY_DIR / "indexes" / "readmes.sqlite3"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MAX_TOKENS = 112  # The model accepts 128 tokens including special tokens.
OVERLAP = 16
STRATEGIES = ("fixed", "structure", "section_windows")
HEADINGS = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class Tokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool,
                 return_offsets_mapping: bool, truncation: bool,
                 verbose: bool) -> dict: ...


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: str
    source: str
    title: str
    section: str
    text: str
    token_count: int


def read_documents() -> list[tuple[str, str]]:
    documents = []
    for day in range(21):
        path = ROOT / f"day-{day:02d}" / "README.md"
        if not path.is_file():
            raise FileNotFoundError(path)
        documents.append((path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8")))
    return documents


def token_offsets(tokenizer: Tokenizer, text: str) -> list[tuple[int, int]]:
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets if end > start]


def token_slices(text: str, tokenizer: Tokenizer, limit: int,
                 overlap: int = 0) -> list[tuple[int, str]]:
    if limit <= 0 or not 0 <= overlap < limit:
        raise ValueError("Expected 0 <= overlap < limit")
    offsets = token_offsets(tokenizer, text)
    if offsets and text[offsets[-1][1]:].strip():
        raise ValueError("Tokenizer did not cover the full input text")
    result: list[tuple[int, str]] = []
    start = 0
    while start < len(offsets):
        end = min(start + limit, len(offsets))
        part = text[offsets[start][0]:offsets[end - 1][1]].strip()
        while len(token_offsets(tokenizer, part)) > limit:
            end -= 1
            if end <= start:
                raise ValueError("A single token exceeds the chunk limit")
            part = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if part:
            result.append((offsets[start][0], part))
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return result


def token_windows(text: str, tokenizer: Tokenizer, limit: int,
                  overlap: int = 0) -> list[str]:
    return [part for _, part in token_slices(text, tokenizer, limit, overlap)]


def sections(text: str) -> list[tuple[int, int, str]]:
    matches = list(HEADINGS.finditer(text))
    if not matches:
        return [(0, len(text), "(без заголовка)")]
    result = []
    if matches[0].start():
        result.append((0, matches[0].start(), "(вступление)"))
    headings: dict[int, str] = {}
    for i, match in enumerate(matches):
        level = len(match.group(1))
        headings[level] = match.group(2).strip()
        for stale in [key for key in headings if key > level]:
            del headings[stale]
        label = " / ".join(headings[key] for key in sorted(headings))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result.append((match.start(), end, label))
    return result


def paragraph_pieces(text: str, tokenizer: Tokenizer, limit: int) -> list[str]:
    pieces = []
    start = 0
    for boundary in PARAGRAPH_BREAK.finditer(text):
        part = text[start:boundary.start()].strip()
        if part:
            pieces.extend(token_windows(part, tokenizer, limit))
        start = boundary.end()
    part = text[start:].strip()
    if part:
        pieces.extend(token_windows(part, tokenizer, limit))
    return pieces


def structural_chunks(text: str, tokenizer: Tokenizer, limit: int) -> list[tuple[str, str]]:
    result = []
    for start, end, section in sections(text):
        pending = ""
        for piece in paragraph_pieces(text[start:end], tokenizer, limit):
            candidate = f"{pending}\n\n{piece}" if pending else piece
            if len(token_offsets(tokenizer, candidate)) <= limit:
                pending = candidate
            else:
                if pending:
                    result.append((section, pending))
                pending = piece
        if pending:
            result.append((section, pending))
    return result


def section_window_chunks(text: str, tokenizer: Tokenizer, limit: int) -> list[tuple[str, str]]:
    result = []
    for start, end, section in sections(text):
        segment = text[start:end]
        heading = HEADINGS.match(segment)
        body = segment[heading.end():].strip() if heading else segment.strip()
        if not body:
            continue
        prefix = f"Раздел: {section}\n\n"
        room = limit - len(token_offsets(tokenizer, prefix)) - 2
        if room < 1:
            raise ValueError(f"Section heading leaves no room for text: {section}")
        while room:
            overlap = min(OVERLAP, room // 4)
            windows = token_windows(body, tokenizer, room, overlap)
            if all(len(token_offsets(tokenizer, prefix + window)) <= limit for window in windows):
                result.extend((section, prefix + window) for window in windows)
                break
            room -= 1
        else:
            raise ValueError(f"Cannot fit a section chunk: {section}")
    return result


def make_chunks(source: str, text: str, tokenizer: Tokenizer,
                strategy: str, limit: int = MAX_TOKENS) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    title_match = HEADINGS.search(text)
    title = title_match.group(2).strip() if title_match else Path(source).stem
    spans = sections(text)
    if strategy == "fixed":
        parts = []
        if limit <= OVERLAP:
            raise ValueError("Chunk limit must exceed overlap")
        for start, part in token_slices(text, tokenizer, limit, OVERLAP):
            section = next((label for left, right, label in spans
                            if left <= start < right), spans[-1][2])
            if part:
                parts.append((section, part))
    elif strategy == "structure":
        parts = structural_chunks(text, tokenizer, limit)
    else:
        parts = section_window_chunks(text, tokenizer, limit)
    chunks = []
    for number, (section, part) in enumerate(parts, 1):
        count = len(token_offsets(tokenizer, part))
        if count > limit:
            raise ValueError(f"Oversized chunk in {source}: {count} tokens")
        chunks.append(Chunk(f"{strategy}:{source}:{number:04d}", strategy,
                            source, title, section, part, count))
    return chunks


def connect_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            section TEXT NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE INDEX chunks_strategy ON chunks(strategy);
    """)


def pack_vector(vector) -> bytes:
    values = [float(value) for value in vector]
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)


def build(path: Path, model) -> dict:
    documents = read_documents()
    tokenizer = model.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer with offset mapping is required")
    if model.max_seq_length < MAX_TOKENS + tokenizer.num_special_tokens_to_add(pair=False):
        raise RuntimeError("The model sequence limit is too short for these chunks")
    chunks = [chunk for source, text in documents for strategy in STRATEGIES
              for chunk in make_chunks(source, text, tokenizer, strategy)]
    if not chunks:
        raise RuntimeError("The corpus produced no chunks")
    digest = hashlib.sha256()
    for source, body in documents:
        digest.update(source.encode("utf-8") + b"\0" + body.encode("utf-8") + b"\0")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".readmes-", suffix=".sqlite3", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        with closing(connect_index(temporary)) as connection, connection:
            create_schema(connection)
            connection.executemany("INSERT INTO meta VALUES (?, ?)", [
                ("model_id", MODEL_ID), ("model_revision", MODEL_REVISION),
                ("corpus_sha256", digest.hexdigest()),
                ("document_count", str(len(documents))), ("max_tokens", str(MAX_TOKENS)),
                ("overlap", str(OVERLAP)),
            ])
            for start in range(0, len(chunks), 32):
                batch = chunks[start:start + 32]
                vectors = model.encode([chunk.text for chunk in batch],
                                       normalize_embeddings=True, show_progress_bar=False)
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding count does not match the chunks")
                rows = []
                for chunk, vector in zip(batch, vectors, strict=True):
                    rows.append((chunk.chunk_id, chunk.strategy, chunk.source,
                                 chunk.title, chunk.section, chunk.text,
                                 chunk.token_count, pack_vector(vector)))
                connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    counts = {strategy: sum(chunk.strategy == strategy for chunk in chunks)
              for strategy in STRATEGIES}
    return {"documents": len(documents), "words": sum(len(text.split()) for _, text in documents),
            "corpus_sha256": digest.hexdigest(), "chunks": counts, "index": str(path)}


def check_model(connection: sqlite3.Connection) -> None:
    values = dict(connection.execute("SELECT key, value FROM meta WHERE key IN ('model_id', 'model_revision')"))
    if values.get("model_id") != MODEL_ID or values.get("model_revision") != MODEL_REVISION:
        raise ValueError("Index embedding model differs from the configured model")


def search(connection: sqlite3.Connection, query: str, model,
           strategy: str, top_k: int = 3) -> list[dict]:
    if strategy not in STRATEGIES or top_k < 1 or not query.strip():
        raise ValueError("Specify a strategy, nonempty query and positive top_k")
    check_model(connection)
    query_vector = [float(value) for value in model.encode(query, normalize_embeddings=True)]
    results = []
    for row in connection.execute(
        "SELECT chunk_id, source, title, section, text, token_count, embedding "
        "FROM chunks WHERE strategy = ?", (strategy,)
    ):
        vector = unpack_vector(row[6])
        if len(vector) != len(query_vector):
            raise ValueError("Embedding dimensions differ")
        score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
        results.append({"chunk_id": row[0], "source": row[1], "title": row[2],
                        "section": row[3], "text": row[4], "token_count": row[5],
                        "score": round(score, 5)})
    results.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    return results[:top_k]


EVALUATION = (
    ("Как ограничивается число генерируемых токенов?", "day-02/README.md"),
    ("Как устроена явная память агента?", "day-11/README.md"),
    ("Какой индекс используется для документации Datex?", "day-17/README.md"),
    ("Как планировщик собирает SSH-входы?", "day-18/README.md"),
    ("Как отчёт скачивается с третьего MCP-сервера?", "day-20/README.md"),
)


def compare(connection: sqlite3.Connection, model) -> dict:
    check_model(connection)
    result = {}
    for strategy in STRATEGIES:
        count, average, longest = connection.execute(
            "SELECT COUNT(*), AVG(token_count), MAX(token_count) FROM chunks WHERE strategy = ?",
            (strategy,),
        ).fetchone()
        queries = []
        reciprocal_ranks = []
        for query, expected in EVALUATION:
            hits = search(connection, query, model, strategy, top_k=5)
            rank = next((i for i, hit in enumerate(hits, 1) if hit["source"] == expected), None)
            reciprocal_ranks.append(1 / rank if rank else 0)
            queries.append({"query": query, "expected": expected, "rank_at_5": rank,
                            "top_source": hits[0]["source"] if hits else None})
        result[strategy] = {"chunks": count, "mean_tokens": round(average or 0, 1),
                            "max_tokens": longest, "hit_at_5": sum(bool(value) for value in reciprocal_ranks),
                            "mrr_at_5": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
                            "queries": queries}
    return result


def demo(path: Path, model, console: Console | None = None) -> None:
    """Rebuild the index and display the assignment results for a terminal demo."""
    console = console or Console()
    summary = build(path, model)
    query = EVALUATION[1][0]
    with closing(connect_index(path)) as connection:
        comparison = compare(connection, model)
        examples = {strategy: search(connection, query, model, strategy, top_k=1)[0]
                    for strategy in STRATEGIES}

    console.print(Panel.fit(
        f"[bold]Корпус:[/] {summary['documents']} README · "
        f"{summary['words']:,}".replace(",", " ") + " слов\n"
        f"[bold]Модель:[/] {MODEL_ID} · {MODEL_REVISION[:12]}\n"
        f"[bold]Индекс:[/] {summary['index']}",
        title="День 21 · Индексация документов", border_style="cyan",
    ))

    guide = Table(title="Как отличаются стратегии", box=box.ROUNDED,
                  header_style="bold cyan", show_lines=True, expand=True)
    guide.add_column("Стратегия", style="bold", width=17)
    guide.add_column("Как делит текст", ratio=3)
    guide.add_column("Что это даёт", ratio=2)
    guide.add_row(
        "fixed\nПо размеру",
        f"Читает документ как сплошной текст. Окна до {MAX_TOKENS} токенов "
        f"с повтором {OVERLAP} токенов между соседними окнами.",
        "Контекст на стыке сохраняется, но один чанк может смешать два раздела. "
        "Метаданные section относятся к началу чанка.",
    )
    guide.add_row(
        "structure\nПо разделам и абзацам",
        f"Сначала делит по заголовкам Markdown, затем собирает абзацы до "
        f"{MAX_TOKENS} токенов. Длинный абзац режет отдельно; перекрытия нет.",
        "Разделы не смешиваются. Короткий раздел может стать маленьким чанком, "
        "иногда лишь с заголовком.",
    )
    guide.add_row(
        "section_windows\nРазделы + окна",
        f"Сначала делит по заголовкам. Внутри раздела создаёт окна до "
        f"{MAX_TOKENS} токенов с перекрытием до {OVERLAP}; путь заголовков "
        "повторяет в каждом чанке.",
        "Сохраняет тему и контекст внутри раздела, пропускает пустые заголовки. "
        "Повторы увеличивают объём индекса.",
    )
    console.print(guide)

    metrics_table = Table(title="Индекс и качество поиска", box=box.ROUNDED,
                          header_style="bold cyan", expand=True)
    for heading in ("Стратегия", "Чанков", "Ср. токенов", "Макс.", "hit@5", "MRR@5"):
        metrics_table.add_column(heading, justify="right" if heading != "Стратегия" else "left")
    best_mrr = max(comparison[strategy]["mrr_at_5"] for strategy in STRATEGIES)
    for strategy in STRATEGIES:
        metrics = comparison[strategy]
        score = Text(str(metrics["mrr_at_5"]),
                     style="bold green" if metrics["mrr_at_5"] == best_mrr else "")
        metrics_table.add_row(strategy, str(metrics["chunks"]),
                              str(metrics["mean_tokens"]), str(metrics["max_tokens"]),
                              f"{metrics['hit_at_5']}/{len(EVALUATION)}", score)
    console.print(metrics_table)

    examples_table = Table(title=f"Первый найденный чанк · «{query}»",
                           box=box.ROUNDED, header_style="bold cyan",
                           show_lines=True, expand=True)
    examples_table.add_column("Стратегия", style="bold", width=17)
    examples_table.add_column("Метаданные и фрагмент", ratio=1)
    for strategy in STRATEGIES:
        example = examples[strategy]
        excerpt = " ".join(example["text"].split())
        if len(excerpt) > 140:
            excerpt = excerpt[:140] + "…"
        examples_table.add_row(strategy,
                               f"source: {example['source']}\n"
                               f"title: {example['title']}\n"
                               f"section: {example['section']}\n"
                               f"chunk_id: {example['chunk_id']}\n"
                               f"text: {excerpt}")
    console.print(examples_table)

    ranks_table = Table(title="Одинаковые вопросы · место нужного README в top-5",
                        box=box.ROUNDED, header_style="bold cyan", expand=True)
    ranks_table.add_column("Вопрос", ratio=1, min_width=40)
    for strategy in STRATEGIES:
        ranks_table.add_column(strategy, justify="center",
                               width=17 if strategy == "section_windows" else 10,
                               no_wrap=True)
    for number, (query, _expected) in enumerate(EVALUATION):
        ranks: list[Text] = []
        for strategy in STRATEGIES:
            rank = comparison[strategy]["queries"][number]["rank_at_5"]
            ranks.append(Text(str(rank) if rank is not None else "—",
                              style="green" if rank == 1 else "yellow" if rank else "red"))
        ranks_table.add_row(f"{number + 1}. {query}", *ranks)
    console.print(ranks_table)

    hit_counts = {comparison[strategy]["hit_at_5"] for strategy in STRATEGIES}
    if len(hit_counts) == 1:
        coverage = (f"Все три стратегии нашли нужный README в top-5 для "
                    f"{hit_counts.pop()} из {len(EVALUATION)} запросов.")
    else:
        scores = ", ".join(f"{strategy} {comparison[strategy]['hit_at_5']}/{len(EVALUATION)}"
                           for strategy in STRATEGIES)
        coverage = f"Попаданий в top-5: {scores}."
    mrr_scores = ", ".join(f"{strategy}={comparison[strategy]['mrr_at_5']}"
                           for strategy in STRATEGIES)
    best_score = max(comparison[strategy]["mrr_at_5"] for strategy in STRATEGIES)
    winners = [strategy for strategy in STRATEGIES
               if comparison[strategy]["mrr_at_5"] == best_score]
    if len(winners) == len(STRATEGIES):
        ranking = f"MRR@5 одинаков: {mrr_scores}."
    else:
        ranking = f"Наибольший MRR@5 у {', '.join(winners)}; значения: {mrr_scores}."
    console.print(Panel(
        f"{coverage} {ranking}\n\n"
        "[dim]structure и section_windows сохраняют границы разделов. "
        "Метрика оценивает позицию документа, не полноту ответа; "
        "пять вопросов не дают общей оценки качества.[/]",
        title="Вывод", border_style="green",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Create all three local indexes")
    search_parser = subparsers.add_parser("search", help="Search one strategy")
    search_parser.add_argument("query")
    search_parser.add_argument("--strategy", choices=STRATEGIES, default="structure")
    search_parser.add_argument("--top-k", type=int, default=3)
    subparsers.add_parser("compare", help="Evaluate all three strategies")
    subparsers.add_parser("demo", help="Build and show strategies, metadata and comparison")
    args = parser.parse_args()
    if args.command not in ("build", "demo") and not args.index.is_file():
        parser.error(f"Index does not exist: {args.index}. Run build first.")
    model = load_model()
    if args.command == "demo":
        demo(args.index, model)
        return
    if args.command == "build":
        result = build(args.index, model)
    else:
        with closing(connect_index(args.index)) as connection:
            if args.command == "search":
                result = search(connection, args.query, model, args.strategy, args.top_k)
            else:
                result = compare(connection, model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
