"""Read-only adapter for day 21; no duplicate indexing implementation."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

DAY21 = Path(__file__).resolve().parent.parent / "day-21"
DEFAULT_INDEX = DAY21 / "indexes" / "readmes.sqlite3"
STRATEGY = "section_windows"
TOP_K = 5


def indexer_module():
    name = "day21_indexer"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, DAY21 / "indexer.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Не найден day-21/indexer.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def validate_index(connection, indexer) -> dict:
    indexer.check_model(connection)
    metadata = dict(connection.execute("SELECT key, value FROM meta"))
    digest = hashlib.sha256()
    for source, body in indexer.read_documents():
        digest.update(source.encode() + b"\0" + body.encode() + b"\0")
    if metadata.get("corpus_sha256") != digest.hexdigest():
        raise ValueError("Корпус изменился: пересоберите индекс командой build дня 21")
    if not connection.execute("SELECT 1 FROM chunks WHERE strategy = ? LIMIT 1", (STRATEGY,)).fetchone():
        raise ValueError("В индексе нет section_windows: выполните build дня 21")
    return metadata


class Retriever:
    def __init__(self, path: Path = DEFAULT_INDEX):
        self.path = path.resolve()
        self.indexer = indexer_module()
        # Validate before loading the relatively expensive embedding model.
        with closing(self.connect()) as connection:
            self.metadata = validate_index(connection, self.indexer)
        os.environ.setdefault("HF_HOME", str(DAY21 / "indexes" / "hf-cache"))
        cache = Path(os.environ["HF_HOME"]) / "hub"
        model_dir = "models--" + self.indexer.MODEL_ID.replace("/", "--")
        snapshot = cache / model_dir / "snapshots" / self.indexer.MODEL_REVISION
        # The day-21 model is already cached locally. Avoid remote metadata
        # checks that can hang or fail when the TUI has no network access.
        if (snapshot / "modules.json").is_file() and (snapshot / "model.safetensors").is_file():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.model = self.indexer.load_model()

    def connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def search(self, question: str) -> list[dict]:
        with closing(self.connect()) as connection:
            return self.indexer.search(connection, question, self.model, STRATEGY, TOP_K)
