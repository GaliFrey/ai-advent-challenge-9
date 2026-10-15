"""Explicit preparation of the pinned, CPU-only E5 model.

Only ``prepare-model`` accesses the network. Importing this module never imports
ML libraries, downloads files or loads weights. Runtime loading is local-only.
Index construction and evaluation are explicit operator commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import time
from urllib.request import urlopen

from .config import PROJECT_ROOT
from .errors import SupportError

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_DIR = PROJECT_ROOT / "data/models/multilingual-e5-small" / MODEL_REVISION
LIMIT = 512
DIMENSION = 384
# Official pinned metadata: SHA-256 for LFS objects; Git blob SHA-1 otherwise.
FILES = {
    "model.safetensors": (470641600, "sha256", "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
    "tokenizer.json": (17082730, "sha256", "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
    "config.json": (655, "git", "60a2a84020f1d74cc53ea9e8c4e91cf4af6c2b68"),
    "tokenizer_config.json": (443, "git", "059214673d9d6d2ee319411e2ffec8c024b816d5"),
    "modules.json": (387, "git", "ac2039abdf6ff023b27c919bb9675cfe378cb10f"),
    "1_Pooling/config.json": (200, "git", "acbd4014202bf4dbc6c03229bcbeef7c4feafb58"),
    "special_tokens_map.json": (167, "git", "e0b1d18ecd0ae4ff1d47bd297d910c0cf83e504b"),
    "sentence_bert_config.json": (57, "git", "4eca68d85ecd3034cf4174d8a4033a75344ea62d"),
}
EXPECTED_BYTES = 487726239
_MODEL = None
_MODEL_LOCK = threading.Lock()


def _missing_model() -> SupportError:
    return SupportError(
        "model_missing",
        "Локальная модель отсутствует или повреждена. Выполните "
        "uv run python -m server.retrieval prepare-model.",
        503,
    )


def verify_model_files(directory: Path) -> dict[str, str]:
    """Verify every byte before any model/configuration loader can run."""
    hashes = {}
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Invalid model directory")
        for name, (size, algorithm, expected) in FILES.items():
            path = directory / name
            if path.parent.is_symlink() or path.is_symlink():
                raise ValueError("Model symlink")
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                    raise ValueError("Model file size/type")
                digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
                sha256 = hashlib.sha256()
                if algorithm == "git":
                    digest.update(f"blob {size}\0".encode())
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    sha256.update(block)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns
                ) or digest.hexdigest() != expected:
                    raise ValueError("Model checksum/race")
                hashes[name] = sha256.hexdigest()
        # Pin module classes and paths, disallowing custom code and path escape.
        modules = json.loads((directory / "modules.json").read_text())
        expected_modules = [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ]
        if modules != expected_modules:
            raise ValueError("Unexpected modules")
    except (OSError, ValueError, TypeError) as exc:
        raise _missing_model() from exc
    return hashes


def _load_local_model(directory: Path):
    verify_model_files(directory)
    try:
        from sentence_transformers import SentenceTransformer

        local = {"local_files_only": True, "trust_remote_code": False}
        model = SentenceTransformer(
            str(directory), device="cpu", backend="torch", token=False,
            local_files_only=True, trust_remote_code=False,
            model_kwargs={**local, "use_safetensors": True},
            processor_kwargs=local.copy(), config_kwargs=local.copy(),
        )
        if model.get_sentence_embedding_dimension() != DIMENSION or model.max_seq_length != LIMIT:
            raise ValueError("Unexpected E5 dimensions/token limit")
        if model.tokenizer is None or not model.tokenizer.is_fast:
            raise ValueError("JSON tokenizer required")
        return model
    except Exception as exc:
        raise SupportError(
            "model_incompatible",
            "Не удалось открыть закреплённую CPU-модель локально. "
            "Проверьте установленное окружение и подготовку модели; "
            "автоматическое скачивание отключено.",
            503,
        ) from exc


def get_model():
    """One successful load per backend process; call from a worker thread."""
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = _load_local_model(MODEL_DIR)
        return _MODEL


def prepare_model() -> dict:
    """Explicit network operation; publish only a fully verified local model.

    No Hugging Face snapshot cache is used: only these eight objects are fetched.
    There are no application retries, remote code or alternate weight formats.
    """
    started = time.perf_counter()
    if MODEL_DIR.exists() or MODEL_DIR.is_symlink():
        hashes = verify_model_files(MODEL_DIR)
        _load_local_model(MODEL_DIR)
        return {"model": MODEL_ID, "revision": MODEL_REVISION, "downloadedBytes": 0,
                "localFileBytes": EXPECTED_BYTES, "sha256": hashes,
                "seconds": time.perf_counter() - started, "alreadyPrepared": True}
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=MODEL_DIR.parent))
    downloaded = 0
    try:
        for name, (size, _algorithm, _expected) in FILES.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            received = 0
            with urlopen(url, timeout=120) as source, target.open("xb") as output:
                while block := source.read(1024 * 1024):
                    received += len(block)
                    if received > size:
                        raise ValueError(f"Unexpected size: {name}")
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if received != size:
                raise ValueError(f"Incomplete download: {name}")
            downloaded += received
        hashes = verify_model_files(temporary)
        # Exercise the actual installed stack before publishing, without network.
        model = _load_local_model(temporary)
        text = "query: Как включить микрофон?"
        count = len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        if count > LIMIT:
            raise ValueError("Unexpected tokenizer budget")
        vector = model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
        import numpy as np
        if vector.shape != (1, DIMENSION) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector[0]), 1, atol=1e-5):
            raise ValueError("Invalid model output")
        result = {"model": MODEL_ID, "revision": MODEL_REVISION,
                  "downloadedBytes": downloaded, "localFileBytes": EXPECTED_BYTES,
                  "sha256": hashes, "seconds": time.perf_counter() - started,
                  "alreadyPrepared": False}
        manifest = temporary / "preparation.json"
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        # A concurrent successful preparation must not be replaced.
        os.rename(temporary, MODEL_DIR)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-model", "build-index", "evaluate"])
    parser.add_argument("--offline", action="store_true", help="Require local-only evaluation (always local-only)")
    args = parser.parse_args()
    if args.offline and args.command != "evaluate":
        parser.error("--offline is only valid with evaluate")
    try:
        if args.command == "prepare-model":
            result = prepare_model()
        else:
            from .retrieval_index import build_index, evaluate
            result = build_index() if args.command == "build-index" else evaluate()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("passed", True) else 1
    except Exception as exc:
        # Preparation is an operator command. Do not print URLs with credentials
        # or arbitrary exception strings; runtime failures never become success.
        print(json.dumps({"error": getattr(exc, "code", "model_preparation_failed"),
                          "message": exc.message if isinstance(exc, SupportError) else
                          "Операция retrieval не завершена; проверьте подготовку и индекс.",
                          "causeType": type(exc.__cause__ or exc).__name__}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
