"""Preparation boundary tests, independent of downloaded model weights."""
import hashlib
import json

import pytest

from server import retrieval
from server.errors import SupportError


def test_missing_model_does_not_attempt_download(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "MODEL_DIR", tmp_path / "missing")
    monkeypatch.setattr(retrieval, "_MODEL", None)

    def network_forbidden(*args, **kwargs):
        pytest.fail("Runtime attempted a model download")

    monkeypatch.setattr(retrieval, "urlopen", network_forbidden)
    with pytest.raises(SupportError) as failure:
        retrieval.get_model()
    assert failure.value.code == "model_missing"
    assert "uv run python -m server.retrieval prepare-model" in failure.value.message
    assert not (tmp_path / "missing").exists()


def test_failed_download_leaves_no_published_or_partial_model(tmp_path, monkeypatch):
    destination = tmp_path / "models" / retrieval.MODEL_REVISION
    monkeypatch.setattr(retrieval, "MODEL_DIR", destination)
    calls = []

    def failed_download(url, **kwargs):
        calls.append(url)
        raise OSError("interrupted transfer")

    monkeypatch.setattr(retrieval, "urlopen", failed_download)
    with pytest.raises(OSError):
        retrieval.prepare_model()
    assert len(calls) == 1  # no hidden retry
    assert retrieval.MODEL_REVISION in calls[0]
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_size_correct_corruption_is_rejected_despite_manifest(tmp_path, monkeypatch):
    expected = b"original"
    monkeypatch.setattr(retrieval, "FILES", {
        "model.safetensors": (len(expected), "sha256", hashlib.sha256(expected).hexdigest())
    })
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    (tmp_path / "preparation.json").write_text(json.dumps({"verified": True}))
    with pytest.raises(SupportError) as failure:
        retrieval.verify_model_files(tmp_path)
    assert failure.value.code == "model_missing"


def test_symlink_model_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / "real"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(SupportError) as failure:
        retrieval.verify_model_files(alias)
    assert failure.value.code == "model_missing"
