"""C4: граница путей веб-ассистента обеспечивается кодом, а не промптом."""

from __future__ import annotations

import pytest

from server import assistant_scope as scope
from server.errors import SupportError


def make_root(tmp_path):
    (tmp_path / "web" / "src").mkdir(parents=True)
    (tmp_path / "web" / "test").mkdir(parents=True)
    for rel in ("web/index.html", "web/style.css"):
        (tmp_path / rel).write_text("x", encoding="utf-8")
    for name in ("app.js", "capture.js", "display.js", "pitch.js", "diagnostics.js", "support.js"):
        (tmp_path / "web" / "src" / name).write_text("// x", encoding="utf-8")
    (tmp_path / "web" / "test" / "page.test.mjs").write_text("// t", encoding="utf-8")
    # Защищённые файлы и каталоги, которые должны быть недоступны.
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "app.py").write_text("secret", encoding="utf-8")
    (tmp_path / ".env").write_text("KEY=1", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "intakes").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("r", encoding="utf-8")
    (tmp_path / "web" / "src" / "assistant.js").write_text("// panel", encoding="utf-8")
    (tmp_path / "web" / "test" / "sub").mkdir()
    (tmp_path / "web" / "test" / "sub" / "x.test.mjs").write_text("// t", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("out", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("rel", [
    "web/index.html",
    "web/style.css",
    "web/src/app.js",
    "web/src/capture.js",
    "web/src/display.js",
    "web/src/pitch.js",
    "web/src/diagnostics.js",
    "web/src/support.js",
    "web/test/page.test.mjs",
    "web/test/new.test.mjs",
])
def test_allowed_paths_resolve(tmp_path, rel):
    root = make_root(tmp_path)
    resolved = scope.resolve(root, rel)
    assert resolved == root / rel
    assert scope.is_allowed_relative(scope.normalize_relative(rel))


@pytest.mark.parametrize("raw", [
    "/etc/passwd",
    "web/../../etc/passwd",
    "../server/app.py",
    "web/src/../../.env",
    "server/app.py",
    ".env",
    ".git/config",
    "data/support.db",
    "tasks/I013/task.json",
    "intakes/I013/idea.md",
    "docs/architecture.md",
    "README.md",
    "web/README.md",
    "web/src/assistant.js",
    "web/test/sub/x.test.mjs",
    "web/test/",
    "web/src",
    "",
    "   ",
    "web\\..\\..\\etc",
])
def test_forbidden_paths_rejected(tmp_path, raw):
    root = make_root(tmp_path)
    with pytest.raises(SupportError) as error:
        scope.resolve(root, raw)
    assert error.value.code == "path_forbidden"


def test_null_byte_and_non_string_rejected(tmp_path):
    root = make_root(tmp_path)
    for raw in ("web/src/app.js\x00", None, 5):
        with pytest.raises(SupportError):
            scope.resolve(root, raw)


def test_symlink_component_rejected(tmp_path):
    root = make_root(tmp_path)
    outside = root / "outside.txt"
    link = root / "web" / "src" / "app.js"
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(SupportError) as error:
        scope.resolve(root, "web/src/app.js")
    assert error.value.code == "path_forbidden"


def test_symlink_directory_rejected(tmp_path):
    root = make_root(tmp_path)
    import shutil

    shutil.rmtree(root / "web" / "test")
    (root / "elsewhere").mkdir()
    (root / "elsewhere" / "x.test.mjs").write_text("// x", encoding="utf-8")
    (root / "web" / "test").symlink_to(root / "elsewhere")
    with pytest.raises(SupportError):
        scope.resolve(root, "web/test/x.test.mjs")


def test_existing_allowed_lists_only_regular_files(tmp_path):
    root = make_root(tmp_path)
    listed = scope.existing_allowed(root)
    assert "web/index.html" in listed
    assert "web/src/app.js" in listed
    assert "web/test/page.test.mjs" in listed
    assert "web/test/sub/x.test.mjs" not in listed
    assert "web/src/assistant.js" not in listed
    assert "server/app.py" not in listed
