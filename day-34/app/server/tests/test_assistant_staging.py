"""C3/C9: staging, фиксированная проверка и атомарное применение."""

from __future__ import annotations

import json

import pytest

from server import assistant_staging, assistant_tools
from server.config import AssistantSettings
from server.errors import SupportError


def make_project(tmp_path):
    (tmp_path / "web" / "src").mkdir(parents=True)
    (tmp_path / "web" / "test").mkdir(parents=True)
    (tmp_path / "web" / "index.html").write_text("<html>old</html>\n", encoding="utf-8")
    (tmp_path / "web" / "style.css").write_text("body{}\n", encoding="utf-8")
    for name in ("app.js", "capture.js", "display.js", "pitch.js", "diagnostics.js", "support.js"):
        (tmp_path / "web" / "src" / name).write_text(f"// {name}\n", encoding="utf-8")
    (tmp_path / "web" / "src" / "assistant.js").write_text("// panel\n", encoding="utf-8")
    (tmp_path / "web" / "test" / "sample.test.mjs").write_text(
        'import test from "node:test"\ntest("ok", () => {})\n', encoding="utf-8"
    )
    return tmp_path


def test_staging_isolated_from_working_frontend(tmp_path):
    root = make_project(tmp_path)
    with assistant_staging.Staging.create(root) as staging:
        staging.write("web/src/app.js", "// changed in staging\n")
        assert "changed in staging" in staging.read("web/src/app.js")
        assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// app.js\n"


def test_apply_conflict_leaves_working_frontend_untouched(tmp_path):
    root = make_project(tmp_path)
    staging = assistant_staging.Staging.create(root)
    try:
        staging.write("web/src/app.js", "// staged app\n")
        staging.write("web/src/pitch.js", "// staged pitch\n")
        (root / "web" / "src" / "pitch.js").write_text("// external edit\n", encoding="utf-8")
        with pytest.raises(SupportError) as error:
            staging.apply(root)
        assert error.value.code == "apply_conflict"
        assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// app.js\n"
        assert (root / "web" / "src" / "pitch.js").read_text(encoding="utf-8") == "// external edit\n"
    finally:
        staging.cleanup()


def test_apply_writes_all_changed_paths(tmp_path):
    root = make_project(tmp_path)
    staging = assistant_staging.Staging.create(root)
    try:
        staging.write("web/src/app.js", "// new app\n")
        staging.write("web/index.html", "<html>new</html>\n")
        changed = staging.apply(root)
        assert changed == ["web/index.html", "web/src/app.js"]
        assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// new app\n"
        assert (root / "web" / "index.html").read_text(encoding="utf-8") == "<html>new</html>\n"
    finally:
        staging.cleanup()


def test_protected_files_are_copied_but_not_reachable(tmp_path):
    root = make_project(tmp_path)
    staging = assistant_staging.Staging.create(root)
    try:
        assert (staging.root / "web" / "src" / "assistant.js").is_file()
        assert staging.exists("web/src/assistant.js") is False
        with pytest.raises(SupportError):
            staging.read("web/src/assistant.js")
    finally:
        staging.cleanup()


def test_run_checks_builds_fixed_command(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    captured = {}

    class Completed:
        returncode = 0
        stdout = "# pass 1\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(assistant_staging.subprocess, "run", fake_run)
    staging = assistant_staging.Staging.create(root)
    try:
        result = staging.run_checks(timeout_seconds=5)
        assert result["passed"] is True
        assert captured["argv"][0] == "node"
        assert captured["argv"][1] == "--test"
        assert captured["argv"][2].endswith("web/test/sample.test.mjs")
        assert captured["kwargs"]["shell"] is False
    finally:
        staging.cleanup()


def test_run_checks_syntax_checks_changed_js(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    calls = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    monkeypatch.setattr(assistant_staging.subprocess, "run", fake_run)
    staging = assistant_staging.Staging.create(root)
    try:
        staging.write("web/src/app.js", "// changed app\n")
        result = staging.run_checks(timeout_seconds=5)
        assert result["passed"] is True
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[0] == "node"
        assert argv[1] == "--check"
        assert argv[2].endswith("web/src/app.js")
        assert kwargs["shell"] is False
        assert result["command"] == "node --check web/src/app.js"
    finally:
        staging.cleanup()


def test_run_checks_targets_changed_tests(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    calls = []

    class Completed:
        returncode = 0
        stdout = "# pass 1\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    monkeypatch.setattr(assistant_staging.subprocess, "run", fake_run)
    staging = assistant_staging.Staging.create(root)
    try:
        staging.write(
            "web/test/sample.test.mjs",
            'import test from "node:test"\ntest("ok", () => {})\n',
        )
        result = staging.run_checks(timeout_seconds=5)
        assert result["passed"] is True
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[0] == "node"
        assert argv[1] == "--test"
        assert argv[2].endswith("web/test/sample.test.mjs")
        assert kwargs["shell"] is False
    finally:
        staging.cleanup()


def test_run_checks_falls_back_to_full_suite_for_non_js(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    calls = []

    class Completed:
        returncode = 0
        stdout = "# pass 1\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    monkeypatch.setattr(assistant_staging.subprocess, "run", fake_run)
    staging = assistant_staging.Staging.create(root)
    try:
        staging.write("web/index.html", "<html>new</html>\n")
        result = staging.run_checks(timeout_seconds=5)
        assert result["passed"] is True
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[0] == "node"
        assert argv[1] == "--test"
        assert argv[2].endswith("web/test/sample.test.mjs")
        assert kwargs["shell"] is False
        assert result["command"] == "node --test web/test/*.mjs"
    finally:
        staging.cleanup()


def test_apply_requires_successful_check(tmp_path):
    root = make_project(tmp_path)
    staging = assistant_staging.Staging.create(root)
    tools = assistant_tools.AssistantTools(staging, settings=AssistantSettings())
    try:
        tools.execute("propose_edit", {"path": "web/src/app.js", "content": "// staged\n"})
        result = json.loads(tools.execute("apply", {}))
        assert result["error"]["code"] == "apply_requires_check"
        tools.last_check = {"passed": False, "output": "fail"}
        tools.edits_since_check = False
        result = json.loads(tools.execute("apply", {}))
        assert result["error"]["code"] == "apply_requires_check"
        assert (root / "web" / "src" / "app.js").read_text(encoding="utf-8") == "// app.js\n"
    finally:
        staging.cleanup()


def test_tools_deny_outside_scope_without_raising(tmp_path):
    root = make_project(tmp_path)
    staging = assistant_staging.Staging.create(root)
    tools = assistant_tools.AssistantTools(staging, settings=AssistantSettings())
    try:
        result = json.loads(tools.execute("read_file", {"path": "server/app.py"}))
        assert result["error"]["code"] == "path_forbidden"
        result = json.loads(tools.execute("propose_edit", {"path": "../.env", "content": "x"}))
        assert result["error"]["code"] == "path_forbidden"
    finally:
        staging.cleanup()


def test_no_dangerous_tool_schemas():
    names = {schema["function"]["name"] for schema in assistant_tools.TOOL_SCHEMAS}
    assert "commit" not in names and "push" not in names
    assert "install" not in names and "config" not in names
    assert names == {"list_files", "read_file", "search", "propose_edit", "run_checks", "apply", "finish"}
