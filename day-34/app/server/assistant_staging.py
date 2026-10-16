"""Staging-копия frontend и атомарное применение изменений ассистента.

Инструменты ассистента работают только со staging-копией ``web/`` во временном
каталоге. Рабочий frontend не изменяется, пока фиксированная проверка не
завершилась успешно и исходные allowlist-файлы не совпали с манифестом.
Применение идёт временным файлом и ``os.replace`` по каждому пути с
best-effort откатом, чтобы при сбое не оставалось частичных изменений.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import assistant_scope as scope
from . import config
from .errors import SupportError

# Фиксированный размер диффа, отдаваемого в историю.
MAX_DIFF_CHARS = 20000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Staging:
    """Изолированная копия ``web/`` с манифестом и безопасным применением."""

    def __init__(self, root: Path, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.manifest: dict[str, str | None] = {}
        self.changed: dict[str, str] = {}
        self.originals: dict[str, str] = {}

    # --- создание и очистка ------------------------------------------------

    @classmethod
    def create(cls, project_root: str | Path | None = None) -> "Staging":
        source_root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
        root = Path(tempfile.mkdtemp(prefix="gt-assistant-"))
        try:
            shutil.copytree(
                source_root / "web", root / "web", symlinks=True, dirs_exist_ok=False
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        staging = cls(root, source_root)
        staging._build_manifest()
        return staging

    def _build_manifest(self) -> None:
        self.manifest = {}
        for rel in scope.existing_allowed(self.root):
            self.manifest[rel] = _sha256(scope.resolve(self.root, rel))

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Staging":
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    # --- работа с файлами --------------------------------------------------

    def exists(self, rel: str) -> bool:
        try:
            return scope.resolve(self.root, rel).is_file()
        except SupportError:
            return False

    def read(self, rel: str) -> str:
        path = scope.resolve(self.root, rel, must_exist=True)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise SupportError(
                "path_not_text", "Файл не является текстом UTF-8.", 400
            ) from None

    def write(self, rel: str, text: str) -> None:
        if not isinstance(text, str):
            raise SupportError("edit_invalid", "Содержимое изменения должно быть строкой.", 400)
        path = scope.resolve(self.root, rel)
        if rel not in self.originals:
            self.originals[rel] = self.read(rel) if self.exists(rel) else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.changed[rel] = text

    def diff(self) -> str:
        chunks: list[str] = []
        for rel in sorted(self.changed):
            original = self.originals.get(rel, "")
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                self.changed[rel].splitlines(keepends=True),
                fromfile=rel,
                tofile=rel,
            )
            chunks.extend(diff)
        text = "".join(chunks)
        if len(text) > MAX_DIFF_CHARS:
            return text[:MAX_DIFF_CHARS] + "\n… (дифф сокращён)"
        return text

    # --- фиксированная проверка -------------------------------------------

    def _check_commands(self) -> list[tuple[list[str], str]] | None:
        """Фиксированные серверные команды по фактически изменённым файлам.

        Возвращает список ``(argv, описание)`` либо ``None``, если проверять нечего.
        Для изменённых ``.js`` — синтаксическая проверка ``node --check``; для
        изменённых ``.mjs``-тестов — ``node --test`` только по ним. Если изменены
        только неисполняемые файлы (``.html``/``.css``) или изменений нет —
        сохраняется полный прогон всех allowlist-тестов.
        """

        changed = sorted(self.changed)
        commands: list[tuple[list[str], str]] = []
        for rel in changed:
            if rel.endswith(".js"):
                commands.append(
                    (
                        ["node", "--check", (self.root / rel).as_posix()],
                        f"node --check {rel}",
                    )
                )
        tests = [
            rel
            for rel in changed
            if rel.startswith(scope.TEST_PREFIX) and rel.endswith(scope.TEST_SUFFIX)
        ]
        if tests:
            commands.append(
                (
                    ["node", "--test", *[(self.root / rel).as_posix() for rel in tests]],
                    "node --test " + " ".join(tests),
                )
            )
        if commands:
            return commands
        all_tests = scope.existing_allowed(self.root, tests_only=True)
        if not all_tests:
            return None
        return [
            (
                ["node", "--test", *[(self.root / rel).as_posix() for rel in all_tests]],
                "node --test web/test/*.mjs",
            )
        ]

    def run_checks(self, *, timeout_seconds: float) -> dict:
        commands = self._check_commands()
        if commands is None:
            return {
                "passed": False,
                "command": "node --test web/test/*.mjs",
                "returncode": None,
                "output": "Нет frontend-тестов в разрешённой области.",
            }
        command_text = "; ".join(description for _, description in commands)
        passed = True
        returncode: int | None = None
        outputs: list[str] = []
        for argv, description in commands:
            try:
                completed = subprocess.run(
                    argv,
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    shell=False,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "command": command_text,
                    "returncode": None,
                    "output": f"Проверка превысила отведённое время: {description}.",
                }
            except FileNotFoundError:
                return {
                    "passed": False,
                    "command": command_text,
                    "returncode": None,
                    "output": "Проверка недоступна в этом окружении.",
                }
            returncode = completed.returncode
            outputs.append((completed.stdout or "") + (completed.stderr or ""))
            if completed.returncode != 0:
                passed = False
                break
        output = "\n".join(part for part in outputs if part)
        if len(output) > 8000:
            output = output[-8000:]
        return {
            "passed": passed,
            "command": command_text,
            "returncode": returncode,
            "output": output,
        }

    # --- применение --------------------------------------------------------

    def apply(self, project_root: str | Path | None = None) -> list[str]:
        target_root = Path(project_root) if project_root is not None else self.project_root
        rels = sorted(self.changed)
        if not rels:
            return []

        conflicts: list[str] = []
        for rel in rels:
            real = scope.resolve(target_root, rel)
            current = _sha256(real) if real.is_file() else None
            if current != self.manifest.get(rel):
                conflicts.append(rel)
        if conflicts:
            raise SupportError(
                "apply_conflict",
                "Исходные frontend-файлы изменились после начала подготовки; изменения не применены.",
                409,
            )

        backups: dict[str, bytes | None] = {}
        written: list[str] = []
        try:
            for rel in rels:
                real = scope.resolve(target_root, rel)
                real.parent.mkdir(parents=True, exist_ok=True)
                backups[rel] = real.read_bytes() if real.is_file() else None
                tmp = real.with_name(real.name + ".gt-assistant.tmp")
                tmp.write_text(self.changed[rel], encoding="utf-8")
                os.replace(tmp, real)
                written.append(rel)
        except Exception:
            for rel in reversed(written):
                real = scope.resolve(target_root, rel)
                backup = backups.get(rel)
                try:
                    if backup is None:
                        real.unlink(missing_ok=True)
                    else:
                        real.write_bytes(backup)
                except OSError:
                    pass
            raise SupportError(
                "apply_failed",
                "Не удалось атомарно применить изменения; выполнен откат.",
                503,
            ) from None
        return rels
