"""Строгая граница путей веб-ассистента.

Разрешённая область задана явным набором здесь, в серверном коде вне области
записи ассистента. Общие frontend-файлы не участвуют в определении границы.
Проверка учитывает абсолютные пути, сегменты ``..`` и символические ссылки на
любом уровне пути. Любое нарушение — ``SupportError`` с безопасным кодом.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import SupportError

# Ровно эти обычные файлы frontend тюнера можно читать и изменять.
ALLOWED_FILES: tuple[str, ...] = (
    "web/index.html",
    "web/style.css",
    "web/src/app.js",
    "web/src/capture.js",
    "web/src/display.js",
    "web/src/pitch.js",
    "web/src/diagnostics.js",
    "web/src/support.js",
)
ALLOWED_FILE_SET = frozenset(ALLOWED_FILES)

# Прямые Node-тесты web/test/<имя>.mjs (без подкаталогов).
TEST_PREFIX = "web/test/"
TEST_SUFFIX = ".mjs"


def _forbidden() -> SupportError:
    return SupportError(
        "path_forbidden", "Путь вне разрешённой frontend-области.", 403
    )


def normalize_relative(raw: str) -> str:
    """Нормализует относительный путь, отклоняя абсолютные и ``..``."""

    if not isinstance(raw, str):
        raise _forbidden()
    if not raw or "\x00" in raw or raw.startswith(("/", "\\")):
        raise _forbidden()
    # Диск Windows (``C:``) и UNC-подобные входы запрещены.
    if len(raw) >= 2 and raw[1] == ":":
        raise _forbidden()
    parts: list[str] = []
    for segment in raw.replace("\\", "/").split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            raise _forbidden()
        parts.append(segment)
    if not parts:
        raise _forbidden()
    return "/".join(parts)


def is_allowed_relative(rel: str) -> bool:
    """Проверяет нормализованный путь против allowlist, не трогая файловую систему."""

    if rel in ALLOWED_FILE_SET:
        return True
    if rel.startswith(TEST_PREFIX):
        name = rel[len(TEST_PREFIX):]
        return (
            bool(name)
            and name != TEST_SUFFIX
            and name.endswith(TEST_SUFFIX)
            and "/" not in name
            and "\\" not in name
        )
    return False


def _contains_symlink(root: Path, rel: str) -> bool:
    current = root
    for part in rel.split("/"):
        current = current / part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def resolve(root: str | Path, raw: str, *, must_exist: bool = False) -> Path:
    """Разрешает путь внутри ``root`` или отклоняет его.

    ``root`` — канонический корень (корень проекта либо staging-корень), внутри
    которого лежит каталог ``web/``. Возвращается путь-кандидат; при
    ``must_exist=True`` обычный файл обязан существовать.
    """

    rel = normalize_relative(raw)
    if not is_allowed_relative(rel):
        raise _forbidden()

    root_real = Path(os.path.realpath(root))
    candidate = root_real / rel
    resolved = Path(os.path.realpath(candidate))
    if resolved != root_real and root_real not in resolved.parents:
        raise _forbidden()
    if _contains_symlink(root_real, rel):
        raise _forbidden()

    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_file():
            raise _forbidden()

    if must_exist and not candidate.is_file():
        raise SupportError(
            "path_missing", "Файл не найден в разрешённой области.", 404
        )
    return candidate


def existing_allowed(root: str | Path, *, tests_only: bool = False) -> list[str]:
    """Перечисляет существующие обычные allowlist-файлы относительно ``root``."""

    root_real = Path(os.path.realpath(root))
    found: list[str] = []
    for rel in ALLOWED_FILES:
        if tests_only:
            continue
        try:
            candidate = resolve(root_real, rel)
        except SupportError:
            continue
        if candidate.is_file():
            found.append(rel)
    if not tests_only:
        test_dir = root_real / TEST_PREFIX.rstrip("/")
        if test_dir.is_dir() and not test_dir.is_symlink():
            try:
                entries = sorted(test_dir.iterdir(), key=lambda item: item.name)
            except OSError:
                entries = []
            for entry in entries:
                rel = TEST_PREFIX + entry.name
                if not is_allowed_relative(rel):
                    continue
                try:
                    candidate = resolve(root_real, rel)
                except SupportError:
                    continue
                if candidate.is_file():
                    found.append(rel)
    else:
        test_dir = root_real / TEST_PREFIX.rstrip("/")
        if test_dir.is_dir() and not test_dir.is_symlink():
            try:
                entries = sorted(test_dir.iterdir(), key=lambda item: item.name)
            except OSError:
                entries = []
            for entry in entries:
                rel = TEST_PREFIX + entry.name
                if not is_allowed_relative(rel):
                    continue
                try:
                    candidate = resolve(root_real, rel)
                except SupportError:
                    continue
                if candidate.is_file():
                    found.append(rel)
    return found
