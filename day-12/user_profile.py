"""Структурированные пользовательские профили и их атомарное хранение."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SAFE_ID = re.compile(r"[a-zA-Z0-9_-]+")


class ProfileError(RuntimeError):
    """Безопасная ошибка чтения, проверки или сохранения профиля."""


@dataclass(frozen=True)
class Profile:
    profile_id: str
    name: str
    address: str
    language: str
    style: str
    response_format: str
    constraints: str

    def check(self) -> None:
        if SAFE_ID.fullmatch(self.profile_id) is None:
            raise ValueError("ID профиля содержит недопустимые символы")
        for label, value in (
            ("Название", self.name),
            ("Обращение", self.address),
            ("Язык", self.language),
            ("Стиль", self.style),
            ("Формат", self.response_format),
            ("Ограничения", self.constraints),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Поле «{label}» не должно быть пустым")

    def normalized(self) -> Profile:
        self.check()
        return Profile(**{key: value.strip() for key, value in asdict(self).items()})


DEFAULT_PROFILES = (
    Profile(
        profile_id="tech-lead",
        name="Краткий техлид",
        address="Коллега",
        language="Русский",
        style="Деловой, прямой, без вступления и повторения вопроса",
        response_format="Ровно три коротких маркированных пункта",
        constraints="Не более 80 слов; без эмодзи; код только по прямому запросу",
    ),
    Profile(
        profile_id="mentor",
        name="Обучающий наставник",
        address="Ученик",
        language="Русский",
        style="Спокойный, поддерживающий, объяснять термины для начинающего",
        response_format="Разделы «План», «Пример» и «Проверка»",
        constraints="Дать небольшой пример; не использовать непояснённые сокращения; не более 250 слов",
    ),
)


def _atomic_save(path: Path, payload: Any) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary = Path(destination.name)
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ProfileError(f"Не удалось сохранить {path.name}") from error


def _profile(raw: Any) -> Profile:
    if not isinstance(raw, dict) or set(raw) != {
        "profile_id",
        "name",
        "address",
        "language",
        "style",
        "response_format",
        "constraints",
    }:
        raise ValueError("неверная структура профиля")
    profile = Profile(**raw).normalized()
    return profile


class ProfileStore:
    def __init__(self, root: Path, *, user_id: str = "demo") -> None:
        if SAFE_ID.fullmatch(user_id) is None:
            raise ValueError("user_id содержит недопустимые символы")
        self.path = root / "profiles" / f"{user_id}.json"

    def load(self) -> tuple[Profile, ...]:
        if not self.path.exists():
            return ()
        try:
            with self.path.open(encoding="utf-8") as source:
                raw = json.load(source)
        except json.JSONDecodeError as error:
            raise ProfileError(f"Повреждён JSON профилей: {self.path.name}") from error
        except OSError as error:
            raise ProfileError(f"Не удалось прочитать {self.path.name}") from error
        if not isinstance(raw, list):
            raise ProfileError("Файл профилей должен содержать JSON-массив")
        try:
            profiles = tuple(_profile(item) for item in raw)
        except (TypeError, ValueError) as error:
            raise ProfileError("Файл профилей имеет неверную структуру") from error
        ids = [item.profile_id for item in profiles]
        if len(ids) != len(set(ids)):
            raise ProfileError("ID профилей должны быть уникальными")
        return profiles

    def save_all(self, profiles: tuple[Profile, ...] | list[Profile]) -> None:
        try:
            checked = tuple(item.normalized() for item in profiles)
        except (TypeError, ValueError) as error:
            raise ProfileError("Нельзя сохранить неверный профиль") from error
        ids = [item.profile_id for item in checked]
        if not checked:
            raise ProfileError("Должен остаться хотя бы один профиль")
        if len(ids) != len(set(ids)):
            raise ProfileError("ID профилей должны быть уникальными")
        _atomic_save(self.path, [asdict(item) for item in checked])

    def ensure_defaults(self) -> tuple[Profile, ...]:
        profiles = self.load()
        if profiles:
            return profiles
        self.save_all(DEFAULT_PROFILES)
        return DEFAULT_PROFILES

    def upsert(self, profile: Profile) -> tuple[Profile, ...]:
        checked = profile.normalized()
        profiles = list(self.load())
        for index, current in enumerate(profiles):
            if current.profile_id == checked.profile_id:
                profiles[index] = checked
                break
        else:
            profiles.append(checked)
        self.save_all(profiles)
        return tuple(profiles)

    def delete(self, profile_id: str) -> tuple[Profile, ...]:
        profiles = tuple(item for item in self.load() if item.profile_id != profile_id)
        if len(profiles) == len(self.load()):
            raise ProfileError(f"Профиль {profile_id} не найден")
        self.save_all(profiles)
        return profiles


def render_profile(profile: Profile) -> str:
    """Преобразует активный профиль в явный блок system-инструкций."""
    checked = profile.normalized()
    return "\n".join(
        (
            "АКТИВНЫЙ ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ",
            f"Название: {checked.name}",
            f"Обращение: {checked.address}",
            f"Язык ответа: {checked.language}",
            f"Стиль: {checked.style}",
            f"Формат ответа: {checked.response_format}",
            f"Ограничения: {checked.constraints}",
            "Применяй этот профиль автоматически к текущему и каждому следующему ответу.",
        )
    )
