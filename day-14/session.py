"""Atomic session storage and append-only policy audit."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from configuration import SAFE_ID


class SessionError(RuntimeError):
    pass


def _atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _message(value: Any, roles: set[str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise ValueError
    if value["role"] not in roles or not isinstance(value["content"], str) or not value["content"].strip():
        raise ValueError
    return {"role": value["role"], "content": value["content"].strip()}


@dataclass
class Session:
    session_id: str
    profile_id: str
    invariant_set_id: str
    invariant_version: int
    invariant_hash: str
    messages: list[dict[str, str]] = field(default_factory=list)
    last_request_messages: list[dict[str, str]] = field(default_factory=list)
    last_checks: list[dict[str, str]] = field(default_factory=list)
    policy_history: list[dict[str, Any]] = field(default_factory=list)

    def checked(self) -> Session:
        if SAFE_ID.fullmatch(self.session_id) is None or SAFE_ID.fullmatch(self.profile_id) is None:
            raise ValueError
        if SAFE_ID.fullmatch(self.invariant_set_id) is None or self.invariant_version < 1:
            raise ValueError
        if len(self.invariant_hash) != 64:
            raise ValueError
        messages = [_message(item, {"user", "assistant"}) for item in self.messages]
        if len(messages) % 2 or any(item["role"] != ("user" if i % 2 == 0 else "assistant") for i, item in enumerate(messages)):
            raise ValueError
        prompt = [_message(item, {"system", "user", "assistant"}) for item in self.last_request_messages]
        checks = []
        for item in self.last_checks:
            if not isinstance(item, dict) or set(item) != {"id", "status", "detail"}:
                raise ValueError
            checks.append({key: str(value) for key, value in item.items()})
        policy_history = []
        for item in self.policy_history:
            if not isinstance(item, dict) or set(item) != {"decision", "violations"}:
                raise ValueError
            if item["decision"] not in {"ALLOWED", "REFUSED"} or not isinstance(item["violations"], list):
                raise ValueError
            violations = item["violations"]
            if any(not isinstance(value, str) or not value for value in violations):
                raise ValueError
            if (item["decision"] == "ALLOWED") != (not violations):
                raise ValueError
            policy_history.append({"decision": item["decision"], "violations": list(violations)})
        if len(policy_history) > len(messages) // 2:
            raise ValueError
        return Session(self.session_id, self.profile_id, self.invariant_set_id, self.invariant_version,
                       self.invariant_hash, messages, prompt, checks, policy_history)


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, session_id: str) -> Path:
        if SAFE_ID.fullmatch(session_id) is None:
            raise SessionError("Неверный ID сессии")
        return self.root / "sessions" / f"{session_id}.json"

    def save(self, session: Session) -> None:
        try:
            checked = session.checked()
            _atomic_save(self.path(checked.session_id), asdict(checked))
        except (OSError, TypeError, ValueError) as error:
            raise SessionError("Не удалось сохранить сессию") from error

    def load(self, session_id: str) -> Session:
        try:
            raw = json.loads(self.path(session_id).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError
            return Session(**raw).checked()
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SessionError(f"Сессия {session_id} повреждена") from error

    def create(self, session_id: str, profile_id: str, set_id: str, version: int, content_hash: str) -> Session:
        if self.path(session_id).exists():
            raise SessionError(f"Сессия {session_id} уже существует")
        session = Session(session_id, profile_id, set_id, version, content_hash)
        self.save(session)
        return session

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in (self.root / "sessions").glob("session-*.json")))

    def append_audit(self, session: Session, event: dict[str, Any]) -> None:
        path = self.root / "audits" / f"{session.session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_id": session.profile_id,
            "invariant_set_id": session.invariant_set_id,
            "invariant_version": session.invariant_version,
            "invariant_hash": session.invariant_hash,
            **event,
        }
        try:
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as error:
            raise SessionError("Не удалось записать audit") from error
