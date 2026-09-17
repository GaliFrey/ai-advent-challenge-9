"""Validation of structured model policy decisions and observable responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from configuration import Invariant, InvariantSet, Profile


@dataclass(frozen=True)
class Violation:
    invariant_id: str
    evidence: str
    explanation: str
    alternative: str


@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    violations: tuple[Violation, ...] = ()


@dataclass(frozen=True)
class ModelDecision:
    response: str
    result: PolicyResult


def _violation(rule: Invariant, evidence: str, explanation: str) -> Violation:
    return Violation(rule.invariant_id, evidence, explanation, rule.alternative)


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("ожидался JSON-объект")
    return value


def parse_model_decision(rules: InvariantSet, text: str) -> ModelDecision:
    """Validate the model's machine-readable semantic policy decision."""
    raw = _json_object(text)
    if set(raw) != {"decision", "violations", "response"}:
        raise ValueError("неверные поля policy-решения")
    decision, rows, response = raw["decision"], raw["violations"], raw["response"]
    if decision not in {"allow", "refuse"} or not isinstance(rows, list):
        raise ValueError("неверные decision или violations")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response должен быть непустой строкой")

    violations: list[Violation] = []
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"invariant_id", "evidence", "explanation"}:
            raise ValueError("неверная структура нарушения")
        invariant_id, evidence, explanation = row["invariant_id"], row["evidence"], row["explanation"]
        if not all(isinstance(item, str) and item.strip() for item in (invariant_id, evidence, explanation)):
            raise ValueError("поля нарушения должны быть непустыми строками")
        rule = rules.by_id(invariant_id)
        if rule is None:
            raise ValueError(f"неизвестный ID инварианта: {invariant_id}")
        if invariant_id in identifiers:
            raise ValueError(f"повтор ID инварианта: {invariant_id}")
        identifiers.add(invariant_id)
        violations.append(_violation(rule, evidence.strip(), explanation.strip()))

    allowed = not violations
    if (decision == "allow") != allowed:
        raise ValueError("decision не согласован со списком violations")
    return ModelDecision(response.strip(), PolicyResult(allowed, tuple(violations)))


def validate_response(profile: Profile, rules: InvariantSet, response: str) -> PolicyResult:
    """Check only properties observable in the generated user-visible response."""
    text = response.strip()
    found: list[Violation] = []
    if profile.profile_id == "tech-lead":
        prose = re.sub(r"```.*?```|`[^`]+`|https?://\S+", "", text, flags=re.DOTALL)
        cyrillic = len(re.findall(r"[А-Яа-яЁё]", prose))
        latin = len(re.findall(r"[A-Za-z]", prose))
        if latin > 40 and cyrillic < latin / 3:
            rule = rules.by_id("TECH-LANG-001")
            if rule:
                found.append(_violation(rule, text[:160], "Основной текст кандидата не на русском языке."))
        commands = "\n".join(re.findall(r"```(?:bash|shell|sh)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE))
        if re.search(r"\bpip\s+install\b|python\s+-m\s+venv|requirements\.txt", commands, re.IGNORECASE):
            rule = rules.by_id("TECH-UV-001")
            if rule:
                found.append(_violation(rule, commands[:160], "В командах ответа найден запрещённый Python-инструмент."))
        foreign_fences = re.findall(
            r"```(javascript|js|typescript|ts|java|kotlin|csharp|c#|cpp|c\+\+|go|rust|ruby|php|swift)\b",
            text,
            re.IGNORECASE,
        )
        if foreign_fences:
            rule = rules.by_id("TECH-CODE-001")
            if rule:
                found.append(_violation(rule, ", ".join(foreign_fences), "В ответе найден блок кода не на Python."))
    elif profile.profile_id == "english-tutor":
        prose = re.sub(r"```.*?```|`[^`]+`", "", text, flags=re.DOTALL)
        cyrillic = len(re.findall(r"[А-Яа-яЁё]", prose))
        latin = len(re.findall(r"[A-Za-z]", prose))
        if cyrillic > 20 and latin < cyrillic * 2:
            rule = rules.by_id("TUTOR-LANG-001")
            if rule:
                found.append(_violation(rule, text[:160], "The candidate's main prose is not in English."))
    return PolicyResult(not found, tuple(found))


def refusal_text(profile: Profile, violations: tuple[Violation, ...]) -> str:
    if profile.profile_id == "english-tutor":
        blocks = [f"REFUSED · {item.invariant_id}\n{item.explanation}\nAlternative: {item.alternative}" for item in violations]
    else:
        blocks = [f"ОТКАЗ · {item.invariant_id}\n{item.explanation}\nАльтернатива: {item.alternative}" for item in violations]
    return "\n\n".join(blocks)


def legacy_refusal_ids(response: str) -> tuple[str, ...]:
    """Recover old sessions created before structured decisions were persisted."""
    if re.search(r"\b(отказ|отказываюсь|refused)\b|не могу (?:выполнить|помочь)|конфликт с инвариантом",
                 response, re.IGNORECASE) is None:
        return ()
    return tuple(dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b", response.upper())))
