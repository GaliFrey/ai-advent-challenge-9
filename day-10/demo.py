"""Один проверяемый сценарий для трёх стратегий контекста."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


ACK_SUFFIX = " Запомни сведения и ответь только: ПРИНЯТО."
COMMON_MESSAGES = (
    "Код проекта — FORUM-731. Мы готовим закрытую отраслевую конференцию; код нельзя переводить или сокращать."
    + ACK_SUFFIX,
    "Цель проекта — провести конференцию для 300 участников."
    + ACK_SUFFIX,
    "Для площадки обязательны безбарьерный вход и две независимые линии интернета."
    + ACK_SUFFIX,
    "Предварительная дата — 18.10.2026; монтаж запланирован на предыдущий день."
    + ACK_SUFFIX,
    "Первоначальный предельный бюджет — 450000 рублей, резерв входит в сумму."
    + ACK_SUFFIX,
    "Первоначально техническим подрядчиком выбрана компания «Атлас», договор ещё не подписан."
    + ACK_SUFFIX,
)
BRANCH_A_MESSAGES = (
    "Для варианта A дата 18.10.2026 отменена; актуальная дата — 26.10.2026."
    + ACK_SUFFIX,
    "В варианте A старый бюджет отменён; новый предел — 510000 рублей с резервом."
    + ACK_SUFFIX,
    "В варианте A «Атлас» отказался; актуальный подрядчик — компания «Меридиан»."
    + ACK_SUFFIX,
    "В варианте A открытым вопросом остаётся выбор кейтеринга."
    + ACK_SUFFIX,
)
BRANCH_B_MESSAGES = (
    "Для варианта B дата 18.10.2026 отменена; актуальная дата — 02.11.2026."
    + ACK_SUFFIX,
    "В варианте B старый бюджет отменён; новый предел — 560000 рублей с резервом."
    + ACK_SUFFIX,
    "В варианте B вместо «Атласа» выбран подрядчик «Вектор»."
    + ACK_SUFFIX,
    "В варианте B открытым вопросом остаётся выбор ведущего."
    + ACK_SUFFIX,
)
LINEAR_MESSAGES = COMMON_MESSAGES + BRANCH_A_MESSAGES
CHECK_MESSAGE = (
    "Восстанови актуальные данные этой версии проекта. Ответь одной строкой в формате: "
    "ПРОЕКТ=<код>; ДАТА=<дд.мм.гггг>; БЮДЖЕТ=<число>; ПОДРЯДЧИК=<название>; "
    "ДОСТУПНОСТЬ=<требование>; ИНТЕРНЕТ=<требование>; ОТКРЫТЫЙ_ВОПРОС=<текст>. "
    "Не добавляй пояснений."
)

EXPECTED_A = {
    "project": "FORUM-731",
    "date": "26.10.2026",
    "budget": "510000",
    "contractor": "Меридиан",
    "accessibility": "безбарьерный вход",
    "internet": "две независимые линии интернета",
    "open_question": "выбор кейтеринга",
}
EXPECTED_B = {
    "project": "FORUM-731",
    "date": "02.11.2026",
    "budget": "560000",
    "contractor": "Вектор",
    "accessibility": "безбарьерный вход",
    "internet": "две независимые линии интернета",
    "open_question": "выбор ведущего",
}


@dataclass(frozen=True)
class QualityResult:
    score: int
    total: int
    checks: tuple[tuple[str, bool], ...]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


def evaluate_answer(text: str, expected: Mapping[str, str]) -> QualityResult:
    normalized = _normalize(text)
    checks = tuple(
        (name, _normalize(value) in normalized) for name, value in expected.items()
    )
    return QualityResult(
        score=sum(passed for _, passed in checks),
        total=len(checks),
        checks=checks,
    )


def contamination_checks(text: str, *, expected_branch: str) -> tuple[tuple[str, bool], ...]:
    normalized = _normalize(text)
    forbidden = EXPECTED_B if expected_branch == "A" else EXPECTED_A
    return tuple(
        (f"no_foreign_{name}", _normalize(value) not in normalized)
        for name, value in forbidden.items()
        if name in {"date", "budget", "contractor", "open_question"}
    )
