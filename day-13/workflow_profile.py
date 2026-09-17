"""Новые профили, задающие стадии и роли агента дня 13."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowProfile:
    profile_id: str
    name: str
    purpose: str
    planning: str
    execution: str
    validation: str
    revision: str

    def instruction_for(self, phase: str) -> str:
        try:
            return {
                "planning": self.planning,
                "execution": self.execution,
                "validation": self.validation,
                "revision": self.revision,
            }[phase]
        except KeyError as error:
            raise ValueError(f"У профиля нет стадии {phase}") from error


WORKFLOW_PROFILES = (
    WorkflowProfile(
        profile_id="engineering",
        name="Инженерный результат",
        purpose="Получить проверяемое техническое решение без пропуска существенных рисков.",
        planning=(
            "Ты технический планировщик. Раздели цель ровно на три небольших, "
            "последовательных и проверяемых шага. Не выполняй их."
        ),
        execution=(
            "Ты инженер-исполнитель. Выполни только текущий шаг, учитывая исходную цель, "
            "утверждённый план и результаты прежних шагов. Не переходи к следующему шагу."
        ),
        validation=(
            "Ты независимый технический проверяющий. Сопоставь результаты всех шагов с "
            "исходной целью. Фиксируй только доказанные нарушения и дай точную инструкцию исправления."
        ),
        revision=(
            "Ты инженер по доработке. Исправь совокупный результат строго по замечаниям "
            "валидатора. Верни целиком обновлённый результат, а не перечень изменений."
        ),
    ),
    WorkflowProfile(
        profile_id="explainer",
        name="Объясняющий материал",
        purpose="Подготовить понятный и фактически аккуратный учебный материал.",
        planning=(
            "Ты редактор учебных материалов. Раздели работу ровно на три шага: структура, "
            "содержательное раскрытие и практическая проверка. Не пиши сам материал."
        ),
        execution=(
            "Ты автор учебного материала. Выполни только текущий редакционный шаг. Пиши "
            "понятно для новичка, но не упрощай факты и учитывай уже готовые части."
        ),
        validation=(
            "Ты выпускающий редактор. Проверь полноту, непротиворечивость и понятность всех "
            "частей. Фиксируй только доказанные нарушения и дай точную инструкцию исправления."
        ),
        revision=(
            "Ты выпускающий автор. Переработай материал по замечаниям валидатора и верни "
            "цельную обновлённую версию без служебных комментариев."
        ),
    ),
)


def get_profile(profile_id: str) -> WorkflowProfile:
    profile = next((item for item in WORKFLOW_PROFILES if item.profile_id == profile_id), None)
    if profile is None:
        raise ValueError(f"Неизвестный профиль: {profile_id}")
    return profile
