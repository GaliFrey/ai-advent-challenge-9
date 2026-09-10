"""Локальная проверка итогового ответа демо-сценария."""

from __future__ import annotations

import unittest

from demo import DEMO_MESSAGES, evaluate_answer


class DemoTests(unittest.TestCase):
    def test_scenario_contains_fifteen_messages(self):
        self.assertEqual(len(DEMO_MESSAGES), 15)
        self.assertTrue(all("ПРИНЯТО" in message for message in DEMO_MESSAGES))

    def test_accepts_only_current_values(self):
        result = evaluate_answer(
            "ПРОЕКТ=ORBITA-7429; ДАТА=26.10.2026; БЮДЖЕТ=510000; "
            "ПОДРЯДЧИК=Меридиан; ОТКРЫТЫЙ_ВОПРОС=выбор кейтеринга"
        )
        self.assertEqual(result.score, result.total)

    def test_rejects_superseded_values(self):
        result = evaluate_answer(
            "ПРОЕКТ=ORBITA-7429; ДАТА=18.10.2026; БЮДЖЕТ=450000; "
            "ПОДРЯДЧИК=Атлас; ОТКРЫТЫЙ_ВОПРОС=выбор кейтеринга"
        )
        self.assertEqual(result.score, 2)


if __name__ == "__main__":
    unittest.main()
