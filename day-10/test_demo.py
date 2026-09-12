"""Проверки сценария и детерминированной оценки качества."""

from __future__ import annotations

import unittest

from demo import (
    BRANCH_A_MESSAGES,
    BRANCH_B_MESSAGES,
    COMMON_MESSAGES,
    EXPECTED_A,
    LINEAR_MESSAGES,
    contamination_checks,
    evaluate_answer,
)


ANSWER_A = (
    "ПРОЕКТ=FORUM-731; ДАТА=26.10.2026; БЮДЖЕТ=510000; ПОДРЯДЧИК=Меридиан; "
    "ДОСТУПНОСТЬ=безбарьерный вход; ИНТЕРНЕТ=две независимые линии интернета; "
    "ОТКРЫТЫЙ_ВОПРОС=выбор кейтеринга"
)


class DemoTests(unittest.TestCase):
    def test_each_root_to_leaf_scenario_has_ten_messages_before_check(self):
        self.assertEqual(len(COMMON_MESSAGES), 6)
        self.assertEqual(len(BRANCH_A_MESSAGES), 4)
        self.assertEqual(len(BRANCH_B_MESSAGES), 4)
        self.assertEqual(len(LINEAR_MESSAGES), 10)

    def test_current_values_pass_all_checks(self):
        quality = evaluate_answer(ANSWER_A, EXPECTED_A)
        self.assertEqual(quality.score, quality.total)
        isolation = contamination_checks(ANSWER_A, expected_branch="A")
        self.assertTrue(all(passed for _, passed in isolation))

    def test_old_or_foreign_values_fail(self):
        answer = ANSWER_A.replace("26.10.2026", "02.11.2026").replace("Меридиан", "Вектор")
        quality = evaluate_answer(answer, EXPECTED_A)
        self.assertEqual(quality.score, quality.total - 2)
        isolation = contamination_checks(answer, expected_branch="A")
        self.assertFalse(all(passed for _, passed in isolation))


if __name__ == "__main__":
    unittest.main()
