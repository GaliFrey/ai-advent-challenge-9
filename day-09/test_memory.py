"""Проверки полного и сжатого состояния диалога."""

from __future__ import annotations

import unittest

from memory import FullHistoryMemory, SummaryMemory


class FullHistoryMemoryTests(unittest.TestCase):
    def test_keeps_every_completed_exchange(self):
        memory = FullHistoryMemory()
        memory.commit_exchange("Вопрос 1", "Ответ 1")
        memory.commit_exchange("Вопрос 2", "Ответ 2")

        self.assertEqual(memory.raw_message_count, 4)
        self.assertEqual(memory.total_message_count, 4)
        self.assertIsNone(memory.summary)
        self.assertEqual(
            [message["role"] for message in memory.context_messages],
            ["user", "assistant", "user", "assistant"],
        )
        self.assertIsNone(memory.compression_plan())


class SummaryMemoryTests(unittest.TestCase):
    def test_summarizes_old_prefix_and_keeps_last_n_verbatim(self):
        memory = SummaryMemory(keep_recent=4, interval=10)
        self.assertEqual(memory.messages_until_summary, 10)
        for index in range(4):
            memory.commit_exchange(f"Вопрос {index}", f"Ответ {index}")
        self.assertEqual(memory.messages_until_summary, 2)
        memory.commit_exchange("Вопрос 4", "Ответ 4")
        self.assertEqual(memory.messages_until_summary, 0)

        plan = memory.compression_plan()
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.prefix_length, 6)
        self.assertEqual(plan.messages[0]["content"], "Вопрос 0")
        self.assertEqual(plan.messages[-1]["content"], "Ответ 2")

        memory.commit_summary(plan, "Сводка первых трёх ходов")

        self.assertEqual(memory.summary, "Сводка первых трёх ходов")
        self.assertEqual(memory.summary_version, 1)
        self.assertEqual(memory.messages_until_summary, 10)
        self.assertEqual(memory.summarized_message_count, 6)
        self.assertEqual(memory.raw_message_count, 4)
        self.assertEqual(memory.total_message_count, 10)
        self.assertEqual(memory.context_messages[0]["content"], "Вопрос 3")

    def test_recursive_plan_contains_previous_summary(self):
        memory = SummaryMemory(keep_recent=4, interval=10)
        for index in range(5):
            memory.commit_exchange(f"Вопрос {index}", f"Ответ {index}")
        first = memory.compression_plan()
        assert first is not None
        memory.commit_summary(first, "Первая сводка")

        for index in range(5, 10):
            memory.commit_exchange(f"Вопрос {index}", f"Ответ {index}")
        second = memory.compression_plan()
        assert second is not None

        self.assertEqual(second.previous_summary, "Первая сводка")
        self.assertEqual(second.prefix_length, 10)
        memory.commit_summary(second, "Обновлённая сводка")
        self.assertEqual(memory.summarized_message_count, 16)
        self.assertEqual(memory.raw_message_count, 4)
        self.assertEqual(memory.total_message_count, 20)
        self.assertEqual(memory.messages_until_summary, 10)

    def test_failed_or_stale_plan_does_not_remove_messages(self):
        memory = SummaryMemory(keep_recent=4, interval=10)
        for index in range(5):
            memory.commit_exchange(f"Вопрос {index}", f"Ответ {index}")
        plan = memory.compression_plan()
        assert plan is not None
        memory.clear()
        for index in range(5):
            memory.commit_exchange(f"Другой вопрос {index}", f"Другой ответ {index}")

        with self.assertRaisesRegex(RuntimeError, "изменилась"):
            memory.commit_summary(plan, "Опоздавшая сводка")

        self.assertEqual(memory.raw_message_count, 10)
        self.assertIsNone(memory.summary)

    def test_rejects_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "чётным"):
            SummaryMemory(keep_recent=3, interval=10)
        with self.assertRaisesRegex(ValueError, "меньше"):
            SummaryMemory(keep_recent=10, interval=10)


if __name__ == "__main__":
    unittest.main()
