"""Проверки трёх структур памяти."""

from __future__ import annotations

import unittest

from memory import BranchingMemory, FactsMemory, SlidingWindowMemory


class SlidingWindowTests(unittest.TestCase):
    def test_only_last_n_messages_are_exposed_to_context(self):
        memory = SlidingWindowMemory(keep_recent=4)
        for index in range(4):
            memory.commit_exchange(f"Вопрос {index}", f"Ответ {index}")

        self.assertEqual(memory.total_message_count, 8)
        self.assertEqual(memory.context_message_count, 4)
        self.assertEqual(memory.discarded_message_count, 4)
        self.assertEqual(memory.context_messages[0]["content"], "Вопрос 2")
        self.assertEqual(memory.context_messages[-1]["content"], "Ответ 3")

    def test_rejects_window_that_breaks_exchange_pairs(self):
        with self.assertRaisesRegex(ValueError, "чётным"):
            SlidingWindowMemory(3)


class FactsMemoryTests(unittest.TestCase):
    def test_facts_are_separate_from_recent_messages(self):
        memory = FactsMemory(keep_recent=4)
        memory.commit_exchange("Код FORUM-731", "ПРИНЯТО", facts={"project": "FORUM-731"})
        memory.commit_exchange("Дата 18.10", "ПРИНЯТО", facts={"project": "FORUM-731", "date": "18.10"})
        memory.commit_exchange("Дата 26.10", "ПРИНЯТО", facts={"project": "FORUM-731", "date": "26.10"})

        self.assertEqual(memory.total_message_count, 6)
        self.assertEqual(memory.discarded_message_count, 2)
        self.assertEqual(memory.facts, {"project": "FORUM-731", "date": "26.10"})
        self.assertNotIn("Код FORUM-731", [item["content"] for item in memory.context_messages])

    def test_returned_facts_are_a_copy(self):
        memory = FactsMemory(4)
        memory.replace_facts({"nested": {"value": 1}})
        returned = memory.facts
        returned["nested"]["value"] = 2
        self.assertEqual(memory.facts["nested"]["value"], 1)


class BranchingMemoryTests(unittest.TestCase):
    def test_branches_share_prefix_and_keep_suffixes_isolated(self):
        memory = BranchingMemory()
        memory.commit_exchange("Общий факт", "ПРИНЯТО")
        memory.create_checkpoint()
        memory.commit_exchange("Только A", "Ответ A")
        memory.switch_branch("B")
        memory.commit_exchange("Только B", "Ответ B")

        branch_b = [item["content"] for item in memory.context_messages]
        self.assertIn("Общий факт", branch_b)
        self.assertIn("Только B", branch_b)
        self.assertNotIn("Только A", branch_b)

        memory.switch_branch("A")
        branch_a = [item["content"] for item in memory.context_messages]
        self.assertIn("Только A", branch_a)
        self.assertNotIn("Только B", branch_a)
        self.assertEqual(memory.total_message_count, 6)

    def test_checkpoint_requires_existing_dialogue_and_is_unique(self):
        memory = BranchingMemory()
        with self.assertRaisesRegex(ValueError, "пустом"):
            memory.create_checkpoint()
        memory.commit_exchange("Вопрос", "Ответ")
        memory.create_checkpoint()
        with self.assertRaisesRegex(ValueError, "уже создан"):
            memory.create_checkpoint()


if __name__ == "__main__":
    unittest.main()
