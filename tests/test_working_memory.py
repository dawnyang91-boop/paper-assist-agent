import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.Working_Memory import WorkingMemory


class WorkingMemoryTest(unittest.TestCase):
    def test_retrieve_overflow_and_session_tools(self):
        memory = WorkingMemory(max_capacity=2, ttl_seconds=3600)
        memory.add_memory("s1", "user", "我正在阅读 SENet 论文", importance=5)
        memory.add_memory("s1", "assistant", "SENet 的核心是通道注意力", importance=6)
        memory.add_memory("s1", "user", "请总结 SE block", importance=5)

        overflow = memory.get_overflow_memories("s1")
        self.assertEqual(len(overflow), 1)
        self.assertEqual(overflow[0]["content"], "我正在阅读 SENet 论文")

        results = memory.retrieve("SENet 通道注意力", "s1", top_k=2)
        self.assertLessEqual(len(results), 2)
        self.assertIn("session_id", results[0])
        self.assertIn("score", results[0])

        history = memory.get_session_history("s1")
        self.assertEqual(len(history), 2)

        memory.clear_session("s1")
        self.assertEqual(memory.get_session_history("s1"), [])


if __name__ == "__main__":
    unittest.main()
