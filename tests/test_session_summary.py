import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from session_summary import SessionSummaryBuilder


class SessionSummaryBuilderTest(unittest.TestCase):
    def test_builds_compact_transcript_memory(self):
        builder = SessionSummaryBuilder(max_chars=40)

        memory = builder.build_memory([
            {"role": "user", "content": "我正在研究 SENet"},
            {"role": "assistant", "content": "SENet 使用 SE block"},
        ])

        self.assertEqual(memory["role"], "transcript:summary")
        self.assertEqual(memory["type"], "transcript_summary")
        self.assertLessEqual(len(memory["content"]), 40)
        self.assertEqual(memory["metadata"]["event_count"], 2)


if __name__ == "__main__":
    unittest.main()
