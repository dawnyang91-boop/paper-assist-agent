import os
import sys
import unittest
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from importance_scorer import (
    build_importance_prompt,
    clamp_importance,
    rule_based_importance,
    score_memory_importance,
    score_overflow_memory,
)


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 9, "reason": "包含长期目标和强约束。"}')
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class ImportanceScorerTest(unittest.TestCase):
    def test_clamp_importance(self):
        self.assertEqual(clamp_importance(-5), 0)
        self.assertEqual(clamp_importance(11), 10)
        self.assertEqual(clamp_importance("6.4"), 6)

    def test_rule_based_scores_preferences_high(self):
        result = rule_based_importance("请记住，我以后默认喜欢简洁的中文回答。", role="user")
        self.assertGreaterEqual(result.score, 7)
        self.assertEqual(result.source, "rule")

    def test_rule_based_scores_greeting_low(self):
        result = rule_based_importance("谢谢", role="user")
        self.assertLessEqual(result.score, 2)

    def test_llm_prompt_contains_score_scale(self):
        prompt = build_importance_prompt("我的长期目标是完成智能问答系统。", role="user")
        self.assertIn("0-2: no long-term value", prompt)
        self.assertIn("9-10: stable identity information", prompt)
        self.assertIn("Return JSON only", prompt)

    def test_llm_score_with_fake_client(self):
        result = score_memory_importance(
            "我的核心目标是完成智能问答系统。",
            role="user",
            memory_type="episodic",
            use_llm=True,
            client=FakeClient(),
        )
        self.assertEqual(result.score, 9)
        self.assertEqual(result.source, "llm")

    def test_score_overflow_memory_uses_episodic_type(self):
        result = score_overflow_memory(
            {"role": "user", "content": "项目决定采用 Qdrant 存储向量。"},
            use_llm=False,
        )
        self.assertGreaterEqual(result.score, 7)


if __name__ == "__main__":
    unittest.main()
