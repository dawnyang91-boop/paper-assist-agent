import os
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.langgraph_adapter import LangGraphAgentRunner, LangGraphUnavailable


class LangGraphAdapterTest(unittest.TestCase):
    def test_raises_clear_error_when_langgraph_missing(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("langgraph"):
                raise ImportError("missing langgraph")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(LangGraphUnavailable):
                LangGraphAgentRunner(agent_graph=object())

    def test_checkpoint_config_adds_thread_id_from_session_id(self):
        runner = object.__new__(LangGraphAgentRunner)

        config = runner._checkpoint_config({"session_id": "demo"})

        self.assertEqual(config["configurable"]["thread_id"], "demo")

    def test_checkpoint_config_preserves_existing_configurable_key(self):
        runner = object.__new__(LangGraphAgentRunner)

        config = runner._checkpoint_config(
            {"session_id": "demo"},
            config={"configurable": {"checkpoint_id": "cp-1"}},
        )

        self.assertEqual(config["configurable"]["checkpoint_id"], "cp-1")
        self.assertNotIn("thread_id", config["configurable"])


if __name__ == "__main__":
    unittest.main()
