import os
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langgraph_adapter import LangGraphAgentRunner, LangGraphUnavailable


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


if __name__ == "__main__":
    unittest.main()
