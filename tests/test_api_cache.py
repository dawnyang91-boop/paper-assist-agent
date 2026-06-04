import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import api
except Exception:  # pragma: no cover
    api = None


@unittest.skipIf(api is None, "fastapi unavailable")
class ApiCacheTest(unittest.TestCase):
    def setUp(self):
        api.clear_assistant_cache()
        api.CONFIG_RUNTIME_SIGNATURE = None

    def tearDown(self):
        api.clear_assistant_cache()
        api.CONFIG_RUNTIME_SIGNATURE = None

    def test_get_cached_assistant_reuses_same_instance(self):
        agent = SimpleNamespace(mcp_manager=None)
        with patch.object(api, "_web_sensory_model_enabled", return_value=False):
            with patch.object(api, "build_assistant", return_value=(agent, "memory")) as build:
                first = api.get_cached_assistant()
                second = api.get_cached_assistant()

        self.assertIs(first, second)
        self.assertEqual(build.call_count, 1)
        self.assertFalse(build.call_args.kwargs["load_sensory_model"])

    def test_get_cached_assistant_separates_sensory_modes(self):
        default_agent = SimpleNamespace(mcp_manager=None)
        sensory_agent = SimpleNamespace(mcp_manager=None)
        with patch.object(api, "build_assistant", side_effect=[(default_agent, None), (sensory_agent, None)]) as build:
            with patch.object(api, "_web_sensory_model_enabled", return_value=False):
                first = api.get_cached_assistant()
            with patch.object(api, "_web_sensory_model_enabled", return_value=True):
                second = api.get_cached_assistant()

        self.assertIs(first[0], default_agent)
        self.assertIs(second[0], sensory_agent)
        self.assertEqual(build.call_count, 2)
        self.assertFalse(build.call_args_list[0].kwargs["load_sensory_model"])
        self.assertTrue(build.call_args_list[1].kwargs["load_sensory_model"])

    def test_clear_assistant_cache_closes_mcp_manager_when_supported(self):
        mcp_manager = SimpleNamespace(closed=False)

        def close():
            mcp_manager.closed = True

        mcp_manager.close = close
        agent = SimpleNamespace(mcp_manager=mcp_manager)
        with patch.object(api, "_web_sensory_model_enabled", return_value=False):
            with patch.object(api, "build_assistant", return_value=(agent, None)):
                api.get_cached_assistant()

        api.clear_assistant_cache()

        self.assertTrue(mcp_manager.closed)
        self.assertEqual(api.ASSISTANT_CACHE, {})

    def test_get_cached_assistant_rebuilds_when_runtime_config_changes(self):
        first_agent = SimpleNamespace(mcp_manager=None)
        second_agent = SimpleNamespace(mcp_manager=None)
        first_config = SimpleNamespace(
            openai_api_key=None,
            openai_base_url=None,
            model_name="gpt-4o-mini",
            qdrant_url=None,
            rag_collection_name="semantic_knowledge_base",
            sensory_model_enabled=False,
            mcp_enabled=False,
            mcp_tool_calling_enabled=False,
            chat_completion_kwargs=lambda: {},
        )
        second_config = SimpleNamespace(
            openai_api_key="sk-test-key",
            openai_base_url="https://api.deepseek.com",
            model_name="deepseek-v4-pro",
            qdrant_url=None,
            rag_collection_name="semantic_knowledge_base",
            sensory_model_enabled=False,
            mcp_enabled=False,
            mcp_tool_calling_enabled=False,
            chat_completion_kwargs=lambda: {"reasoning_effort": "high"},
        )

        with patch.object(api, "env_signature", return_value=(("mock.env", 1.0),)):
            with patch.object(api, "reload_config_from_env", side_effect=[first_config, second_config, second_config]):
                with patch.object(api, "_web_sensory_model_enabled", return_value=False):
                    with patch.object(api, "build_assistant", side_effect=[(first_agent, None), (second_agent, None)]) as build:
                        first = api.get_cached_assistant()
                        second = api.get_cached_assistant()
                        third = api.get_cached_assistant()

        self.assertIs(first[0], first_agent)
        self.assertIs(second[0], second_agent)
        self.assertIs(third[0], second_agent)
        self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
