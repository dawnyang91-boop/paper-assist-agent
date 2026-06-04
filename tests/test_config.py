import os
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config as config_module
from config import get_config


class ConfigTest(unittest.TestCase):
    def tearDown(self):
        get_config.cache_clear()

    def test_default_embedding_uses_local_minilm(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(config_module, "_ENV_FILES", ()):
            get_config.cache_clear()
            config = get_config()

        self.assertEqual(config.embed_model_type, "local")
        self.assertEqual(config.embed_model_name, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config.embed_vector_size, 384)

    def test_embed_vector_size_prefers_embed_variable(self):
        with patch.dict(os.environ, {"EMBED_VECTOR_SIZE": "768", "QDRANT_VECTOR_SIZE": "1024"}, clear=False):
            get_config.cache_clear()
            self.assertEqual(get_config().embed_vector_size, 768)

    def test_embed_vector_size_falls_back_to_legacy_qdrant_variable(self):
        with patch.dict(os.environ, {"QDRANT_VECTOR_SIZE": "1024"}, clear=False), patch.object(config_module, "_ENV_FILES", ()):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("EMBED_VECTOR_SIZE", None)
                get_config.cache_clear()
                self.assertEqual(get_config().embed_vector_size, 1024)

    def test_deepseek_base_url_enables_reasoning_and_thinking_defaults(self):
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "MODEL_NAME": "deepseek-v4-pro",
        }, clear=True):
            get_config.cache_clear()
            kwargs = get_config().chat_completion_kwargs()

        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "enabled"}})

    def test_llm_extra_body_can_override_thinking_payload(self):
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "LLM_REASONING_EFFORT": "medium",
            "LLM_EXTRA_BODY": '{"thinking": {"type": "disabled"}, "custom": true}',
        }, clear=True):
            get_config.cache_clear()
            kwargs = get_config().chat_completion_kwargs()

        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(kwargs["extra_body"]["thinking"], {"type": "disabled"})
        self.assertTrue(kwargs["extra_body"]["custom"])


if __name__ == "__main__":
    unittest.main()
