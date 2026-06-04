import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from fastapi.testclient import TestClient
    import api
except Exception:  # pragma: no cover
    TestClient = None
    api = None


class FakeAgent:
    def __init__(self):
        self.answer_kwargs = None

    def answer(self, question, session_id=None, write_memory=True, **kwargs):
        self.answer_kwargs = kwargs
        stream_callback = kwargs.get("stream_callback")
        if stream_callback:
            stream_callback("SENet 使用 ")
            stream_callback("SE block。[D1]")
        document = SimpleNamespace(
            doc_id="D1",
            source_file="SENet.md",
            chunk_index=1,
            score=0.9,
            heading_paths=["核心贡献"],
        )
        return SimpleNamespace(
            answer="SENet 使用 SE block。[D1]",
            built_context=SimpleNamespace(documents=[document]),
            metadata={"stop_reason": "final"},
        )


@dataclass
class FakeFactWriteResult:
    facts: list
    graph_written: bool = False


@unittest.skipIf(TestClient is None, "fastapi test client unavailable")
class ApiTest(unittest.TestCase):
    def setUp(self):
        api.clear_assistant_cache()
        self.client = TestClient(api.app)

    def tearDown(self):
        api.clear_assistant_cache()

    def test_health(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ask_returns_trace_when_requested(self):
        agent = FakeAgent()
        with patch.object(api, "build_assistant", return_value=(agent, None)) as build:
            with patch.object(api, "_web_sensory_model_enabled", return_value=False):
                response = self.client.post(
                    "/ask",
                    json={"question": "SENet?", "session_id": "demo", "show_trace": True},
                )

        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["answer"], "SENet 使用 SE block。[D1]")
        self.assertEqual(data["sources"][0]["doc_id"], "D1")
        self.assertEqual(data["trace"]["stop_reason"], "final")
        self.assertTrue(agent.answer_kwargs["transcript_enabled"])
        self.assertFalse(build.call_args.kwargs["load_sensory_model"])
        self.assertEqual(build.call_count, 1)

    def test_ask_rejects_empty_question(self):
        response = self.client.post("/ask", json={"question": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "empty_question")

    def test_stream_returns_ndjson_events(self):
        agent = FakeAgent()
        with patch.object(api, "build_assistant", return_value=(agent, None)) as build:
            with patch.object(api, "_web_sensory_model_enabled", return_value=False):
                response = self.client.post(
                    "/ask/stream",
                    json={"question": "SENet?", "session_id": "demo", "show_trace": True},
                )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"event": "start"', response.text)
        self.assertIn('"event": "token"', response.text)
        self.assertIn('"event": "answer"', response.text)
        self.assertIn('"event": "done"', response.text)
        self.assertTrue(agent.answer_kwargs["transcript_enabled"])
        self.assertFalse(build.call_args.kwargs["load_sensory_model"])
        self.assertEqual(build.call_count, 1)

    def test_stream_loads_sensory_model_when_enabled(self):
        agent = FakeAgent()
        with patch.object(api, "build_assistant", return_value=(agent, None)) as build:
            with patch.object(api, "_web_sensory_model_enabled", return_value=True):
                response = self.client.post(
                    "/ask/stream",
                    json={"question": "SENet?", "session_id": "demo"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(build.call_args.kwargs["load_sensory_model"])

    def test_web_assistant_is_cached_across_requests(self):
        agent = FakeAgent()
        with patch.object(api, "build_assistant", return_value=(agent, None)) as build:
            first = self.client.post("/ask", json={"question": "SENet?", "session_id": "demo"})
            second = self.client.post("/ask", json={"question": "SE block?", "session_id": "demo"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(build.call_count, 1)

    def test_assistant_cache_keeps_separate_sensory_modes(self):
        default_agent = FakeAgent()
        sensory_agent = FakeAgent()
        with patch.object(api, "build_assistant", side_effect=[(default_agent, None), (sensory_agent, None)]) as build:
            with patch.object(api, "_web_sensory_model_enabled", return_value=False):
                first = self.client.post("/ask", json={"question": "SENet?", "session_id": "demo"})
            with patch.object(api, "_web_sensory_model_enabled", return_value=True):
                second = self.client.post("/ask", json={"question": "图片里有什么?", "session_id": "demo"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(build.call_count, 2)
        self.assertFalse(build.call_args_list[0].kwargs["load_sensory_model"])
        self.assertTrue(build.call_args_list[1].kwargs["load_sensory_model"])

    def test_chunk_text_splits_answer(self):
        chunks = list(api._chunk_text("abcdef", size=2))

        self.assertEqual(chunks, ["ab", "cd", "ef"])

    def test_json_line_handles_dataclass_trace(self):
        line = api._json_line({
            "event": "answer",
            "data": {"trace": {"writeback_events": [{"result": FakeFactWriteResult(facts=["x"])}]}},
        })

        self.assertIn('"facts": ["x"]', line)
        self.assertIn('"graph_written": false', line)

    def test_web_index_is_available(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("私域问答助手", response.text)

    def test_session_management_endpoints(self):
        with patch.object(api, "_transcript_store") as store_factory:
            store = store_factory.return_value
            store.list_sessions.return_value = [{"session_id": "demo", "event_count": 1}]
            store.load.return_value = [{"role": "user", "content": "你好"}]
            store.rename.return_value = True
            store.delete.return_value = True
            store.replace.return_value = None

            list_response = self.client.get("/sessions")
            info_response = self.client.get("/sessions/demo")
            rename_response = self.client.post("/sessions/demo/rename", json={"new_session_id": "new-demo"})
            summary_response = self.client.post("/sessions/demo/summary", json={"max_events": 5})
            delete_response = self.client.delete("/sessions/demo")

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(info_response.status_code, 200)
        self.assertEqual(rename_response.status_code, 200)
        self.assertEqual(summary_response.status_code, 200)
        self.assertEqual(delete_response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
