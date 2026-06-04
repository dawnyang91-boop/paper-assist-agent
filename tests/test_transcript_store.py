import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from storage.transcript_store import TranscriptStore


class TranscriptStoreTest(unittest.TestCase):
    def test_append_and_load_jsonl_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TranscriptStore(root_dir=temp_dir, enabled=True)

            store.append("demo/session", "user", "你好", metadata={"turn": 1}, ts=1.0)
            store.append("demo/session", "assistant", "你好。", ts=2.0)
            events = store.load("demo/session")

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["role"], "user")
        self.assertEqual(events[0]["metadata"]["turn"], 1)
        self.assertEqual(events[1]["content"], "你好。")

    def test_disabled_store_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TranscriptStore(root_dir=temp_dir, enabled=False)

            event = store.append("demo", "user", "不会写入")
            events = store.load("demo")

        self.assertIsNone(event)
        self.assertEqual(events, [])

    def test_missing_transcript_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TranscriptStore(root_dir=temp_dir, enabled=True)

            self.assertEqual(store.load("missing"), [])

    def test_list_rename_delete_and_replace_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TranscriptStore(root_dir=temp_dir, enabled=True)
            store.append("demo", "user", "你好", ts=1.0)
            store.append("demo", "assistant", "你好。", ts=2.0)

            sessions = store.list_sessions()
            renamed = store.rename("demo", "renamed")
            store.replace("renamed", [{"role": "summary", "content": "摘要", "ts": 3.0}])
            deleted = store.delete("renamed")

        self.assertEqual(sessions[0]["session_id"], "demo")
        self.assertTrue(renamed)
        self.assertTrue(deleted)


if __name__ == "__main__":
    unittest.main()
