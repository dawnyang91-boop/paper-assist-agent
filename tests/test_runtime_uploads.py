import io
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import AppConfig
from rag.qdrant_utils import QdrantVectorDimensionError, ensure_collection_vector_size
from storage.background_tasks import LocalTaskStore
from storage.redis_cache import RedisJsonCache
from storage.upload_store import UploadStore


class RuntimeUploadTest(unittest.TestCase):
    def test_upload_store_saves_lists_and_deletes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(rag_upload_dir=tmpdir, rag_upload_allowed_extensions=".md")
            store = UploadStore(config=config)
            upload = SimpleNamespace(filename="note.md", file=io.BytesIO(b"# hello"))

            saved = store.save_files([upload], user_id="user@example.com")

            self.assertEqual(len(saved), 1)
            self.assertTrue(os.path.exists(saved[0]["path"]))
            self.assertEqual(len(store.list_files(user_id="user@example.com")), 1)
            self.assertTrue(store.delete_file(saved[0]["file_id"], user_id="user@example.com"))
            self.assertEqual(store.list_files(user_id="user@example.com"), [])

    def test_upload_store_replace_files_resets_pending_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(rag_upload_dir=tmpdir, rag_upload_allowed_extensions=".md")
            store = UploadStore(config=config)

            first = SimpleNamespace(filename="old.md", file=io.BytesIO(b"# old"))
            second = SimpleNamespace(filename="new.md", file=io.BytesIO(b"# new"))
            store.save_files([first], user_id="user@example.com")
            saved = store.replace_files([second], user_id="user@example.com")
            files = store.list_files(user_id="user@example.com")

            self.assertEqual(len(saved), 1)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["file_id"], saved[0]["file_id"])

    def test_local_task_store_tracks_status(self):
        store = LocalTaskStore()
        task_id = store.create_task("rag_upload_ingest", user_id="u1")
        store.update_task(task_id, {"status": "running", "progress": 50})
        task = store.get_task(task_id)

        self.assertEqual(task["type"], "rag_upload_ingest")
        self.assertEqual(task["status"], "running")
        self.assertEqual(task["progress"], "50")

    def test_redis_cache_disabled_when_runtime_missing(self):
        cache = RedisJsonCache(runtime=None, config=AppConfig(redis_enabled=False))

        self.assertFalse(cache.enabled)
        self.assertIsNone(cache.make_key("hyde", "question"))

    def test_qdrant_dimension_mismatch_fails_early(self):
        vectors = SimpleNamespace(size=1024)
        params = SimpleNamespace(vectors=vectors)
        collection = SimpleNamespace(config=SimpleNamespace(params=params))
        qdrant = SimpleNamespace(
            collection_exists=lambda collection_name: True,
            get_collection=lambda collection_name: collection,
        )

        with self.assertRaises(QdrantVectorDimensionError):
            ensure_collection_vector_size(qdrant, "semantic_knowledge_base", 384)


if __name__ == "__main__":
    unittest.main()
