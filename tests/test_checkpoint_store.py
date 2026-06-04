import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from checkpoint_store import SQLiteCheckpointStore


class SQLiteCheckpointStoreTest(unittest.TestCase):
    def test_save_and_load_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "checkpoints.sqlite3")
            store = SQLiteCheckpointStore(db_path=db_path, enabled=True)

            first_id = store.save("demo", "start", {"step": 1})
            second_id = store.save("demo", "final", {"step": 2})
            latest = store.latest("demo")
            all_items = store.list("demo")

        self.assertEqual(first_id, 1)
        self.assertEqual(second_id, 2)
        self.assertEqual(latest["node"], "final")
        self.assertEqual(latest["payload"]["step"], 2)
        self.assertEqual(len(all_items), 2)

    def test_disabled_store_is_noop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "checkpoints.sqlite3")
            store = SQLiteCheckpointStore(db_path=db_path, enabled=False)

            checkpoint_id = store.save("demo", "node", {"x": 1})

        self.assertIsNone(checkpoint_id)
        self.assertFalse(os.path.exists(db_path))


if __name__ == "__main__":
    unittest.main()
