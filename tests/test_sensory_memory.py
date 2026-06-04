import os
import sys
import time
import unittest
from dataclasses import dataclass


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Sensory_Memory import SensoryMemory
from config import AppConfig


@dataclass
class FakeHit:
    id: str
    score: float
    payload: dict


class FakeQdrant:
    def __init__(self):
        self.last_collection_name = None

    def search(self, collection_name, query_vector, limit):
        self.last_collection_name = collection_name
        return [
            FakeHit(
                id="image-1",
                score=0.7,
                payload={"caption": "一张智能问答系统思维导图", "timestamp": time.time(), "importance": 6},
            ),
            FakeHit(
                id="image-2",
                score=0.4,
                payload={"caption": "其他图片", "timestamp": time.time(), "importance": 3},
            ),
        ][:limit]


class SensoryMemoryTest(unittest.TestCase):
    def test_retrieve_without_loading_clip_model(self):
        qdrant = FakeQdrant()
        memory = SensoryMemory(qdrant, collection_name="sensory_test", load_model=False)

        results = memory.retrieve(query_vector=[0.1, 0.2, 0.3], top_k=2)

        self.assertEqual(qdrant.last_collection_name, "sensory_test")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "image-1")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_embedding_requires_loaded_model(self):
        memory = SensoryMemory(FakeQdrant(), load_model=False)
        with self.assertRaises(RuntimeError):
            memory.get_text_embedding("智能问答系统")

    def test_resolve_model_source_prefers_existing_local_path(self):
        config = AppConfig(sensory_model_local_path=os.path.dirname(__file__))
        memory = SensoryMemory(FakeQdrant(), config=config, load_model=False)

        self.assertEqual(memory.model_source, os.path.realpath(os.path.dirname(__file__)))

    def test_from_pretrained_kwargs_use_local_files_only_and_token(self):
        config = AppConfig(
            sensory_model_local_files_only=True,
            hf_token="hf_test",
        )
        memory = SensoryMemory(FakeQdrant(), config=config, load_model=False)

        self.assertEqual(memory._from_pretrained_kwargs()["local_files_only"], True)
        self.assertEqual(memory._from_pretrained_kwargs()["token"], "hf_test")


if __name__ == "__main__":
    unittest.main()
