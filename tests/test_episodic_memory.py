import os
import sys
import tempfile
import unittest
from dataclasses import dataclass


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.Episodic_Memory import EpisodicMemory


@dataclass
class FakeHit:
    id: str
    score: float
    payload: dict


class FakeQdrant:
    def __init__(self):
        self.points = []
        self.last_collection_name = None

    def upsert(self, collection_name, points):
        self.last_collection_name = collection_name
        self.points.extend(points)

    def search(self, collection_name, query_vector, limit):
        self.last_collection_name = collection_name
        return [
            FakeHit(
                id=point.id,
                score=0.9,
                payload=point.payload or {},
            )
            for point in self.points[:limit]
        ]


class EpisodicMemoryTest(unittest.TestCase):
    def test_add_and_retrieve_memory(self):
        qdrant = FakeQdrant()
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            memory = EpisodicMemory(qdrant, db_path=tmp.name, collection_name="episodic_test")
            memory.add_memory(
                memory_id="memory-1",
                session_id="session-1",
                content="用户问过 SENet 的通道注意力机制",
                vector=[0.1, 0.2, 0.3],
                importance_score=6,
            )

            results = memory.retrieve(query_vector=[0.1, 0.2, 0.3], top_k=1)

        self.assertEqual(qdrant.last_collection_name, "episodic_test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "用户问过 SENet 的通道注意力机制")
        self.assertIn("importance", results[0])
        self.assertIn("score", results[0])


if __name__ == "__main__":
    unittest.main()
