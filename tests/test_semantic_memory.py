import os
import sys
import unittest
from dataclasses import dataclass


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.Semantic_Memory import SemanticMemory


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
                id="chunk-1",
                score=0.8,
                payload={"page_content": "SENet 使用 SE block 建模通道依赖。"},
            ),
            FakeHit(
                id="chunk-2",
                score=0.6,
                payload={"content": "ResNet 是常见卷积网络基线。"},
            ),
        ][:limit]


class FakePollutedQdrant:
    def search(self, collection_name, query_vector, limit):
        return [
            FakeHit(
                id="polluted",
                score=0.9,
                payload={"page_content": "未配置 OPENAI_API_KEY，以下是基于检索上下文的摘要式回答：..."},
            ),
            FakeHit(
                id="clean",
                score=0.7,
                payload={"page_content": "处理长时程任务的工程方法包括压缩整合。"},
            ),
        ][:limit]


class FakeNeo4j:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def query(self, cypher, parameters):
        self.calls.append((cypher, parameters))
        return self.records


class SemanticMemoryTest(unittest.TestCase):
    def test_retrieve_reads_page_content_and_content_fallback(self):
        qdrant = FakeQdrant()
        memory = SemanticMemory(qdrant, collection_name="semantic_test")

        results = memory.retrieve(
            query="SENet 的核心是什么？",
            query_vector=[0.1, 0.2, 0.3],
            query_importance=5,
            top_k=2,
        )

        self.assertEqual(qdrant.last_collection_name, "semantic_test")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["content"], "SENet 使用 SE block 建模通道依赖。")
        self.assertEqual(results[1]["content"], "ResNet 是常见卷积网络基线。")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_retrieve_from_graph_scores_short_entity_paths(self):
        neo4j = FakeNeo4j([
            {
                "query_entity": "SENet",
                "candidate_entity": "SE block",
                "distance": 1,
                "relation_types": ["USES"],
            }
        ])
        memory = SemanticMemory(FakeQdrant(), collection_name="semantic_test", neo4j_client=neo4j)

        score = memory.retrieve_from_graph(
            "SENet 的核心贡献是什么？",
            candidate_content="SENet 使用 SE block 建模通道依赖。",
        )

        self.assertGreaterEqual(score, 0.95)
        self.assertEqual(neo4j.calls[0][1]["query_entities"][0], "SENet")
        self.assertIn("SE", " ".join(neo4j.calls[0][1]["candidate_entities"]))

    def test_retrieve_includes_graph_similarity_in_results(self):
        neo4j = FakeNeo4j([{"distance": 2, "relation_types": ["RELATED_TO"]}])
        memory = SemanticMemory(FakeQdrant(), collection_name="semantic_test", neo4j_client=neo4j)

        results = memory.retrieve(
            query="SENet 的核心是什么？",
            query_vector=[0.1, 0.2, 0.3],
            query_importance=5,
            top_k=1,
        )

        self.assertIn("graph_sim", results[0])
        self.assertGreater(results[0]["graph_sim"], 0)

    def test_retrieve_skips_polluted_context(self):
        memory = SemanticMemory(FakePollutedQdrant(), collection_name="semantic_test")

        results = memory.retrieve(
            query="DNA 的详细合成流程是什么？",
            query_vector=[0.1, 0.2, 0.3],
            query_importance=5,
            top_k=2,
        )

        self.assertEqual(len(results), 1)
        self.assertNotIn("未配置 OPENAI_API_KEY", results[0]["content"])


if __name__ == "__main__":
    unittest.main()
