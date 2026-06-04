import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.rag_query import (
    QUERY_MODE_BASIC,
    QUERY_MODE_HYDE,
    QUERY_MODE_MQE,
    QueryGenerator,
    RAGQueryEngine,
)


class FakeChatCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][-1]["content"]
        if "Rewrite the same retrieval question" in prompt or "改写" in prompt:
            content = '["SENet 的创新点是什么？", "SE block 如何建模通道依赖？"]'
        else:
            content = "SENet 通过 SE block 对通道特征进行自适应重标定，从而提升 CNN 表征能力。"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


@dataclass
class FakeHit:
    id: str
    score: float
    payload: dict


class FakeQdrant:
    def __init__(self):
        self.search_calls = []

    def search(self, collection_name, query_vector, limit):
        self.search_calls.append({
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
        })
        call_index = len(self.search_calls)
        return [
            FakeHit(
                id=f"hit-{call_index}-1",
                score=0.7 + call_index / 100,
                payload={
                    "page_content": "SENet 使用 SE block 建模通道依赖。",
                    "content_hash": "same-content",
                    "source_file": "SENet.md",
                    "chunk_index": 1,
                },
            ),
            FakeHit(
                id=f"hit-{call_index}-2",
                score=0.5 + call_index / 100,
                payload={
                    "page_content": f"不同候选片段 {call_index}",
                    "content_hash": f"unique-{call_index}",
                    "source_file": "SENet.md",
                    "chunk_index": call_index + 1,
                },
            ),
        ][:limit]


class FakeQdrantQueryPoints:
    def __init__(self):
        self.query_points_calls = []

    def query_points(self, collection_name, query, limit):
        self.query_points_calls.append({
            "collection_name": collection_name,
            "query": query,
            "limit": limit,
        })
        return SimpleNamespace(points=[
            FakeHit(
                id="query-points-hit-1",
                score=0.88,
                payload={
                    "page_content": "新版 Qdrant SDK 使用 query_points 完成向量检索。",
                    "content_hash": "query-points-content",
                    "source_file": "Qdrant.md",
                    "chunk_index": 1,
                },
            )
        ])


class RAGQueryTest(unittest.TestCase):
    def test_query_generator_builds_basic_mqe_hyde(self):
        generator = QueryGenerator(llm_client=FakeLLMClient())
        variants = generator.build_queries(
            "SENet 的核心贡献是什么？",
            modes=(QUERY_MODE_BASIC, QUERY_MODE_MQE, QUERY_MODE_HYDE),
            mqe_count=2,
        )

        modes = [variant.mode for variant in variants]
        self.assertIn(QUERY_MODE_BASIC, modes)
        self.assertIn(QUERY_MODE_MQE, modes)
        self.assertIn(QUERY_MODE_HYDE, modes)
        self.assertEqual(len([item for item in modes if item == QUERY_MODE_MQE]), 2)
        self.assertTrue(any("SE block" in variant.text for variant in variants))

    def test_generate_query_vectors_for_three_modes(self):
        engine = RAGQueryEngine(
            qdrant_client=FakeQdrant(),
            collection_name="semantic_test",
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(llm_client=FakeLLMClient()),
        )
        variants = engine.generate_query_vectors("SENet 的核心贡献是什么？", mqe_count=2)

        self.assertEqual(len(variants), 4)
        self.assertTrue(all(variant.vector for variant in variants))
        self.assertEqual(variants[0].mode, QUERY_MODE_BASIC)

    def test_search_dedupes_retrieved_chunks(self):
        qdrant = FakeQdrant()
        engine = RAGQueryEngine(
            qdrant_client=qdrant,
            collection_name="semantic_test",
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(llm_client=FakeLLMClient()),
        )

        results = engine.search(
            "SENet 的核心贡献是什么？",
            top_k=5,
            per_query_limit=2,
            mqe_count=2,
        )

        self.assertEqual(len(qdrant.search_calls), 4)
        self.assertLess(len(results), 8)
        self.assertEqual(results[0].payload["content_hash"], "same-content")
        self.assertGreater(len(results[0].source_hits), 1)

    def test_search_supports_qdrant_query_points_api(self):
        qdrant = FakeQdrantQueryPoints()
        engine = RAGQueryEngine(
            qdrant_client=qdrant,
            collection_name="semantic_test",
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(llm_client=FakeLLMClient()),
        )

        results = engine.search(
            "Qdrant 新版 SDK 如何检索？",
            modes=(QUERY_MODE_BASIC,),
            top_k=1,
            per_query_limit=1,
        )

        self.assertEqual(len(qdrant.query_points_calls), 1)
        self.assertEqual(results[0].payload["content_hash"], "query-points-content")


if __name__ == "__main__":
    unittest.main()
