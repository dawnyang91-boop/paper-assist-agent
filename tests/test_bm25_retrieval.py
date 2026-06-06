import os
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import get_config
from rag.bm25_index import BM25ChunkIndex, BM25Document
from rag.rag_query import QUERY_MODE_BASIC, QueryGenerator, RAGQueryEngine


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[1.0, float(len(text))] for text in texts]


class FakeQdrant:
    def search(self, collection_name, query_vector, limit):
        return [
            SimpleNamespace(
                id="generic-vector",
                score=0.82,
                payload={
                    "page_content": "This chunk discusses generic agent workflow and RAG retrieval.",
                    "content_hash": "generic-vector",
                    "source_file": "agent.md",
                    "chunk_index": 1,
                },
            )
        ]


class BM25RetrievalTest(unittest.TestCase):
    def test_hybrid_search_adds_bm25_entity_candidate(self):
        config = replace(
            get_config(),
            rag_bm25_enabled=True,
            rag_bm25_index_path="/tmp/nonexistent-bm25-test.pkl",
            rag_hybrid_candidate_k=10,
        )
        engine = RAGQueryEngine(
            qdrant_client=FakeQdrant(),
            collection_name="semantic_test",
            config=config,
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(config=config, llm_client=None),
        )
        engine.bm25_index = BM25ChunkIndex.build([
            BM25Document(
                doc_id="mght-doc",
                content="MGHT proposes multi-granularity hierarchical transformer modeling for recommendation.",
                source_file="2026-IJCAI-MGHT.pdf",
                chunk_index=3,
                payload={
                    "page_content": "MGHT proposes multi-granularity hierarchical transformer modeling for recommendation.",
                    "content_hash": "mght-doc",
                    "source_file": "2026-IJCAI-MGHT.pdf",
                    "chunk_index": 3,
                    "acronyms": ["MGHT"],
                },
            )
        ])

        results = engine.search(
            "MGHT 的核心方法是什么？",
            modes=(QUERY_MODE_BASIC,),
            top_k=2,
            per_query_limit=1,
        )

        self.assertTrue(any(item.payload.get("source_file") == "2026-IJCAI-MGHT.pdf" for item in results))
        diagnostics = engine.last_retrieval_diagnostics
        self.assertTrue(diagnostics["bm25_index_loaded"])
        self.assertIn("MGHT", diagnostics["entity_profile"]["entities"])
        self.assertGreaterEqual(diagnostics["bm25_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
