import os
import sys
import unittest
from dataclasses import replace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import get_config
from rag.rag_query import RetrievedChunk
from rag.reranker import CandidateReranker


class EntityAwareRerankTest(unittest.TestCase):
    def test_strong_entity_candidate_beats_generic_candidate(self):
        config = replace(
            get_config(),
            rag_entity_aware_enabled=True,
            rag_entity_required_for_strong_query=True,
            rag_rerank_entity_weight=0.6,
            rag_rerank_entity_missing_penalty=0.4,
        )
        reranker = CandidateReranker(config=config)
        generic = RetrievedChunk(
            id="generic",
            score=0.95,
            payload={"page_content": "This document explains generic retrieval workflows and agent systems."},
            query_mode="vector_basic",
            query_text="MGHT 的核心方法是什么？",
        )
        entity = RetrievedChunk(
            id="entity",
            score=0.65,
            payload={
                "page_content": "MGHT introduces a multi-granularity hierarchical transformer for recommendation.",
                "acronyms": ["MGHT"],
            },
            query_mode="bm25",
            query_text="MGHT 的核心方法是什么？",
        )

        ranked = reranker.rerank("MGHT 的核心方法是什么？", [generic, entity], top_k=2)

        self.assertEqual(ranked[0].chunk.id, "entity")
        self.assertGreater(ranked[0].entity_score, 0)
        self.assertGreater(ranked[1].entity_missing_penalty, 0)


if __name__ == "__main__":
    unittest.main()
