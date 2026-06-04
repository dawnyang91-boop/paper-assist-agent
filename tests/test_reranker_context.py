import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.context_builder import ContextBuilder
from config import AppConfig
from rag.rag_query import RetrievedChunk
from rag.reranker import CandidateReranker, lexical_overlap, topic_overlap


def make_chunk(content, score=0.5, mode="basic", content_hash=None, chunk_index=1, source_hits=None):
    return RetrievedChunk(
        id=content_hash or content,
        score=score,
        payload={
            "page_content": content,
            "content_hash": content_hash or content,
            "source_file": "SENet.md",
            "chunk_index": chunk_index,
            "heading_paths": ["研究核心"],
        },
        query_mode=mode,
        query_text="SENet 的核心贡献是什么？",
        source_hits=source_hits or [],
    )


class RerankerContextTest(unittest.TestCase):
    def test_lexical_overlap_handles_chinese_and_ascii_terms(self):
        score = lexical_overlap("SENet 的通道注意力是什么", "SENet 使用 SE block 建模通道依赖和注意力。")
        self.assertGreater(score, 0)

    def test_topic_overlap_ignores_generic_question_words(self):
        score = topic_overlap(
            "DNA 的详细合成流程是什么",
            "第七章介绍智能体系统、工具调用流程和上下文工程。",
        )

        self.assertEqual(score, 0.0)

    def test_topic_overlap_keeps_domain_terms(self):
        score = topic_overlap(
            "DNA 的详细合成流程是什么",
            "DNA 合成通常涉及复制起始、引物延伸和连接等步骤。",
        )

        self.assertGreater(score, 0.0)

    def test_reranker_promotes_relevant_multi_hit_chunk(self):
        candidates = [
            make_chunk("无关片段，讨论其他网络结构。", score=0.95, content_hash="other", chunk_index=2),
            make_chunk(
                "SENet 使用 SE block 建模通道依赖，并进行特征重标定。",
                score=0.75,
                content_hash="senet",
                source_hits=[
                    {"query_mode": "basic", "score": 0.75},
                    {"query_mode": "mqe", "score": 0.7},
                    {"query_mode": "hyde", "score": 0.68},
                ],
            ),
        ]
        ranked = CandidateReranker().rerank("SENet 如何建模通道依赖？", candidates, top_k=2)

        self.assertEqual(ranked[0].payload["content_hash"], "senet")
        self.assertGreater(ranked[0].lexical_score, ranked[1].lexical_score)
        self.assertEqual(ranked[0].rank, 1)

    def test_context_builder_builds_numbered_documents_and_memory(self):
        ranked = CandidateReranker().rerank(
            "SENet 的核心贡献是什么？",
            [
                make_chunk("SENet 使用 SE block 显式建模通道依赖。", score=0.9, content_hash="a", chunk_index=1),
                make_chunk("SE block 包括 squeeze 和 excitation 两步。", score=0.8, content_hash="b", chunk_index=2),
            ],
            top_k=2,
        )

        built = ContextBuilder().build_context(
            "SENet 的核心贡献是什么？",
            ranked,
            memories=[{"role": "user", "content": "用户正在阅读 SENet 论文。"}],
            top_k=2,
        )

        self.assertEqual(len(built.documents), 2)
        self.assertIn("[D1]", built.context)
        self.assertIn("[D2]", built.context)
        self.assertIn("[M1] user", built.context)
        self.assertIn("Do not invent facts", built.context)
        self.assertGreater(built.token_estimate, 0)

    def test_context_builder_top_k_zero_uses_token_budget_not_fixed_count(self):
        chunks = [
            make_chunk(f"第 {index} 个相关片段：SENet 与通道注意力。", score=1.0 - index * 0.01, content_hash=f"chunk-{index}")
            for index in range(1, 5)
        ]
        ranked = CandidateReranker().rerank("请综合说明 SENet 的贡献", chunks, top_k=4)
        builder = ContextBuilder(config=AppConfig(rag_context_top_k=2, rag_context_max_tokens=2000))

        built = builder.build_context("请综合说明 SENet 的贡献", ranked, top_k=0)

        self.assertEqual(len(built.documents), 4)
        self.assertIn("[D4]", built.context)

    def test_context_builder_trims_long_chunks_to_fit_more_sources(self):
        chunks = [
            make_chunk(("相关片段 " + str(index) + "。") * 160, score=1.0 - index * 0.01, content_hash=f"long-{index}")
            for index in range(1, 5)
        ]
        ranked = CandidateReranker().rerank("请综合说明 SENet 的贡献", chunks, top_k=4)
        builder = ContextBuilder(config=AppConfig(
            rag_context_top_k=0,
            rag_context_max_tokens=900,
            rag_context_max_doc_tokens=120,
        ))

        built = builder.build_context("请综合说明 SENet 的贡献", ranked, top_k=0)

        self.assertGreaterEqual(len(built.documents), 3)
        self.assertIn("[片段已截断以容纳更多引用来源]", built.context)


if __name__ == "__main__":
    unittest.main()
