import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.Working_Memory import WorkingMemory
from memory.memory_manager import MemoryManager


class FakeEmbeddingService:
    def embed_text(self, text):
        return [1.0, float(len(text))]


class FakeEpisodicMemory:
    def __init__(self):
        self.added = []

    def add_memory(self, **kwargs):
        self.added.append(kwargs)

    def retrieve(self, query_vector, top_k=5):
        return [{"content": "episodic", "score": 1.0}]


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.created_collections = []

    def collection_exists(self, collection_name):
        return bool(self.created_collections)

    def create_collection(self, collection_name, vectors_config):
        self.created_collections.append((collection_name, vectors_config))

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))

    def search(self, collection_name, query_vector, limit):
        return []


class FakeQueryPointsResponse:
    def __init__(self, points):
        self.points = points


class FakeQueryPointsQdrant:
    def __init__(self, points=None):
        self.points = points or []

    def query_points(self, collection_name, query, limit):
        return FakeQueryPointsResponse(self.points[:limit])


class FakeNeo4j:
    def __init__(self):
        self.written = []

    def write_facts(self, facts, session_id=None):
        self.written.append((facts, session_id))


class FakeSemanticMemory:
    def __init__(self):
        self.qdrant = FakeQdrant()
        self.collection_name = "semantic_test"
        self.neo4j = FakeNeo4j()

    def retrieve(self, query, query_vector, query_importance, top_k=5):
        return [{"content": "semantic", "score": 1.0}]


class FailingQdrant(FakeQdrant):
    def upsert(self, collection_name, points):
        raise RuntimeError("Vector dimension error: expected dim: 1024, got 384")


class FailingSemanticMemory(FakeSemanticMemory):
    def __init__(self):
        super().__init__()
        self.qdrant = FailingQdrant()


class FailingRetrieveSemanticMemory(FakeSemanticMemory):
    def retrieve(self, query, query_vector, query_importance, top_k=5):
        raise RuntimeError("semantic retrieve failed")


class FakeSensoryMemory:
    def __init__(self):
        self.qdrant = FakeQdrant()
        self.collection_name = "sensory_test"

    def get_image_embedding(self, image_path):
        return [0.1, 0.2, 0.3]

    def retrieve(self, query_vector, top_k=3):
        return [{"payload": {"caption": "sensory"}, "score": 1.0}]


class DimMismatchSensoryMemory:
    def retrieve(self, query_vector, top_k=3):
        raise RuntimeError("Wrong input: Vector dimension error: expected dim: 512, got 1024")

    def get_text_embedding(self, text):
        raise RuntimeError("SensoryMemory 未加载中文 CLIP 模型，请使用 load_model=True 初始化。")


class SensoryTextEmbeddingMemory:
    def __init__(self):
        self.calls = []

    def retrieve(self, query_vector, top_k=3):
        self.calls.append(query_vector)
        if len(self.calls) == 1:
            raise RuntimeError("Wrong input: Vector dimension error: expected dim: 512, got 1024")
        return [{"payload": {"caption": "sensory fallback"}, "score": 1.0}]

    def get_text_embedding(self, text):
        return [0.1] * 512


class FakeChatCompletions:
    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "Compress the following overflowed working memories" in prompt or "压缩成一段情景记忆" in prompt:
            content = "用户询问 SENet，并关注 SE block 的通道依赖建模。"
        elif "Extract stable facts" in prompt or "抽取值得长期保存" in prompt:
            content = '["项目决定采用 Qdrant 存储语义向量。", "用户偏好简洁中文回答。"]'
        elif "caption" in prompt:
            content = "一张与智能问答系统相关的图片"
        else:
            content = "ok"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class MemoryManagerTest(unittest.TestCase):
    def build_manager(self, working_capacity=1, llm_client=None):
        working = WorkingMemory(max_capacity=working_capacity, ttl_seconds=3600)
        episodic = FakeEpisodicMemory()
        semantic = FakeSemanticMemory()
        sensory = FakeSensoryMemory()
        manager = MemoryManager(
            working_memory=working,
            episodic_memory=episodic,
            semantic_memory=semantic,
            sensory_memory=sensory,
            embedding_service=FakeEmbeddingService(),
            llm_client=llm_client,
        )
        return manager, working, episodic, semantic, sensory

    def test_working_overflow_is_summarized_into_episodic_memory(self):
        manager, _, episodic, _, _ = self.build_manager(llm_client=FakeLLMClient())

        results = manager.add_interaction(
            session_id="s1",
            user_message="我正在阅读 SENet 论文。",
            assistant_answer="SENet 的核心是 SE block。",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].overflow_count, 1)
        self.assertEqual(len(episodic.added), 1)
        self.assertIn("SENet", episodic.added[0]["content"])
        self.assertIn("vector", episodic.added[0])

    def test_high_value_qa_writes_semantic_fact_and_graph(self):
        manager, _, _, semantic, _ = self.build_manager(llm_client=FakeLLMClient())

        result = manager.process_high_value_qa(
            question="项目向量存储怎么选？",
            answer="项目决定采用 Qdrant 存储语义向量。用户偏好简洁中文回答。",
            session_id="s1",
        )

        self.assertGreaterEqual(len(result.facts), 1)
        self.assertGreaterEqual(len(result.semantic_ids), 1)
        self.assertTrue(result.graph_written)
        self.assertEqual(semantic.qdrant.upserts[0][0], "semantic_test")
        self.assertEqual(semantic.neo4j.written[0][1], "s1")

    def test_high_value_qa_does_not_crash_when_semantic_write_fails(self):
        manager, _, _, semantic, _ = self.build_manager(llm_client=FakeLLMClient())
        manager.semantic_memory = FailingSemanticMemory()

        result = manager.process_high_value_qa(
            question="项目向量存储怎么选？",
            answer="项目决定采用 Qdrant 存储语义向量。用户偏好简洁中文回答。",
            session_id="s1",
        )

        self.assertGreaterEqual(len(result.facts), 1)
        self.assertEqual(result.semantic_ids, [])
        self.assertTrue(result.errors)

    def test_high_value_qa_skips_polluted_fallback_answer(self):
        manager, _, _, semantic, _ = self.build_manager(llm_client=FakeLLMClient())

        result = manager.process_high_value_qa(
            question="DNA 的详细合成流程是什么？",
            answer="未配置 OPENAI_API_KEY，以下是基于检索上下文的摘要式回答：没有相关内容。",
            session_id="s1",
        )

        self.assertEqual(result.facts, [])
        self.assertEqual(result.semantic_ids, [])
        self.assertEqual(semantic.qdrant.upserts, [])

    def test_image_input_writes_sensory_memory(self):
        manager, _, _, _, sensory = self.build_manager(llm_client=FakeLLMClient())

        result = manager.process_media_input(
            session_id="s1",
            file_path="/tmp/智能问答系统.png",
            modality="image",
            importance=6,
        )

        self.assertEqual(result.modality, "image")
        self.assertIn("图片", result.caption)
        self.assertEqual(sensory.qdrant.upserts[0][0], "sensory_test")
        point = sensory.qdrant.upserts[0][1][0]
        self.assertEqual(point.payload["source_path"], "/tmp/智能问答系统.png")

    def test_audio_input_uses_transcriber_and_text_embedding(self):
        manager, _, _, _, sensory = self.build_manager()
        manager.audio_transcriber = lambda path: "用户说请总结 SENet"

        result = manager.process_media_input(
            session_id="s1",
            file_path="/tmp/question.wav",
            modality="audio",
        )

        self.assertEqual(result.modality, "audio")
        self.assertIn("用户说请总结 SENet", result.caption)
        point = sensory.qdrant.upserts[0][1][0]
        self.assertEqual(point.vector, [1.0, float(len(result.caption))])

    def test_retrieve_memories_queries_all_configured_modules(self):
        manager, working, _, _, _ = self.build_manager(working_capacity=3)
        working.add_memory("s1", "user", "SENet 通道注意力", importance=5)

        results = manager.retrieve_memories("SENet", session_id="s1", top_k=2)

        self.assertIn("working", results)
        self.assertIn("episodic", results)
        self.assertIn("semantic", results)
        self.assertIn("sensory", results)
        self.assertIn("_diagnostics", results)

    def test_retrieve_memories_reports_semantic_errors(self):
        manager, _, _, semantic, _ = self.build_manager()
        manager.semantic_memory = FailingRetrieveSemanticMemory()

        results = manager.retrieve_memories("SENet", session_id="s1", top_k=2)

        self.assertEqual(results["semantic"], [])
        self.assertIn("error", results["_diagnostics"]["semantic"])

    def test_sensory_dimension_mismatch_is_reported_as_skipped_when_model_unloaded(self):
        manager, _, _, _, _ = self.build_manager()
        manager.sensory_memory = DimMismatchSensoryMemory()

        results = manager.retrieve_memories("SENet", session_id="s1", top_k=2)

        self.assertEqual(results["sensory"], [])
        self.assertTrue(results["_diagnostics"]["sensory"]["skipped"])
        self.assertIn("expected dim: 512", results["_diagnostics"]["sensory"]["first_error"])

    def test_sensory_dimension_mismatch_can_fallback_to_sensory_text_embedding(self):
        manager, _, _, _, _ = self.build_manager()
        sensory = SensoryTextEmbeddingMemory()
        manager.sensory_memory = sensory

        results = manager.retrieve_memories("SENet", session_id="s1", top_k=2)

        self.assertEqual(results["sensory"][0]["payload"]["caption"], "sensory fallback")
        self.assertEqual(results["_diagnostics"]["sensory"]["fallback"], "sensory_text_embedding")
        self.assertEqual(len(sensory.calls[1]), 512)


if __name__ == "__main__":
    unittest.main()
