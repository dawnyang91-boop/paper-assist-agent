import os
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.qa_agent import QAAgent
from agent_dag.dag_context import AgentContext
from agent_dag.evidence_store import EvidenceItem, SharedEvidenceStore
from config import AppConfig
from rag.rag_query import QUERY_MODE_BASIC, QueryGenerator, RAGQueryEngine
from sentinel.runtime.evidence_policy import ShareLevel


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[1.0, float(len(text))] for text in texts]


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="SENet 通过 SE block 建模通道依赖。[D1]"))]
        )


class FakeLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


@dataclass
class FakeHit:
    id: str
    score: float
    payload: dict


class FakeQdrant:
    def search(self, collection_name, query_vector, limit):
        return [
            FakeHit(
                id="hit-1",
                score=0.9,
                payload={
                    "page_content": "SENet 通过 SE block 显式建模通道依赖。",
                    "content_hash": "senet",
                    "source_file": "SENet.md",
                    "chunk_index": 1,
                },
            )
        ]


def build_agent(config=None):
    config = config or AppConfig(openai_api_key="test-key", agent_runtime="dag")
    query_engine = RAGQueryEngine(
        qdrant_client=FakeQdrant(),
        collection_name="semantic_test",
        config=config,
        embedding_service=FakeEmbeddingService(),
        query_generator=QueryGenerator(config=config, llm_client=None),
    )
    return QAAgent(
        query_engine=query_engine,
        config=config,
        llm_client=FakeLLMClient(),
    )


def test_agent_context_copy_isolates_mutable_state_and_filters_privileged_context():
    context = AgentContext(
        task_id="task-1",
        node_id="root",
        role="root",
        user_query="hello",
        user_context={"session_id": "demo", "nested": {"a": 1}},
        privileged_context={"agent": object(), "secret": "hidden"},
    )

    child = context.copy_for_node(
        node_id="writer",
        role="writer",
        permissions=["read_context"],
        allowed_tools=[],
        privileged_keys=["agent"],
    )
    child.user_context["nested"]["a"] = 2

    assert context.user_context["nested"]["a"] == 1
    assert "agent" in child.privileged_context
    assert "secret" not in child.privileged_context
    assert child.permissions == ["read_context"]


def test_evidence_store_hides_privileged_and_high_risk_evidence_from_writer():
    store = SharedEvidenceStore()
    store.add(EvidenceItem(
        evidence_id="safe-doc",
        task_id="task-1",
        source_node="rag",
        source_type="retriever",
        content="safe",
        sanitized_content="safe",
        share_level=ShareLevel.CITED_EVIDENCE.value,
    ))
    store.add(EvidenceItem(
        evidence_id="private-memory",
        task_id="task-1",
        source_node="memory",
        source_type="memory",
        content="private",
        sanitized_content="private",
        share_level=ShareLevel.PRIVILEGED.value,
    ))
    store.add(EvidenceItem(
        evidence_id="risky-web",
        task_id="task-1",
        source_node="web",
        source_type="external_retriever",
        content="risky",
        sanitized_content="risky",
        risk_score=0.9,
        taint_labels=["tool_output_untrusted"],
        share_level=ShareLevel.SANITIZED_FACTS.value,
    ))

    visible_ids = {item.evidence_id for item in store.visible_to("writer")}

    assert visible_ids == {"safe-doc"}


def test_qa_agent_dag_runtime_returns_answer_and_trace():
    agent = build_agent()

    result = agent.answer("SENet 的核心贡献是什么？", modes=(QUERY_MODE_BASIC,), include_memory_manager=False, include_mcp=False)

    assert "SE block" in result.answer
    assert result.metadata["runtime"] == "dag"
    assert result.metadata["dag"]["nodes"]
    assert result.metadata["dag"]["edges"]
    assert result.metadata["document_count"] >= 1


def test_qa_agent_dag_follow_up_restores_previous_sources():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = AppConfig(
            openai_api_key="test-key",
            agent_runtime="dag",
            transcript_dir=temp_dir,
            transcript_resume_enabled=True,
        )
        agent = build_agent(config=config)

        agent.answer(
            "请总结 SENet 这篇论文",
            session_id="demo",
            modes=(QUERY_MODE_BASIC,),
            include_memory_manager=False,
            include_mcp=False,
            transcript_enabled=True,
        )
        result = agent.answer(
            "它的实验效果如何？",
            session_id="demo",
            modes=(QUERY_MODE_BASIC,),
            include_memory_manager=False,
            include_mcp=False,
            transcript_enabled=True,
        )

        assert "SENet.md" in result.metadata["retrieval_diagnostics"].get("source_file_hints", [])
        assert result.metadata["document_count"] >= 1
        assert "[D1]" in result.answer
