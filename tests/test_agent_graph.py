import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import AppConfig
from agent_graph import AgentGraph
from mcp_manager import MCPToolResult
from qa_agent import QAAgent
from rag_query import QUERY_MODE_BASIC, QueryGenerator, RAGQueryEngine
from transcript_store import TranscriptStore
from transcript_store import TranscriptStore


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


class FakeMemoryManager:
    def __init__(self):
        self.retrieve_calls = []
        self.add_calls = []
        self.fact_calls = []
        self.working_memory = SimpleNamespace(
            history=[],
            get_session_history=lambda session_id: self.working_memory.history,
            add_memory=self._add_working_memory,
        )

    def _add_working_memory(self, session_id, role, content, importance=None):
        self.working_memory.history.append({
            "session_id": session_id,
            "role": role,
            "content": content,
            "importance": importance,
        })

    def retrieve_memories(self, query, session_id=None, top_k=3):
        self.retrieve_calls.append((query, session_id, top_k))
        return {"working": [{"role": "user", "content": "用户正在读 SENet。"}]}

    def add_interaction(self, session_id, user_message, assistant_answer):
        self.add_calls.append((session_id, user_message, assistant_answer))
        return []

    def process_high_value_qa(self, question, answer, session_id=None):
        self.fact_calls.append((question, answer, session_id))
        return SimpleNamespace(facts=["SENet 使用 SE block"], semantic_ids=[], graph_written=False, errors=[])


class FakeRuleMCPManager:
    def __init__(self):
        self.calls = []

    def retrieve_context(self, question):
        self.calls.append(question)
        return [{
            "role": "mcp:fetch.fetch",
            "type": "mcp",
            "content": "网页内容：Example Domain",
        }]


class FakeOnlineSearchMCPManager:
    def __init__(self):
        self.calls = []

    def retrieve_online_context(self, question):
        self.calls.append(question)
        return [{
            "role": "mcp:brave-search.brave_web_search",
            "type": "mcp",
            "content": "线上搜索结果：CTR 预测中，非序列特征常用 MLP/DeepFM，序列特征常用 DIN/DIEN/Transformer。",
            "metadata": {"server": "brave-search", "tool": "brave_web_search", "success": True},
        }]


class FakeAgenticMCPManager:
    def __init__(self):
        self.calls = []

    def get_openai_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "mcp__fetch__fetch",
                "description": "Fetch URL.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                },
            },
        }]

    def call_openai_tool(self, function_name, arguments):
        self.calls.append((function_name, arguments))
        return MCPToolResult(
            server="fetch",
            tool="fetch",
            content="工具结果：Example Domain",
            success=True,
        )


class FakeToolPlanningCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools"):
            tool_call = SimpleNamespace(
                id="call-1",
                type="function",
                function=SimpleNamespace(name="mcp__fetch__fetch", arguments='{"url": "https://example.com"}'),
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call]))])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="已结合工具结果回答。[D1]"))]
        )


class FakeToolPlanningLLM:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeToolPlanningCompletions())


class FakePlannerCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if "下一步" in prompt:
            content = '{"action": "call_tools", "tool_requests": [{"server": "fetch"}], "reason": "需要抓取 URL"}'
        else:
            content = "已结合工具结果回答。[D1]"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakePlannerLLM:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakePlannerCompletions())


class AgentGraphTest(unittest.TestCase):
    def build_agent(self, config=None, llm_client=None, memory_manager=None, mcp_manager=None):
        config = config or AppConfig(openai_api_key="test-key")
        llm_client = llm_client or FakeLLMClient()
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
            memory_manager=memory_manager,
            mcp_manager=mcp_manager,
            llm_client=llm_client,
        )

    def test_answer_graph_writes_memory_only_when_enabled(self):
        memory_manager = FakeMemoryManager()
        agent = self.build_agent(memory_manager=memory_manager)

        result = agent.answer("SENet 的核心贡献是什么？", session_id="demo", modes=(QUERY_MODE_BASIC,), write_memory=True)

        self.assertIn("[D1]", result.answer)
        self.assertEqual(len(memory_manager.add_calls), 1)
        self.assertEqual(len(memory_manager.fact_calls), 1)
        self.assertEqual(result.metadata["stop_reason"], "final")

    def test_assistant_transcript_saves_answer_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(openai_api_key="test-key", transcript_dir=temp_dir)
            agent = self.build_agent(config=config)

            agent.answer("SENet 的核心贡献是什么？", session_id="demo", modes=(QUERY_MODE_BASIC,), transcript_enabled=True)

            events = TranscriptStore(root_dir=temp_dir, enabled=True).load("demo")
            assistant_events = [event for event in events if event.get("role") == "assistant"]
            self.assertEqual(len(assistant_events), 1)
            references = assistant_events[0]["metadata"]["references"]
            self.assertIn("sources", references)
            self.assertGreaterEqual(len(references["sources"]), 1)
            self.assertEqual(references["sources"][0]["doc_id"], "D1")

    def test_rule_mcp_tool_result_enters_context(self):
        mcp_manager = FakeRuleMCPManager()
        agent = self.build_agent(mcp_manager=mcp_manager)

        result = agent.answer("请抓取 https://example.com 并回答", modes=(QUERY_MODE_BASIC,))

        self.assertEqual(mcp_manager.calls, ["请抓取 https://example.com 并回答"])
        self.assertEqual(result.metadata["mcp_context_count"], 1)
        self.assertIn("Example Domain", result.built_context.context)

    def test_online_search_fallback_runs_when_local_context_is_irrelevant(self):
        mcp_manager = FakeOnlineSearchMCPManager()
        agent = self.build_agent(mcp_manager=mcp_manager)

        result = agent.answer("CTR 预测里非序列特征和序列特征分别适合什么模型？")

        self.assertTrue(result.metadata["online_search_fallback"])
        self.assertEqual(result.metadata["decisions"][0]["action"], "call_tools")
        self.assertEqual(mcp_manager.calls, ["CTR 预测里非序列特征和序列特征分别适合什么模型？"])
        self.assertEqual(result.metadata["mcp_context_count"], 1)
        self.assertIn("CTR 预测", result.built_context.context)

    def test_general_knowledge_fallback_when_local_context_irrelevant_and_online_unavailable(self):
        agent = self.build_agent()

        result = agent.answer("CTR 预测里非序列特征和序列特征分别适合什么模型？")

        self.assertFalse(result.metadata["online_search_fallback"])
        self.assertTrue(result.metadata["allow_general_knowledge_fallback"])
        self.assertTrue(result.metadata["general_knowledge_fallback_used"])
        self.assertEqual(result.metadata["general_knowledge_context_count"], 1)
        self.assertFalse(result.metadata["local_context_assessment"]["online_available"])
        self.assertIn("local context is insufficient", result.built_context.context)
        self.assertIn("model general knowledge", result.built_context.context)
        self.assertIn("Model general-knowledge draft", result.built_context.context)

    def test_agentic_mcp_runs_as_graph_tool_node(self):
        config = AppConfig(
            openai_api_key="test-key",
            mcp_enabled=True,
            mcp_tool_calling_enabled=True,
        )
        mcp_manager = FakeAgenticMCPManager()
        llm = FakeToolPlanningLLM()
        agent = self.build_agent(config=config, llm_client=llm, mcp_manager=mcp_manager)

        result = agent.answer("请抓取 https://example.com 并回答", modes=(QUERY_MODE_BASIC,))

        self.assertEqual(result.metadata["mcp_tool_call_count"], 1)
        self.assertEqual(mcp_manager.calls[0][0], "mcp__fetch__fetch")
        self.assertIn("工具结果：Example Domain", result.built_context.context)

    def test_llm_planner_can_choose_tool_action(self):
        config = AppConfig(
            openai_api_key="test-key",
            agent_loop_planner_enabled=True,
        )
        mcp_manager = FakeRuleMCPManager()
        llm = FakePlannerLLM()
        agent = self.build_agent(config=config, llm_client=llm, mcp_manager=mcp_manager)

        result = agent.answer("请抓取 https://example.com 并回答", modes=(QUERY_MODE_BASIC,))

        self.assertEqual(result.metadata["decisions"][0]["action"], "call_tools")
        self.assertEqual(mcp_manager.calls, ["请抓取 https://example.com 并回答"])

    def test_tool_observation_is_trimmed_by_budget(self):
        config = AppConfig(
            openai_api_key="test-key",
            agent_loop_max_tool_observation_chars=20,
        )
        mcp_manager = FakeRuleMCPManager()
        mcp_manager.retrieve_context = lambda question: [{
            "role": "mcp:fetch.fetch",
            "type": "mcp",
            "content": "x" * 100,
        }]
        agent = self.build_agent(config=config, mcp_manager=mcp_manager)

        result = agent.answer("请抓取 https://example.com 并回答", modes=(QUERY_MODE_BASIC,))

        observation = result.metadata["tool_observations"][0]
        self.assertLessEqual(len(observation["content"]), 20)
        self.assertTrue(observation["metadata"]["compressed"])

    def test_transcript_resume_adds_recent_events_to_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TranscriptStore(root_dir=temp_dir, enabled=True)
            store.append("demo", "user", "我之前关注 SE block。", ts=1.0)
            config = AppConfig(
                openai_api_key="test-key",
                transcript_enabled=True,
                transcript_resume_enabled=True,
                transcript_dir=temp_dir,
            )
            agent = self.build_agent(config=config)

            result = agent.answer("SENet 的核心贡献是什么？", session_id="demo", modes=(QUERY_MODE_BASIC,))

        self.assertEqual(result.metadata["resumed_transcript_count"], 1)
        self.assertIn("transcript:summary", result.built_context.context)
        self.assertIn("我之前关注 SE block", result.built_context.context)

    def test_transcript_summary_can_fuse_into_working_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TranscriptStore(root_dir=temp_dir, enabled=True)
            store.append("demo", "user", "我之前关注 SE block。", ts=1.0)
            memory_manager = FakeMemoryManager()
            config = AppConfig(
                openai_api_key="test-key",
                transcript_enabled=True,
                transcript_resume_enabled=True,
                transcript_dir=temp_dir,
                session_summary_fuse_memory=True,
            )
            agent = self.build_agent(config=config, memory_manager=memory_manager)

            result = agent.answer("SENet 的核心贡献是什么？", session_id="demo", modes=(QUERY_MODE_BASIC,))

        self.assertTrue(result.metadata["resumed_transcript_count"])
        self.assertEqual(memory_manager.working_memory.history[0]["role"], "transcript_summary")

    def test_planner_parser_accepts_code_fence_and_action_alias(self):
        agent = self.build_agent()
        graph = AgentGraph(agent=agent, config=agent.config)

        data = graph._parse_json_object('```json\n{"action": "tool", "tool_requests": {"server": "fetch"}}\n```')
        action = graph._normalize_planner_action(data["action"])
        tool_requests = graph._normalize_tool_requests(data["tool_requests"])

        self.assertEqual(action, "call_tools")
        self.assertEqual(tool_requests[0]["server"], "fetch")


if __name__ == "__main__":
    unittest.main()
