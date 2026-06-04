import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.qa_agent import QAAgent
from config import AppConfig
from rag.rag_query import QUERY_MODE_BASIC, QUERY_MODE_HYDE, QUERY_MODE_MQE
from tools.mcp_manager import MCPToolResult


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if "Rewrite the same retrieval question" in prompt or "改写" in prompt:
            content = '["SENet 的创新点是什么？", "SE block 如何建模通道依赖？"]'
        elif "hypothetical answer paragraph" in prompt or "假设性答案段落" in prompt:
            content = "SENet 通过 SE block 对通道特征进行自适应重标定。"
        elif "Normalize the user question" in prompt:
            if "https://example.com" in prompt:
                content = '{"english_question":"Fetch https://example.com and answer the question.","response_language":"Chinese"}'
            else:
                content = '{"english_question":"What is the core contribution of SENet?","response_language":"Chinese"}'
        else:
            content = "SENet 的核心贡献是通过 SE block 显式建模通道依赖，并进行特征重标定。[D1]"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class FakeTruncatedCompletions:
    def __init__(self):
        self.calls = []
        self.answer_calls = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if "Normalize the user question" in prompt:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"english_question":"SENet authors","response_language":"Chinese"}'),
                        finish_reason="stop",
                    )
                ]
            )
        self.answer_calls += 1
        if self.answer_calls == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="第一段回答还没有结束"),
                        finish_reason="length",
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="第二段从中断处继续，并完整收尾。[D1]"),
                    finish_reason="stop",
                )
            ]
        )


class FakeTruncatedLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeTruncatedCompletions())


@dataclass
class FakeHit:
    id: str
    score: float
    payload: dict


class FakeQdrant:
    def __init__(self):
        self.search_calls = []

    def search(self, collection_name, query_vector, limit):
        self.search_calls.append((collection_name, query_vector, limit))
        call_index = len(self.search_calls)
        return [
            FakeHit(
                id=f"hit-{call_index}-1",
                score=0.8,
                payload={
                    "page_content": "SENet 使用 SE block 显式建模通道依赖，并重标定通道特征。",
                    "content_hash": "senet-main",
                    "source_file": "SENet.md",
                    "chunk_index": 1,
                    "heading_paths": ["研究核心"],
                },
            ),
            FakeHit(
                id=f"hit-{call_index}-2",
                score=0.5,
                payload={
                    "page_content": "SE block 包括 squeeze 和 excitation 两个阶段。",
                    "content_hash": f"senet-extra-{call_index}",
                    "source_file": "SENet.md",
                    "chunk_index": call_index + 1,
                    "heading_paths": ["方法"],
                },
            ),
        ]


class FakeEmptyQdrant:
    def __init__(self):
        self.search_calls = []

    def search(self, collection_name, query_vector, limit):
        self.search_calls.append((collection_name, query_vector, limit))
        return []


class FakeMemoryManager:
    def __init__(self):
        self.calls = []

    def retrieve_memories(self, query, session_id=None, top_k=3):
        self.calls.append((query, session_id, top_k))
        return {
            "working": [{"role": "user", "content": "用户正在阅读 SENet 论文。", "score": 1.0}],
            "episodic": [{"content": "用户之前关注 SE block 的通道依赖建模。", "score": 0.8}],
            "semantic": [{"content": "项目决定采用 Qdrant 存储语义向量。", "score": 0.7}],
            "sensory": [{"payload": {"caption": "一张智能问答系统思维导图"}, "score": 0.6}],
        }


class FakeMCPManager:
    def __init__(self):
        self.calls = []

    def retrieve_context(self, question):
        self.calls.append(question)
        return [{
            "role": "mcp:fetch.fetch",
            "type": "mcp",
            "content": "MCP 抓取到的网页内容：SENet 是一种通道注意力网络。",
        }]


class FakeAgenticMCPManager:
    def __init__(self):
        self.calls = []

    def get_openai_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "mcp__fetch__fetch",
                "description": "Fetch URL content.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        }]

    def call_openai_tool(self, function_name, arguments):
        self.calls.append((function_name, arguments))
        return MCPToolResult(
            server="fetch",
            tool="fetch",
            content="工具结果：Example Domain 页面内容。",
            success=True,
        )


class FakeSkillManager:
    def __init__(self):
        self.calls = []

    def retrieve_context(self, question):
        self.calls.append(question)
        return [{
            "role": "skill:senet_explainer",
            "type": "skill",
            "content": "Skill 输出：SENet 回答需要包含 squeeze、excitation 和通道重标定。",
        }]


class FakeToolCallingCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and not any(message.get("role") == "tool" for message in kwargs["messages"]):
            tool_call = SimpleNamespace(
                id="call-1",
                type="function",
                function=SimpleNamespace(
                    name="mcp__fetch__fetch",
                    arguments='{"url": "https://www.example.com"}',
                ),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call]))]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="已根据 MCP fetch 结果回答：Example Domain。[D1]"))]
        )


class FakeToolCallingLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeToolCallingCompletions())


class FakeGeneralKnowledgeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system_prompt = kwargs["messages"][0]["content"]
        user_prompt = kwargs["messages"][-1]["content"]
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "本地文档/记忆中没有找到足够依据。基于模型通用知识，"
                            "CTR 中非序列特征通常优先考虑 Wide&Deep、DeepFM 或 DCN；"
                            "序列特征通常优先考虑 DIN、DIEN、Transformer 或 SIM。"
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        )


class FakeGeneralKnowledgeLLMClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeGeneralKnowledgeCompletions())


class QAAgentTest(unittest.TestCase):
    def build_agent(self, llm_client=None):
        from rag.rag_query import QueryGenerator, RAGQueryEngine

        config = AppConfig(openai_api_key=None, mcp_enabled=False, mcp_tool_calling_enabled=False)
        fake_llm = llm_client or FakeLLMClient()
        query_engine = RAGQueryEngine(
            qdrant_client=FakeQdrant(),
            collection_name="semantic_test",
            config=config,
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(llm_client=fake_llm),
        )
        return QAAgent(query_engine=query_engine, config=config, llm_client=fake_llm)

    def test_understand_query_routes_simple_fact_to_basic(self):
        agent = self.build_agent()
        understanding = agent.understand_query("SENet作者")

        self.assertEqual(understanding.query_type, "simple_fact")
        self.assertEqual(tuple(understanding.modes), (QUERY_MODE_BASIC,))

    def test_understand_query_routes_explanatory_to_all_modes(self):
        agent = self.build_agent()
        understanding = agent.understand_query("SENet 的核心贡献是什么？")

        self.assertEqual(understanding.query_type, "explanatory")
        self.assertIn(QUERY_MODE_BASIC, understanding.modes)
        self.assertIn(QUERY_MODE_MQE, understanding.modes)
        self.assertIn(QUERY_MODE_HYDE, understanding.modes)

    def test_understand_query_routes_complex_design_to_hyde(self):
        agent = self.build_agent()
        understanding = agent.understand_query("我想做一个根据输入 prompt 推荐阅读文献的 agent，具体该怎么设计？")

        self.assertEqual(understanding.query_type, "complex")
        self.assertIn(QUERY_MODE_BASIC, understanding.modes)
        self.assertIn(QUERY_MODE_MQE, understanding.modes)
        self.assertIn(QUERY_MODE_HYDE, understanding.modes)

    def test_answer_runs_retrieve_rerank_context_and_llm(self):
        llm = FakeLLMClient()
        agent = self.build_agent(llm_client=llm)

        result = agent.answer(
            "SENet 的核心贡献是什么？",
            memories=[{"role": "user", "content": "用户正在阅读 SENet 论文。"}],
            top_k=6,
            per_query_limit=2,
            rerank_top_k=3,
            context_top_k=2,
        )

        self.assertIn("SE block", result.answer)
        self.assertIn("[D1]", result.answer)
        self.assertGreaterEqual(result.metadata["candidate_count"], 2)
        self.assertEqual(result.metadata["document_count"], 2)
        self.assertIn("[M1] user", result.built_context.context)
        self.assertIn("Retrieved documents", result.built_context.context)

    def test_answer_complex_query_uses_hyde_and_dynamic_retrieval_budget(self):
        llm = FakeLLMClient()
        agent = self.build_agent(llm_client=llm)

        result = agent.answer("我想做一个根据输入 prompt 推荐阅读文献的 agent，具体该怎么设计？")

        self.assertEqual(result.metadata["query_type"], "complex")
        self.assertIn(QUERY_MODE_HYDE, result.metadata["active_modes"])
        self.assertEqual(result.metadata["retrieval_budget_reason"], "complex_dynamic")
        self.assertEqual(result.metadata["effective_query_top_k"], agent.config.rag_complex_query_top_k)
        self.assertEqual(result.metadata["effective_context_top_k"], agent.config.rag_complex_context_top_k)
        self.assertGreater(result.metadata["document_count"], 2)

    def test_answer_continues_when_llm_finish_reason_is_length(self):
        llm = FakeTruncatedLLMClient()
        agent = self.build_agent(llm_client=llm)

        result = agent.answer("SENet作者")

        self.assertIn("第一段回答还没有结束", result.answer)
        self.assertIn("第二段从中断处继续", result.answer)
        self.assertEqual(result.metadata["answer_generation"]["finish_reasons"], ["length", "stop"])
        self.assertEqual(result.metadata["answer_generation"]["continuation_count"], 1)
        self.assertFalse(result.metadata["answer_generation"]["truncated"])

    def test_answer_automatically_includes_memory_manager_results(self):
        llm = FakeLLMClient()
        memory_manager = FakeMemoryManager()
        agent = self.build_agent(llm_client=llm)
        agent.memory_manager = memory_manager

        result = agent.answer(
            "SENet 的核心贡献是什么？",
            session_id="session-1",
            modes=(QUERY_MODE_BASIC,),
            top_k=3,
            per_query_limit=2,
            context_top_k=2,
        )

        self.assertEqual(memory_manager.calls[0][1], "session-1")
        self.assertEqual(result.metadata["memory_count"], 4)
        self.assertIn("用户之前关注 SE block", result.built_context.context)
        self.assertIn("一张智能问答系统思维导图", result.built_context.context)

    def test_polluted_memory_is_not_included_in_context(self):
        agent = self.build_agent()
        flattened = agent._flatten_retrieved_memories({
            "semantic": [
                {"content": "未配置 OPENAI_API_KEY，以下是基于检索上下文的摘要式回答：..."},
                {"content": "用户正在研究 SENet 的通道注意力。"},
            ]
        })

        self.assertEqual(len(flattened), 1)
        self.assertIn("SENet", flattened[0]["content"])

    def test_answer_includes_mcp_context_when_configured(self):
        llm = FakeLLMClient()
        mcp_manager = FakeMCPManager()
        agent = self.build_agent(llm_client=llm)
        agent.mcp_manager = mcp_manager

        result = agent.answer(
            "请抓取 https://example.com 并回答",
            modes=(QUERY_MODE_BASIC,),
            top_k=3,
            per_query_limit=2,
            context_top_k=2,
        )

        self.assertEqual(mcp_manager.calls[0], "Fetch https://example.com and answer the question.")
        self.assertEqual(result.metadata["mcp_context_count"], 1)
        self.assertIn("MCP 抓取到的网页内容", result.built_context.context)

    def test_answer_includes_skill_context_when_configured(self):
        llm = FakeLLMClient()
        skill_manager = FakeSkillManager()
        agent = self.build_agent(llm_client=llm)
        agent.skill_manager = skill_manager

        result = agent.answer(
            "SENet 的核心贡献是什么？",
            modes=(QUERY_MODE_BASIC,),
            top_k=3,
            per_query_limit=2,
            context_top_k=2,
        )

        self.assertEqual(skill_manager.calls[0], "What is the core contribution of SENet?")
        self.assertEqual(result.metadata["skill_context_count"], 1)
        self.assertIn("Skill 输出", result.built_context.context)

    def test_answer_can_use_agentic_mcp_tool_calling(self):
        from rag.rag_query import QueryGenerator, RAGQueryEngine

        config = AppConfig(
            openai_api_key="test-key",
            mcp_enabled=True,
            mcp_tool_calling_enabled=True,
        )
        llm = FakeToolCallingLLMClient()
        mcp_manager = FakeAgenticMCPManager()
        query_engine = RAGQueryEngine(
            qdrant_client=FakeQdrant(),
            collection_name="semantic_test",
            config=config,
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(llm_client=FakeLLMClient()),
        )
        agent = QAAgent(
            query_engine=query_engine,
            config=config,
            mcp_manager=mcp_manager,
            llm_client=llm,
        )

        result = agent.answer(
            "请抓取 https://www.example.com 并回答",
            modes=(QUERY_MODE_BASIC,),
            top_k=3,
            per_query_limit=2,
            context_top_k=2,
        )

        self.assertIn("Example Domain", result.answer)
        self.assertEqual(result.metadata["mcp_tool_call_count"], 1)
        self.assertEqual(mcp_manager.calls[0][0], "mcp__fetch__fetch")
        self.assertEqual(result.metadata["mcp_context_count"], 0)

    def test_answer_fallback_without_llm_client_or_key(self):
        from rag.rag_query import QueryGenerator, RAGQueryEngine

        config = AppConfig(openai_api_key=None, openai_base_url=None)
        query_engine = RAGQueryEngine(
            qdrant_client=FakeQdrant(),
            collection_name="semantic_test",
            config=config,
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(llm_client=None),
        )
        agent = QAAgent(query_engine=query_engine, config=config, llm_client=None)
        result = agent.answer("SENet 的核心贡献是什么？", modes=(QUERY_MODE_BASIC,))

        self.assertIn("未配置 OPENAI_API_KEY", result.answer)
        self.assertIn("[D1]", result.answer)

    def test_general_knowledge_fallback_with_no_documents_still_calls_llm(self):
        from rag.rag_query import QueryGenerator, RAGQueryEngine

        config = AppConfig(openai_api_key="test-key", agent_loop_online_fallback_enabled=False)
        llm = FakeGeneralKnowledgeLLMClient()
        query_engine = RAGQueryEngine(
            qdrant_client=FakeEmptyQdrant(),
            collection_name="semantic_test",
            config=config,
            embedding_service=FakeEmbeddingService(),
            query_generator=QueryGenerator(llm_client=llm),
        )
        agent = QAAgent(query_engine=query_engine, config=config, llm_client=llm)

        result = agent.answer("CTR 预测里非序列特征和序列特征分别适合什么模型？", modes=(QUERY_MODE_BASIC,))

        self.assertIn("基于模型通用知识", result.answer)
        self.assertTrue(result.metadata["allow_general_knowledge_fallback"])
        self.assertTrue(result.metadata["general_knowledge_fallback_used"])
        self.assertEqual(result.metadata["general_knowledge_context_count"], 1)
        self.assertEqual(result.metadata["document_count"], 0)
        self.assertIn("using your model general knowledge", llm.chat.completions.last_system_prompt)


if __name__ == "__main__":
    unittest.main()
