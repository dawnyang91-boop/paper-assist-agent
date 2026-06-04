from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from config import AppConfig, get_config
from content_filters import is_polluted_context
from context_builder import BuiltContext, ContextBuilder
from rag_query import QUERY_MODE_BASIC, QUERY_MODE_HYDE, QUERY_MODE_MQE, RAGQueryEngine, RetrievedChunk
from reranker import CandidateReranker, RankedChunk
from sentinel.defenses.security_manager import SecurityManager


@dataclass
class QueryUnderstanding:
    question: str
    query_type: str
    modes: Sequence[str]
    reason: str


@dataclass
class QAResult:
    question: str
    answer: str
    query_understanding: QueryUnderstanding
    candidates: List[RetrievedChunk]
    ranked_chunks: List[RankedChunk]
    built_context: BuiltContext
    metadata: Dict[str, Any] = field(default_factory=dict)


class QAAgent:
    """End-to-end RAG QA agent: understand query, retrieve, rerank, build context, answer."""

    def __init__(
        self,
        qdrant_client: Any = None,
        config: Optional[AppConfig] = None,
        query_engine: Optional[RAGQueryEngine] = None,
        reranker: Optional[CandidateReranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        memory_manager: Any = None,
        mcp_manager: Any = None,
        skill_manager: Any = None,
        llm_client: Any = None,
        security_manager: Any = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client
        self.query_engine = query_engine or self._build_query_engine(qdrant_client)
        self.reranker = reranker or CandidateReranker(config=self.config)
        self.context_builder = context_builder or ContextBuilder(config=self.config)
        self.memory_manager = memory_manager
        self.mcp_manager = mcp_manager
        self.skill_manager = skill_manager
        self.security_manager = security_manager or SecurityManager.from_app_config(self.config)
        self._last_answer_generation_metadata: Dict[str, Any] = {}

    def understand_query(self, question: str) -> QueryUnderstanding:
        clean_question = question.strip()
        if not clean_question:
            return QueryUnderstanding(
                question=clean_question,
                query_type="empty",
                modes=(QUERY_MODE_BASIC,),
                reason="空问题只保留基础查询模式。",
            )

        complex_markers = [
            "为什么", "如何", "怎么", "怎样", "分析", "比较", "区别", "关系", "影响",
            "总结", "归纳", "解释", "评价", "优缺点", "流程", "原理", "机制", "贡献",
            "设计", "方案", "建议", "推荐", "构建", "实现", "规划", "复杂", "详细",
        ]
        hyde_markers = ["解释", "总结", "原理", "机制", "贡献", "方法", "流程", "优缺点"]

        if any(marker in clean_question for marker in hyde_markers):
            return QueryUnderstanding(
                question=clean_question,
                query_type="explanatory",
                modes=(QUERY_MODE_BASIC, QUERY_MODE_MQE, QUERY_MODE_HYDE),
                reason="问题偏解释/总结/机制类，启用 Basic + MQE + HyDE 提升语义覆盖。",
            )

        if len(clean_question) <= 20 and not any(marker in clean_question for marker in complex_markers):
            return QueryUnderstanding(
                question=clean_question,
                query_type="simple_fact",
                modes=(QUERY_MODE_BASIC,),
                reason="问题较短且偏事实查询，优先使用基础搜索降低噪声。",
            )

        return QueryUnderstanding(
            question=clean_question,
            query_type="complex",
            modes=(QUERY_MODE_BASIC, QUERY_MODE_MQE, QUERY_MODE_HYDE),
            reason="问题可能涉及多个表达角度或需要综合组织，启用 Basic + MQE + HyDE。",
        )

    def normalize_question_to_english(self, question: str) -> Dict[str, str]:
        """Normalize user input into English for retrieval while forcing Chinese final answers."""
        original = (question or "").strip()
        if not original:
            return {
                "original_question": original,
                "english_question": original,
                "response_language": "Chinese",
            }
        client = self._get_llm_client()
        if client is None:
            return {
                "original_question": original,
                "english_question": original,
                "response_language": "Chinese",
            }

        prompt = f"""Normalize the user question for an English-first RAG and memory system.
Return only valid JSON with these keys:
- "english_question": a faithful, natural English retrieval question. Preserve paper titles, model names, IDs, code symbols, equations, URLs, and technical terms.
- "response_language": always use "Chinese" for this assistant.

Original user question:
{original}"""
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise translation and query-normalization service. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=400,
                **self.config.chat_completion_kwargs(),
            )
            data = _parse_json_object(response.choices[0].message.content or "")
            english_question = str(data.get("english_question") or "").strip() or original
            return {
                "original_question": original,
                "english_question": english_question,
                "response_language": "Chinese",
            }
        except Exception:
            return {
                "original_question": original,
                "english_question": original,
                "response_language": "Chinese",
            }

    def answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        memories: Optional[Iterable[Dict[str, Any]]] = None,
        include_memory_manager: bool = True,
        include_mcp: bool = True,
        include_skills: bool = True,
        memory_top_k: int = 3,
        modes: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        per_query_limit: Optional[int] = None,
        mqe_count: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        context_top_k: Optional[int] = None,
        write_memory: bool = False,
        transcript_enabled: Optional[bool] = None,
        transcript_store_override: Optional[Any] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> QAResult:
        from agent_graph import AgentGraph
        from transcript_store import TranscriptStore

        transcript_store = transcript_store_override
        if transcript_store is None and transcript_enabled is not None:
            transcript_store = TranscriptStore(
                root_dir=self.config.transcript_dir,
                enabled=transcript_enabled,
            )
        graph = AgentGraph(agent=self, config=self.config, transcript_store=transcript_store)
        if self.config.agent_runtime == "langgraph":
            state = self._run_langgraph(
                graph=graph,
                question=question,
                session_id=session_id,
                memories=memories,
                include_memory_manager=include_memory_manager,
                include_mcp=include_mcp,
                include_skills=include_skills,
                memory_top_k=memory_top_k,
                modes=modes,
                top_k=top_k,
                per_query_limit=per_query_limit,
                mqe_count=mqe_count,
                rerank_top_k=rerank_top_k,
                context_top_k=context_top_k,
                write_memory=write_memory,
                stream_callback=stream_callback,
            )
        else:
            state = graph.run(
                question=question,
                session_id=session_id,
                memories=memories,
                include_memory_manager=include_memory_manager,
                include_mcp=include_mcp,
                include_skills=include_skills,
                memory_top_k=memory_top_k,
                modes=modes,
                top_k=top_k,
                per_query_limit=per_query_limit,
                mqe_count=mqe_count,
                rerank_top_k=rerank_top_k,
                context_top_k=context_top_k,
                write_memory=write_memory,
                stream_callback=stream_callback,
            )
        understanding = state.usage.get("query_understanding") or self.understand_query(question)
        active_modes = tuple(state.query_plan.retrieval_modes if state.query_plan else (modes or understanding.modes))
        built_context = state.built_context
        if built_context is None:
            built_context = self.context_builder.build_context(
                question=question,
                ranked_chunks=[],
                memories=[],
            )
        merged_memories = state.usage.get("context_memories", [])
        mcp_tool_calls = [
            {
                "function": item.get("metadata", {}).get("function"),
                "server": item.get("metadata", {}).get("server"),
                "tool": item.get("metadata", {}).get("tool"),
                "success": item.get("metadata", {}).get("success"),
                "error": item.get("metadata", {}).get("error"),
            }
            for item in state.tool_observations
            if item.get("type") == "mcp_tool"
        ]
        self._last_mcp_tool_calls = mcp_tool_calls
        general_knowledge_contexts = [
            item for item in merged_memories if item.get("type") == "general_knowledge"
        ]
        counted_memories = [
            item for item in merged_memories if item.get("type") != "general_knowledge"
        ]
        return QAResult(
            question=question,
            answer=state.answer,
            query_understanding=understanding,
            candidates=state.candidates,
            ranked_chunks=state.ranked_chunks,
            built_context=built_context,
            metadata={
                "active_modes": list(active_modes),
                "query_type": understanding.query_type,
                "retrieval_budget_reason": state.usage.get("retrieval_budget_reason"),
                "effective_query_top_k": state.usage.get("effective_query_top_k"),
                "effective_per_query_limit": state.usage.get("effective_per_query_limit"),
                "effective_rerank_top_k": state.usage.get("effective_rerank_top_k"),
                "effective_context_top_k": state.usage.get("effective_context_top_k"),
                "candidate_count": len(state.candidates),
                "ranked_count": len(state.ranked_chunks),
                "document_count": len(built_context.documents),
                "memory_count": len(counted_memories),
                "general_knowledge_context_count": len(general_knowledge_contexts),
                "memory_retrieval_diagnostics": state.usage.get("memory_retrieval_diagnostics", {}),
                "local_context_assessment": state.usage.get("local_context_assessment", {}),
                "online_search_fallback": state.usage.get("online_search_fallback", False),
                "allow_general_knowledge_fallback": state.usage.get("allow_general_knowledge_fallback", True),
                "general_knowledge_fallback_used": state.usage.get("general_knowledge_fallback_used", False),
                "general_knowledge_draft_chars": state.usage.get("general_knowledge_draft_chars", 0),
                "mcp_context_count": len([item for item in merged_memories if item.get("type") == "mcp"]),
                "mcp_tool_call_count": len(mcp_tool_calls),
                "mcp_tool_calls": mcp_tool_calls,
                "skill_context_count": len([item for item in merged_memories if item.get("type") == "skill"]),
                "context_tokens": built_context.token_estimate,
                "answer_generation": state.usage.get("answer_generation", {}),
                "sentinel": state.usage.get("sentinel", {}),
                "loop_steps": state.usage.get("loop_steps", state.loop_step),
                "stop_reason": state.stop_reason,
                "errors": state.errors,
                "warnings": state.warnings,
                "verification": state.verification,
                "decisions": state.decisions,
                "resumed_transcript_count": len(state.resumed_transcript),
                "last_checkpoint_id": state.usage.get("last_checkpoint_id"),
                "checkpoint_enabled": state.usage.get("checkpoint_enabled"),
                "tool_observations": state.tool_observations,
                "memory_contexts": merged_memories,
                "writeback_events": state.writeback_events,
            },
        )

    def _run_langgraph(self, graph: Any, **payload: Any):
        try:
            from langgraph_adapter import LangGraphAgentRunner

            runner = LangGraphAgentRunner(agent_graph=graph)
            result = runner.invoke(payload)
            state = result.get("_agent_state")
            if state is None:
                raise RuntimeError("LangGraph runner 未返回 AgentState。")
            return state
        except Exception as exc:
            print(f"[警告] LangGraph 运行失败，已回退到 legacy agent graph：{exc}")
            return graph.run(**payload)

    def _build_memory_context(
        self,
        question: str,
        session_id: Optional[str],
        explicit_memories: Optional[Iterable[Dict[str, Any]]],
        include_memory_manager: bool,
        include_mcp: bool,
        include_skills: bool,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        memories = list(explicit_memories or [])
        if include_skills and self.skill_manager is not None:
            try:
                memories.extend(self.skill_manager.retrieve_context(question))
            except Exception as exc:
                memories.append({
                    "role": "skill_error",
                    "type": "skill",
                    "content": f"Skill 执行失败：{exc}",
                })

        if include_memory_manager and self.memory_manager is not None:
            try:
                retrieved = self.memory_manager.retrieve_memories(
                    query=question,
                    session_id=session_id,
                    top_k=top_k,
                )
                memories.extend(self._flatten_retrieved_memories(retrieved))
            except Exception as exc:
                memories.append({
                    "role": "memory_error",
                    "content": f"记忆检索失败：{exc}",
                })

        if include_mcp and self.mcp_manager is not None:
            try:
                memories.extend(self.mcp_manager.retrieve_context(question))
            except Exception as exc:
                memories.append({
                    "role": "mcp_error",
                    "type": "mcp",
                    "content": f"MCP 上下文获取失败：{exc}",
                })
        return memories

    def _flatten_retrieved_memories(self, retrieved: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        flattened = []
        for memory_type, items in retrieved.items():
            if memory_type.startswith("_"):
                continue
            for item in items:
                content = item.get("content")
                if not content and isinstance(item.get("payload"), dict):
                    payload = item["payload"]
                    content = payload.get("caption") or payload.get("content") or payload.get("page_content")
                if not content:
                    continue
                if is_polluted_context(content):
                    continue
                role = item.get("role") or item.get("type") or memory_type
                flattened.append({
                    "role": role,
                    "type": memory_type,
                    "content": content,
                    "score": item.get("score"),
                    "importance": item.get("importance"),
                    "metadata": item,
                })
        return flattened

    def generate_answer(
        self,
        built_context: BuiltContext,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        self._last_answer_generation_metadata = {}
        client = self._get_llm_client()
        if client is None:
            self._last_answer_generation_metadata = {"fallback": True, "reason": "missing_llm_client_or_key"}
            return self._fallback_answer(built_context)
        if self._agentic_mcp_enabled():
            return self._generate_answer_with_mcp_tools(client, built_context, stream_callback=stream_callback)
        return self._generate_answer_without_tools(client, built_context, stream_callback=stream_callback)

    def generate_general_knowledge_draft(self, question: str) -> Optional[str]:
        """Generate a HyDE-style draft answer from the base LLM when local context is insufficient."""
        client = self._get_llm_client()
        if client is None:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You create a general-knowledge draft for a private-domain QA assistant. "
                    "First silently restate the user question as a clear professional question, then answer from model general knowledge. "
                    "Do not cite local document IDs, do not pretend the answer came from the private knowledge base, and state assumptions when needed. "
                    "Write the draft in Chinese."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Local documents and memories do not provide enough evidence. "
                    "Answer the following question directly from general knowledge. Be specific, practical, and write in Chinese:\n"
                    f"{question}"
                ),
            },
        ]
        content, _ = self._create_chat_completion_content(
            client,
            {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.qa_temperature,
                "max_tokens": min(self.config.qa_max_answer_tokens, 1200),
                **self.config.chat_completion_kwargs(),
            },
            stream=True,
        )
        return content.strip() or None

    def _system_prompt_for_context(self, built_context: BuiltContext, allow_tools: bool = False) -> str:
        allow_general = bool(getattr(built_context, "allow_general_knowledge_fallback", False))
        if allow_general:
            prompt = (
                "You are a rigorous private-domain QA assistant. Prefer local documents, memories, skills, and tool context. "
                "If those contexts are insufficient, explicitly state in Chinese: 本地文档/记忆中没有找到足够依据。 "
                "Then give a direct, complete, and practical answer using your model general knowledge. "
                "Do not attach [D] or [M] citations to general-knowledge claims. "
                "For complex questions, start with the conclusion, then explain rationale, applicable scenarios, and caveats. "
                "The final answer must be written in Chinese."
            )
        else:
            prompt = (
                "You are a rigorous private-domain QA assistant. Answer from the provided context. "
                "If the context is insufficient, say so clearly in Chinese. Cite local documents with IDs such as [D1] when used. "
                "For complex questions, synthesize multiple pieces of evidence and provide a structured, sufficiently detailed answer. "
                "The final answer must be written in Chinese."
            )
        if allow_tools:
            prompt += (
                "If extra file reading, web fetching, or SQLite querying is needed, call the available MCP tools. "
                "Tool results are auxiliary context; the final answer still needs clear grounding."
            )
        return prompt

    def _generate_answer_with_mcp_tools(
        self,
        client: Any,
        built_context: BuiltContext,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        self._last_mcp_tool_calls = []
        tools = self.mcp_manager.get_openai_tools()
        if not tools:
            return self._generate_answer_without_tools(client, built_context, stream_callback=stream_callback)

        messages = [
            {
                "role": "system",
                "content": self._system_prompt_for_context(built_context, allow_tools=True),
            },
            {"role": "user", "content": built_context.context},
        ]

        for _ in range(max(1, self.config.mcp_tool_calling_max_iterations)):
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config.qa_temperature,
                max_tokens=self.config.qa_max_answer_tokens,
                **self.config.chat_completion_kwargs(),
            )
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                content = (getattr(message, "content", "") or "").strip()
                if content and stream_callback:
                    stream_callback(content)
                finish_reason = self._choice_finish_reason(response)
                return self._continue_if_truncated(
                    client=client,
                    messages=messages,
                    content=content,
                    finish_reason=finish_reason,
                    request_kwargs={
                        "model": self.config.model_name,
                        "temperature": self.config.qa_temperature,
                        "max_tokens": self.config.qa_max_answer_tokens,
                        **self.config.chat_completion_kwargs(),
                    },
                    stream_callback=stream_callback,
                )

            messages.append(self._assistant_tool_call_message(message, tool_calls))
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                function_name = getattr(function, "name", "")
                arguments = getattr(function, "arguments", "{}")
                result = self.mcp_manager.call_openai_tool(function_name, arguments)
                observation = result.content if result.success else f"MCP 工具调用失败：{result.error or result.content}"
                self._last_mcp_tool_calls.append({
                    "function": function_name,
                    "server": result.server,
                    "tool": result.tool,
                    "success": result.success,
                    "error": result.error,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, "id", ""),
                    "content": observation,
                })

        messages.append({
            "role": "user",
            "content": "The tool-call iteration limit has been reached. Give the final answer in Chinese using the available context and tool observations.",
        })
        request_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.qa_temperature,
            "max_tokens": self.config.qa_max_answer_tokens,
            **self.config.chat_completion_kwargs(),
        }
        content, finish_reason = self._create_chat_completion_content(
            client,
            request_kwargs,
            stream=True,
            stream_callback=stream_callback,
        )
        content = content.strip()
        return self._continue_if_truncated(
            client=client,
            messages=messages,
            content=content,
            finish_reason=finish_reason,
            request_kwargs={key: value for key, value in request_kwargs.items() if key != "messages"},
            stream_callback=stream_callback,
        )

    def _generate_answer_without_tools(
        self,
        client: Any,
        built_context: BuiltContext,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self._system_prompt_for_context(built_context),
            },
            {"role": "user", "content": built_context.context},
        ]
        request_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.qa_temperature,
            "max_tokens": self.config.qa_max_answer_tokens,
            **self.config.chat_completion_kwargs(),
        }
        content, finish_reason = self._create_chat_completion_content(
            client,
            request_kwargs,
            stream=True,
            stream_callback=stream_callback,
        )
        content = content.strip()
        return self._continue_if_truncated(
            client=client,
            messages=messages,
            content=content,
            finish_reason=finish_reason,
            request_kwargs={key: value for key, value in request_kwargs.items() if key != "messages"},
            stream_callback=stream_callback,
        )

    def _create_chat_completion_content(
        self,
        client: Any,
        request_kwargs: Dict[str, Any],
        stream: bool = True,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, Optional[str]]:
        if not stream:
            response = client.chat.completions.create(**request_kwargs)
            content = getattr(response.choices[0].message, "content", "") or ""
            if content and stream_callback:
                stream_callback(content)
            return content, self._choice_finish_reason(response)

        try:
            response = client.chat.completions.create(**request_kwargs, stream=True)
        except TypeError:
            response = client.chat.completions.create(**request_kwargs)

        if hasattr(response, "choices"):
            content = getattr(response.choices[0].message, "content", "") or ""
            if content and stream_callback:
                stream_callback(content)
            return content, self._choice_finish_reason(response)

        pieces: List[str] = []
        finish_reason: Optional[str] = None
        for chunk in response:
            choices = getattr(chunk, "choices", []) or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                if stream_callback:
                    stream_callback(content)
                pieces.append(content)
        return "".join(pieces), finish_reason

    def _choice_finish_reason(self, response: Any) -> Optional[str]:
        choices = getattr(response, "choices", []) or []
        if not choices:
            return None
        return getattr(choices[0], "finish_reason", None)

    def _continue_if_truncated(
        self,
        client: Any,
        messages: List[Dict[str, Any]],
        content: str,
        finish_reason: Optional[str],
        request_kwargs: Dict[str, Any],
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        pieces = [content] if content else []
        finish_reasons = [finish_reason]
        continuation_count = 0
        current_finish_reason = finish_reason
        max_continuations = max(0, self.config.qa_max_continuations)

        while current_finish_reason == "length" and continuation_count < max_continuations:
            continuation_count += 1
            continuation_messages = [
                *messages,
                {"role": "assistant", "content": "\n".join(pieces)},
                {
                    "role": "user",
                    "content": (
                        "The previous answer was interrupted by the output token limit. Continue from the interruption point without repeating earlier content. "
                        "Keep the same Markdown structure, preserve existing [D1]/[D2] citation IDs, and write in Chinese."
                    ),
                },
            ]
            next_content, current_finish_reason = self._create_chat_completion_content(
                client,
                {**request_kwargs, "messages": continuation_messages},
                stream=True,
                stream_callback=stream_callback,
            )
            next_content = next_content.strip()
            if next_content:
                pieces.append(next_content)
            finish_reasons.append(current_finish_reason)

        truncated = current_finish_reason == "length"
        self._last_answer_generation_metadata = {
            "finish_reasons": finish_reasons,
            "continuation_count": continuation_count,
            "truncated": truncated,
            "max_tokens": self.config.qa_max_answer_tokens,
            "max_continuations": max_continuations,
        }
        answer = "\n\n".join(piece for piece in pieces if piece)
        if truncated:
            answer += (
                "\n\n（回答仍可能因模型输出长度限制被截断。可以调大 "
                "`QA_MAX_ANSWER_TOKENS` 或 `QA_MAX_CONTINUATIONS` 后重试。）"
            )
        return answer.strip()

    def _assistant_tool_call_message(self, message: Any, tool_calls: List[Any]) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [
                {
                    "id": getattr(tool_call, "id", ""),
                    "type": getattr(tool_call, "type", "function"),
                    "function": {
                        "name": getattr(getattr(tool_call, "function", None), "name", ""),
                        "arguments": getattr(getattr(tool_call, "function", None), "arguments", "{}"),
                    },
                }
                for tool_call in tool_calls
            ],
        }

    def _agentic_mcp_enabled(self) -> bool:
        return bool(
            self.mcp_manager is not None
            and self.config.mcp_enabled
            and self.config.mcp_tool_calling_enabled
        )

    def _build_query_engine(self, qdrant_client: Any) -> RAGQueryEngine:
        if qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ValueError("未传入 qdrant_client，且未安装 qdrant-client。") from exc

            qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.qdrant_timeout,
            )
        return RAGQueryEngine(qdrant_client=qdrant_client, config=self.config)

    def _get_llm_client(self):
        if self.llm_client is not None:
            return self.llm_client
        if not self.config.openai_api_key:
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        self.llm_client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )
        return self.llm_client

    def _fallback_answer(self, built_context: BuiltContext) -> str:
        if not built_context.documents:
            if getattr(built_context, "allow_general_knowledge_fallback", False):
                error = self._last_answer_generation_metadata.get("error") if isinstance(self._last_answer_generation_metadata, dict) else None
                error_hint = f" 实际错误：{_safe_error_message(error)}" if error else ""
                return (
                    "本地文档/记忆中没有找到足够依据；同时当前 LLM 客户端不可用或调用失败，"
                    "因此无法继续使用模型通用知识生成回答。请检查 OPENAI_API_KEY、OPENAI_BASE_URL、"
                    f"MODEL_NAME，或查看 Trace 中的 generate_answer 错误。{error_hint}"
                )
            return "当前没有检索到可用文档片段，无法基于私域知识库给出可靠答案。"

        lines = ["未配置 OPENAI_API_KEY，以下是基于检索上下文的摘要式回答："]
        for document in built_context.documents[:3]:
            snippet = document.content.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(f"- [{document.doc_id}] {snippet}")
        return "\n".join(lines)


def _parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text or "", flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _safe_error_message(error: Any, max_chars: int = 420) -> str:
    if not error:
        return ""
    text = str(error).replace("\n", " ").strip()
    text = re.sub(r"(?i)(api[_-]?key['\":= ]+)[^, )}]+", r"\1***", text)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
