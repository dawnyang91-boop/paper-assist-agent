from typing import Any

from agent_state import AgentState


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    """Optional LangGraph runner that exposes the assistant loop as named nodes."""

    def __init__(self, agent_graph: Any):
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise LangGraphUnavailable("未安装 langgraph；请先安装 langgraph 后再启用该适配器。") from exc

        self.agent_graph = agent_graph
        self.MemorySaver = MemorySaver
        self.StateGraph = StateGraph
        self.END = END
        self.compiled = self._compile()

    def invoke(self, state: Any, config: dict | None = None):
        return self.compiled.invoke(state, config=config or {})

    def _compile(self):
        graph = self.StateGraph(dict)
        graph.add_node("process_user_input", self._process_user_input)
        graph.add_node("load_session_state", self._load_session_state)
        graph.add_node("understand_query", self._understand_query)
        graph.add_node("retrieve_rag", self._retrieve_rag)
        graph.add_node("retrieve_memory", self._retrieve_memory)
        graph.add_node("apply_skills", self._apply_skills)
        graph.add_node("assess_local_context", self._assess_local_context)
        graph.add_node("decide_next_action", self._decide_next_action)
        graph.add_node("call_mcp_tools", self._call_mcp_tools)
        graph.add_node("build_context", self._build_context)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("verify_answer", self._verify_answer)
        graph.add_node("write_memory", self._write_memory)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("process_user_input")
        graph.add_conditional_edges("process_user_input", self._route_after_input, {
            "continue": "load_session_state",
            "finalize": "finalize",
        })
        graph.add_edge("load_session_state", "understand_query")
        graph.add_edge("understand_query", "retrieve_rag")
        graph.add_edge("retrieve_rag", "retrieve_memory")
        graph.add_edge("retrieve_memory", "apply_skills")
        graph.add_edge("apply_skills", "assess_local_context")
        graph.add_edge("assess_local_context", "decide_next_action")
        graph.add_conditional_edges("decide_next_action", self._route_after_decision, {
            "call_tools": "call_mcp_tools",
            "build_context": "build_context",
        })
        graph.add_edge("call_mcp_tools", "build_context")
        graph.add_edge("build_context", "generate_answer")
        graph.add_edge("generate_answer", "verify_answer")
        graph.add_conditional_edges("verify_answer", self._route_after_verify, {
            "write_memory": "write_memory",
            "finalize": "finalize",
        })
        graph.add_edge("write_memory", "finalize")
        graph.add_edge("finalize", self.END)
        return graph.compile(checkpointer=self.MemorySaver())

    def _ensure_state(self, payload: dict) -> AgentState:
        if payload.get("_agent_state") is not None:
            return payload["_agent_state"]
        qa_state = AgentState(
            session_id=payload.get("session_id") or "default",
            raw_input=payload.get("question", ""),
            max_loop_steps=self.agent_graph.config.agent_loop_max_steps,
            explicit_memories=list(payload.get("memories") or []),
        )
        qa_state.usage.update({
            "include_memory_manager": payload.get("include_memory_manager", True),
            "include_mcp": payload.get("include_mcp", True),
            "include_skills": payload.get("include_skills", True),
            "memory_top_k": payload.get("memory_top_k", 3),
            "modes_override": list(payload.get("modes")) if payload.get("modes") is not None else None,
            "top_k": payload.get("top_k"),
            "per_query_limit": payload.get("per_query_limit"),
            "mqe_count": payload.get("mqe_count"),
            "rerank_top_k": payload.get("rerank_top_k"),
            "context_top_k": payload.get("context_top_k"),
            "write_memory": payload.get("write_memory", False),
        })
        payload["_agent_state"] = qa_state
        return qa_state

    def _export(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        payload["answer"] = state.answer
        payload["metadata"] = state.usage
        payload["errors"] = list(state.errors)
        payload["stop_reason"] = state.stop_reason
        return payload

    def _process_user_input(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.process_user_input(state)
        self.agent_graph.save_checkpoint(state, "process_user_input")
        return self._export(payload)

    def _route_after_input(self, payload: dict) -> str:
        return "continue" if self._ensure_state(payload).should_enter_loop else "finalize"

    def _load_session_state(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.load_session_state(state)
        self.agent_graph.save_checkpoint(state, "load_session_state")
        self.agent_graph.record_transcript(state, role="user", content=state.raw_input)
        return self._export(payload)

    def _understand_query(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.understand_query(
            state,
            modes=payload.get("modes"),
            include_mcp=payload.get("include_mcp", True),
            include_skills=payload.get("include_skills", True),
        )
        self.agent_graph.plan_retrieval_budget(
            state,
            top_k=payload.get("top_k"),
            per_query_limit=payload.get("per_query_limit"),
            rerank_top_k=payload.get("rerank_top_k"),
            context_top_k=payload.get("context_top_k"),
        )
        self.agent_graph.prepare_loop_context(state)
        self.agent_graph.save_checkpoint(state, "understand_query")
        return self._export(payload)

    def _retrieve_rag(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.retrieve_rag(
            state,
            top_k=state.usage.get("effective_query_top_k"),
            per_query_limit=state.usage.get("effective_per_query_limit"),
            mqe_count=payload.get("mqe_count"),
            rerank_top_k=state.usage.get("effective_rerank_top_k"),
        )
        self.agent_graph.save_checkpoint(state, "retrieve_rag")
        return self._export(payload)

    def _retrieve_memory(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.retrieve_memory(
            state,
            include_memory_manager=payload.get("include_memory_manager", True),
            top_k=payload.get("memory_top_k", 3),
        )
        return self._export(payload)

    def _apply_skills(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.apply_skills(state, include_skills=payload.get("include_skills", True))
        return self._export(payload)

    def _assess_local_context(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.assess_local_context(state, include_mcp=payload.get("include_mcp", True))
        self.agent_graph.save_checkpoint(state, "retrieve_context")
        return self._export(payload)

    def _decide_next_action(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        decision = self.agent_graph.decide_next_action(state, include_mcp=payload.get("include_mcp", True))
        state.decisions.append({"action": decision.action, "reason": decision.reason, "tool_requests": decision.tool_requests})
        payload["_decision"] = decision
        return self._export(payload)

    def _route_after_decision(self, payload: dict) -> str:
        decision = payload.get("_decision")
        return "call_tools" if getattr(decision, "action", "") == "call_tools" else "build_context"

    def _call_mcp_tools(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.call_mcp_tools(state, payload.get("_decision"))
        self.agent_graph.prepare_loop_context(state)
        self.agent_graph.save_checkpoint(state, "call_mcp_tools")
        return self._export(payload)

    def _build_context(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.prepare_general_knowledge_fallback(state)
        self.agent_graph.build_context(state, context_top_k=state.usage.get("effective_context_top_k"))
        return self._export(payload)

    def _generate_answer(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.generate_answer(state, stream_callback=payload.get("stream_callback"))
        return self._export(payload)

    def _verify_answer(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.verify_answer(state)
        self.agent_graph.save_checkpoint(state, "generate_and_verify")
        return self._export(payload)

    def _route_after_verify(self, payload: dict) -> str:
        return "write_memory" if payload.get("write_memory", False) else "finalize"

    def _write_memory(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.write_memory(state)
        self.agent_graph.save_checkpoint(state, "write_memory")
        return self._export(payload)

    def _finalize(self, payload: dict) -> dict:
        state = self._ensure_state(payload)
        self.agent_graph.finalize(state)
        self.agent_graph.save_checkpoint(state, "finalize")
        return self._export(payload)
