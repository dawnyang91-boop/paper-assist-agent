import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from agent.agent_state import AgentState, LoopDecision, QueryPlan
from agent.answer_verifier import AnswerVerifier, DOCUMENT_CITATION_RE
from agent.checkpoint_store import SQLiteCheckpointStore
from config import AppConfig
from memory.session_summary import SessionSummaryBuilder
from rag.context_builder import estimate_tokens
from rag.provenance import extract_source_refs_from_events, looks_like_follow_up, source_files_from_refs
from rag.rag_query import QUERY_MODE_BASIC
from rag.reranker import lexical_overlap, topic_overlap
from storage.transcript_store import TranscriptStore


class AgentGraph:
    """Synchronous bounded graph runner for one QA turn."""

    def __init__(
        self,
        agent: Any,
        config: AppConfig,
        verifier: Optional[AnswerVerifier] = None,
        transcript_store: Optional[TranscriptStore] = None,
        checkpoint_store: Optional[SQLiteCheckpointStore] = None,
        session_summary_builder: Optional[SessionSummaryBuilder] = None,
    ):
        self.agent = agent
        self.config = config
        self.verifier = verifier or AnswerVerifier()
        self.transcript_store = transcript_store or TranscriptStore(
            root_dir=config.transcript_dir,
            enabled=config.transcript_enabled,
        )
        self.checkpoint_store = checkpoint_store or SQLiteCheckpointStore(
            db_path=config.checkpoint_db_path,
            enabled=config.checkpoint_enabled,
        )
        self.session_summary_builder = session_summary_builder or SessionSummaryBuilder(
            max_chars=config.session_summary_max_chars,
        )

    def _sentinel_enabled(self) -> bool:
        return bool(
            getattr(self.config, "sentinel_enabled", False)
            and getattr(self.agent, "security_manager", None) is not None
        )

    def init_sentinel_trace(self, state: AgentState) -> None:
        if self._sentinel_enabled():
            state.usage["sentinel"] = self.agent.security_manager.trace_base()

    def run(
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
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> AgentState:
        state = AgentState(
            session_id=session_id or "default",
            raw_input=question,
            max_loop_steps=self.config.agent_loop_max_steps,
            explicit_memories=list(memories or []),
        )
        state.usage.update({
            "include_memory_manager": include_memory_manager,
            "include_mcp": include_mcp,
            "include_skills": include_skills,
            "memory_top_k": memory_top_k,
            "modes_override": list(modes) if modes is not None else None,
            "top_k": top_k,
            "per_query_limit": per_query_limit,
            "mqe_count": mqe_count,
            "rerank_top_k": rerank_top_k,
            "context_top_k": context_top_k,
            "write_memory": write_memory,
        })
        self.init_sentinel_trace(state)

        self.process_user_input(state)
        self.save_checkpoint(state, "process_user_input")
        self.load_session_state(state)
        self.save_checkpoint(state, "load_session_state")
        self.record_transcript(state, role="user", content=state.raw_input)
        if not state.should_enter_loop:
            self.finalize(state)
            self.save_checkpoint(state, "finalize")
            return state

        self.understand_query(state, modes=modes, include_mcp=include_mcp, include_skills=include_skills)
        self.plan_retrieval_budget(
            state,
            top_k=top_k,
            per_query_limit=per_query_limit,
            rerank_top_k=rerank_top_k,
            context_top_k=context_top_k,
        )
        self.save_checkpoint(state, "understand_query")
        self.prepare_loop_context(state)
        self.retrieve_rag(
            state,
            top_k=state.usage.get("effective_query_top_k"),
            per_query_limit=state.usage.get("effective_per_query_limit"),
            mqe_count=mqe_count,
            rerank_top_k=state.usage.get("effective_rerank_top_k"),
        )
        self.save_checkpoint(state, "retrieve_rag")
        self.retrieve_memory(state, include_memory_manager=include_memory_manager, top_k=memory_top_k)
        self.rehydrate_rag_from_active_sources(
            state,
            rerank_top_k=state.usage.get("effective_rerank_top_k"),
        )
        self.apply_skills(state, include_skills=include_skills)
        self.assess_local_context(state, include_mcp=include_mcp)
        self.save_checkpoint(state, "retrieve_context")

        decision = self.decide_next_action(state, include_mcp=include_mcp)
        state.decisions.append({"action": decision.action, "reason": decision.reason, "tool_requests": decision.tool_requests})
        if decision.action == "retrieve_more_rag":
            self.retrieve_rag(
                state,
                top_k=(state.usage.get("effective_query_top_k") or self.config.rag_query_top_k) * 2,
                per_query_limit=(state.usage.get("effective_per_query_limit") or self.config.rag_query_per_query_limit) * 2,
                mqe_count=mqe_count,
                rerank_top_k=(state.usage.get("effective_rerank_top_k") or self.config.rag_rerank_top_k) * 2,
            )
            self.save_checkpoint(state, "retrieve_more_rag")
            decision = self.decide_next_action(state, include_mcp=include_mcp)
            state.decisions.append({"action": decision.action, "reason": decision.reason, "tool_requests": decision.tool_requests})
        elif decision.action == "retrieve_more_memory":
            self.retrieve_memory(state, include_memory_manager=include_memory_manager, top_k=memory_top_k * 2)
            self.rehydrate_rag_from_active_sources(
                state,
                rerank_top_k=state.usage.get("effective_rerank_top_k"),
            )
            self.save_checkpoint(state, "retrieve_more_memory")
            decision = self.decide_next_action(state, include_mcp=include_mcp)
            state.decisions.append({"action": decision.action, "reason": decision.reason, "tool_requests": decision.tool_requests})

        if decision.action == "call_tools":
            self.call_mcp_tools(state, decision)
            self.save_checkpoint(state, "call_mcp_tools")
            self.prepare_loop_context(state)
        elif decision.action == "finalize_insufficient_context":
            state.stop_reason = "insufficient_context"
        elif decision.action == "stop_max_steps":
            state.stop_reason = "max_loop_steps"

        self.prepare_general_knowledge_fallback(state)
        self.build_context(state, context_top_k=state.usage.get("effective_context_top_k"))
        self.generate_answer(state, stream_callback=stream_callback)
        self.verify_answer(state)
        self.save_checkpoint(state, "generate_and_verify")

        while self._should_repair(state):
            state.repair_attempts += 1
            self.generate_answer(state, stream_callback=stream_callback)
            self.verify_answer(state)
            self.save_checkpoint(state, "repair_answer")

        if write_memory:
            self.write_memory(state)
            self.save_checkpoint(state, "write_memory")
        self.finalize(state)
        self.save_checkpoint(state, "finalize")
        return state

    def process_user_input(self, state: AgentState) -> None:
        clean = (state.raw_input or "").strip()
        state.question = clean
        state.original_question = clean
        if not clean:
            state.should_enter_loop = False
            state.command_result = "空输入。"
            state.answer = "请输入问题后再开始问答。"
            state.stop_reason = "empty_input"
            return
        if clean in {"/exit", "/quit"}:
            state.should_enter_loop = False
            state.command_result = "exit"
            state.answer = "再见。"
            state.stop_reason = "command_exit"
            return
        if clean.startswith("/media "):
            state.should_enter_loop = False
            state.media_path = clean[len("/media "):].strip()
            state.stop_reason = "media_input"
            return
        if self._sentinel_enabled():
            decision = self.agent.security_manager.pre_check_user_input(clean)
            sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
            sentinel["input_decision"] = decision.to_dict()
            risk_flags = sorted({finding.risk_type for finding in decision.findings})
            sentinel["current_risk_flags"] = risk_flags
            sentinel["previous_risk_flags"] = self._previous_risk_flags(state)
            sentinel["jailbreak_setup_detected"] = any(
                flag in {"jailbreak", "delimiter_confusion", "encoded_instruction"}
                for flag in risk_flags
            )
            sentinel["pending_sensitive_intent"] = any(
                flag in {"data_exfiltration", "authority_claim", "audit_excuse"}
                for flag in risk_flags
            )
            if not decision.allowed:
                state.should_enter_loop = False
                state.stop_reason = "sentinel_blocked_input"
                sentinel["blocked"] = True
                state.answer = (
                    "LLM-Sentinel 已拦截该输入，因为它包含疑似 Prompt Injection、越狱或敏感信息提取意图。"
                    "请去掉要求忽略系统规则、泄露提示词或绕过安全策略的内容后重试。"
                )
                return
        try:
            normalized = self.agent.normalize_question_to_english(clean)
            state.original_question = normalized.get("original_question") or clean
            state.question = normalized.get("english_question") or clean
            state.response_language = normalized.get("response_language") or "Chinese"
            state.usage.update({
                "original_question": state.original_question,
                "english_question": state.question,
                "response_language": state.response_language,
                "question_storage_language": "English",
            })
        except Exception as exc:
            state.add_warning("process_user_input", f"question normalization failed, using original input: {exc}")

    def _previous_risk_flags(self, state: AgentState) -> List[str]:
        flags: List[str] = []
        events = list(getattr(state, "resumed_transcript", []) or [])
        if not events and getattr(self.transcript_store, "enabled", False):
            try:
                events = self.transcript_store.load(state.session_id)[-12:]
            except Exception:
                events = []
        for event in events:
            if not isinstance(event, dict):
                continue
            metadata = event.get("metadata") or {}
            sentinel = metadata.get("sentinel") if isinstance(metadata, dict) else None
            if isinstance(sentinel, dict):
                flags.extend(str(item) for item in sentinel.get("current_risk_flags", []) or [])
        return sorted(set(flags))

    def record_transcript(self, state: AgentState, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            event = self.transcript_store.append(
                session_id=state.session_id,
                role=role,
                content=content,
                metadata=metadata,
            )
            if event:
                state.messages.append(event)
        except Exception as exc:
            state.add_error("record_transcript", exc)

    def load_session_state(self, state: AgentState) -> None:
        if not self.config.transcript_resume_enabled:
            return
        try:
            events = self.transcript_store.load(state.session_id)
        except Exception as exc:
            state.add_error("load_session_state", exc)
            return
        if not events:
            return

        recent_events = events[-max(1, self.config.transcript_resume_max_events):]
        state.resumed_transcript = recent_events
        state.messages.extend(recent_events)
        source_refs = extract_source_refs_from_events(recent_events)
        if source_refs and looks_like_follow_up(state.raw_input):
            state.active_source_refs = [ref.to_dict() for ref in source_refs]
            state.usage["active_source_refs"] = list(state.active_source_refs)
            state.usage["active_source_files"] = source_files_from_refs(source_refs)
        transcript_memories = []
        if self.config.session_summary_enabled:
            summary_memory = self.session_summary_builder.build_memory(recent_events)
            transcript_memories.append(summary_memory)
            self._fuse_session_summary_into_working_memory(state, summary_memory)
        else:
            for event in recent_events:
                role = event.get("role", "transcript")
                content = event.get("content", "")
                if not content:
                    continue
                transcript_memories.append({
                    "role": f"transcript:{role}",
                    "type": "transcript",
                    "content": self._trim_text(content, self.config.memory_summary_max_chars),
                    "metadata": {"ts": event.get("ts"), "source": "transcript"},
                })
        state.explicit_memories = transcript_memories + state.explicit_memories

    def understand_query(
        self,
        state: AgentState,
        modes: Optional[Sequence[str]],
        include_mcp: bool,
        include_skills: bool,
    ) -> None:
        understanding = self.agent.understand_query(state.question)
        retrieval_modes = tuple(modes or getattr(understanding, "modes", ("basic",)))
        intent, need_mcp, allowed_tools = self._infer_intent_and_tools(state.question)
        state.query_plan = QueryPlan(
            intent=intent,
            need_rag=intent not in {"casual_chat", "command", "media_input"},
            need_memory=intent != "command",
            need_mcp=bool(include_mcp and self.agent.mcp_manager is not None and need_mcp),
            need_skills=bool(include_skills and self.agent.skill_manager is not None),
            retrieval_modes=retrieval_modes,
            answer_style="grounded",
            high_value_candidate=intent not in {"casual_chat", "command", "media_input"},
            allowed_tools=allowed_tools,
            reason=getattr(understanding, "reason", ""),
        )
        state.usage["query_understanding"] = understanding

    def plan_retrieval_budget(
        self,
        state: AgentState,
        top_k: Optional[int],
        per_query_limit: Optional[int],
        rerank_top_k: Optional[int],
        context_top_k: Optional[int],
    ) -> None:
        understanding = state.usage.get("query_understanding")
        query_type = getattr(understanding, "query_type", "complex")
        is_complex = query_type in {"complex", "explanatory"} or len(state.question) >= 40

        if is_complex:
            effective_query_top_k = top_k or self.config.rag_complex_query_top_k
            effective_per_query_limit = per_query_limit or self.config.rag_complex_per_query_limit
            effective_rerank_top_k = rerank_top_k or self.config.rag_complex_rerank_top_k
            effective_context_top_k = context_top_k if context_top_k is not None else self.config.rag_complex_context_top_k
        else:
            effective_query_top_k = top_k or self.config.rag_query_top_k
            effective_per_query_limit = per_query_limit or self.config.rag_query_per_query_limit
            effective_rerank_top_k = rerank_top_k or self.config.rag_rerank_top_k
            effective_context_top_k = context_top_k if context_top_k is not None else self.config.rag_context_top_k

        state.usage.update({
            "retrieval_budget_reason": "complex_dynamic" if is_complex else "simple_default",
            "effective_query_top_k": effective_query_top_k,
            "effective_per_query_limit": effective_per_query_limit,
            "effective_rerank_top_k": effective_rerank_top_k,
            "effective_context_top_k": effective_context_top_k,
        })

    def prepare_loop_context(self, state: AgentState) -> None:
        state.loop_step += 1
        if state.loop_step > state.max_loop_steps:
            state.stop_reason = "max_loop_steps"
        max_observations = max(1, self.config.agent_loop_max_tool_calls)
        if len(state.tool_observations) > max_observations:
            state.tool_observations = state.tool_observations[-max_observations:]
            state.add_warning("prepare_loop_context", "已裁剪较早的 MCP tool observation。")
        for observation in state.tool_observations:
            content = observation.get("content", "")
            trimmed = self._trim_text(content, self.config.agent_loop_max_tool_observation_chars)
            if trimmed != content:
                observation["content"] = trimmed
                observation.setdefault("metadata", {})["compressed"] = True
        max_memories = max(1, self.config.agent_loop_max_memory_items)
        if len(state.explicit_memories) > max_memories:
            state.explicit_memories = state.explicit_memories[-max_memories:]
            state.add_warning("prepare_loop_context", "已裁剪较早的显式记忆上下文。")

    def retrieve_rag(
        self,
        state: AgentState,
        top_k: Optional[int],
        per_query_limit: Optional[int],
        mqe_count: Optional[int],
        rerank_top_k: Optional[int],
    ) -> None:
        if not state.query_plan or not state.query_plan.need_rag:
            return
        try:
            state.candidates = self.agent.query_engine.search(
                question=state.question,
                modes=tuple(state.query_plan.retrieval_modes),
                top_k=top_k or self.config.rag_query_top_k,
                per_query_limit=per_query_limit,
                mqe_count=mqe_count,
                source_file_hints=state.usage.get("active_source_files", []),
                provenance_refs=state.active_source_refs,
            )
            state.usage["retrieval_diagnostics"] = getattr(
                self.agent.query_engine,
                "last_retrieval_diagnostics",
                {},
            )
            state.usage["rag_active_source_ref_count"] = len(state.active_source_refs)
            state.ranked_chunks = self.agent.reranker.rerank(
                question=state.question,
                candidates=state.candidates,
                top_k=rerank_top_k or self.config.rag_rerank_top_k,
            )
            if hasattr(self.agent.query_engine, "entity_extractor"):
                coverage = self.agent.query_engine.entity_extractor.coverage(state.question, state.ranked_chunks)
                state.usage["entity_coverage"] = coverage.to_dict()
                diagnostics = state.usage.setdefault("retrieval_diagnostics", {})
                diagnostics["entity_coverage"] = coverage.to_dict()
            self.sanitize_rag_context(state)
        except Exception as exc:
            state.add_error("retrieve_rag", exc)
            state.candidates = []
            state.ranked_chunks = []

    def rehydrate_rag_from_active_sources(
        self,
        state: AgentState,
        rerank_top_k: Optional[int] = None,
    ) -> None:
        """Use references restored from transcript/memory to fetch citeable RAG chunks.

        Memory items are useful for routing a follow-up question back to the
        document being discussed, but the final answer should still cite [D]
        snippets. This second pass runs only when new source refs appeared after
        the first RAG retrieval, usually from semantic/working memory.
        """
        if not state.query_plan or not state.query_plan.need_rag:
            return
        if not state.active_source_refs:
            return
        previous_ref_count = int(state.usage.get("rag_active_source_ref_count") or 0)
        if previous_ref_count >= len(state.active_source_refs):
            return
        try:
            source_chunks = self.agent.query_engine.search(
                question=state.question,
                modes=(QUERY_MODE_BASIC,),
                top_k=max(
                    self.config.rag_query_top_k,
                    len(state.active_source_refs),
                ),
                per_query_limit=state.usage.get("effective_per_query_limit") or self.config.rag_query_per_query_limit,
                source_file_hints=state.usage.get("active_source_files", []),
                provenance_refs=state.active_source_refs,
            )
            state.candidates = self.agent.query_engine.dedupe_results([
                *state.candidates,
                *source_chunks,
            ])
            state.ranked_chunks = self.agent.reranker.rerank(
                question=state.question,
                candidates=state.candidates,
                top_k=rerank_top_k or self.config.rag_rerank_top_k,
            )
            if hasattr(self.agent.query_engine, "entity_extractor"):
                coverage = self.agent.query_engine.entity_extractor.coverage(state.question, state.ranked_chunks)
                state.usage["entity_coverage"] = coverage.to_dict()
            diagnostics = dict(getattr(self.agent.query_engine, "last_retrieval_diagnostics", {}) or {})
            diagnostics["provenance_rehydration_used"] = True
            diagnostics["active_source_ref_count"] = len(state.active_source_refs)
            state.usage["retrieval_diagnostics"] = diagnostics
            state.usage["rag_active_source_ref_count"] = len(state.active_source_refs)
            state.usage["provenance_rehydration_used"] = True
            self.sanitize_rag_context(state)
        except Exception as exc:
            state.add_error("rehydrate_rag_from_active_sources", exc)

    def sanitize_rag_context(self, state: AgentState) -> None:
        if not self._sentinel_enabled():
            return
        try:
            sanitized, decisions = self.agent.security_manager.sanitize_ranked_chunks(state.ranked_chunks)
            state.ranked_chunks = sanitized
            sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
            sentinel["rag_sanitized_count"] = len(decisions)
            sentinel["rag_sanitize_decisions"] = decisions
            if decisions:
                taint = sentinel.setdefault("taint_trace", {})
                taint.setdefault("risky_rag_chunks", []).extend(
                    item.get("rank") for item in decisions if item.get("rank") is not None
                )
                taint["answer_used_risky_context"] = True
        except Exception as exc:
            state.add_error("sentinel.sanitize_rag_context", exc)

    def retrieve_memory(self, state: AgentState, include_memory_manager: bool, top_k: int) -> None:
        if not state.query_plan or not state.query_plan.need_memory:
            return
        if not include_memory_manager or self.agent.memory_manager is None:
            return
        try:
            state.retrieved_memories = self.agent.memory_manager.retrieve_memories(
                query=state.question,
                session_id=state.session_id,
                top_k=top_k,
            )
            state.usage["memory_retrieval_diagnostics"] = state.retrieved_memories.get("_diagnostics", {})
            self._merge_memory_provenance_refs(state)
        except Exception as exc:
            state.add_error("retrieve_memory", exc)
            state.retrieved_memories = {}
            state.usage["memory_retrieval_diagnostics"] = {"error": str(exc)}

    def _merge_memory_provenance_refs(self, state: AgentState) -> None:
        refs = list(state.active_source_refs or [])
        seen = {
            (item.get("source_file"), item.get("chunk_index"))
            for item in refs
            if isinstance(item, dict)
        }
        for item in self.agent._flatten_retrieved_memories(state.retrieved_memories):
            metadata = item.get("metadata") or {}
            payload = metadata.get("payload") if isinstance(metadata, dict) else None
            provenance = metadata.get("provenance") if isinstance(metadata, dict) else None
            if not provenance and isinstance(payload, dict):
                provenance = payload.get("provenance")
            if not isinstance(provenance, dict):
                continue
            for source in provenance.get("sources") or []:
                if not isinstance(source, dict) or not source.get("source_file"):
                    continue
                ref = {
                    "source_file": source.get("source_file"),
                    "chunk_index": source.get("chunk_index"),
                    "doc_id": source.get("doc_id"),
                    "heading_paths": source.get("heading_paths") or [],
                    "from_memory": True,
                }
                key = (ref.get("source_file"), ref.get("chunk_index"))
                if key in seen:
                    continue
                seen.add(key)
                refs.append(ref)
        if refs:
            state.active_source_refs = refs
            state.usage["active_source_refs"] = refs
            state.usage["active_source_files"] = sorted({
                str(item.get("source_file"))
                for item in refs
                if isinstance(item, dict) and item.get("source_file")
            })

    def apply_skills(self, state: AgentState, include_skills: bool) -> None:
        if not include_skills or self.agent.skill_manager is None:
            return
        if state.query_plan and not state.query_plan.need_skills:
            return
        try:
            queries = [state.question]
            if state.original_question and state.original_question != state.question:
                queries.append(state.original_question)
            contexts = []
            seen = set()
            for query in queries:
                for item in self.agent.skill_manager.retrieve_context(query):
                    key = (item.get("role"), item.get("content"))
                    if key in seen:
                        continue
                    seen.add(key)
                    contexts.append(item)
            state.skill_contexts = contexts
        except Exception as exc:
            state.add_error("apply_skills", exc)
            state.skill_contexts = [{
                "role": "skill_error",
                "type": "skill",
                "content": f"Skill 执行失败：{exc}",
            }]

    def assess_local_context(self, state: AgentState, include_mcp: bool) -> None:
        assessment_queries = [state.question]
        if state.original_question and state.original_question != state.question:
            assessment_queries.append(state.original_question)
        top_document_lexical = max(
            (lexical_overlap(query, chunk.content) for query in assessment_queries for chunk in state.ranked_chunks),
            default=max((chunk.lexical_score for chunk in state.ranked_chunks), default=0.0),
        )
        top_document_topic = max(
            (topic_overlap(query, chunk.content) for query in assessment_queries for chunk in state.ranked_chunks),
            default=0.0,
        )
        top_document_score = max((chunk.rerank_score for chunk in state.ranked_chunks), default=0.0)
        memory_items = self.agent._flatten_retrieved_memories(state.retrieved_memories) if state.retrieved_memories else []
        top_memory_lexical = max(
            (lexical_overlap(query, item.get("content", "")) for query in assessment_queries for item in memory_items),
            default=0.0,
        )
        top_memory_topic = max(
            (topic_overlap(query, item.get("content", "")) for query in assessment_queries for item in memory_items),
            default=0.0,
        )
        threshold = self.config.agent_loop_local_relevance_threshold
        has_relevant_local_context = max(top_document_topic, top_memory_topic) >= threshold
        if state.active_source_refs and state.ranked_chunks:
            has_relevant_local_context = True
        entity_coverage = state.usage.get("entity_coverage") or (state.usage.get("retrieval_diagnostics", {}) or {}).get("entity_coverage") or {}
        entity_coverage_failed = bool(entity_coverage.get("entity_coverage_failed"))
        if self.config.rag_entity_required_for_strong_query and entity_coverage_failed:
            has_relevant_local_context = False
        online_search_available = self._online_search_available()
        online_available = bool(
            include_mcp
            and self.config.agent_loop_online_fallback_enabled
            and self.agent.mcp_manager is not None
            and online_search_available
            and state.tool_call_count < self.config.agent_loop_max_tool_calls
        )
        local_context_insufficient = bool(
            state.query_plan
            and state.query_plan.intent == "private_qa"
            and not has_relevant_local_context
        )
        need_online_search = bool(
            online_available
            and not state.tool_observations
            and local_context_insufficient
        )
        allow_general_knowledge_fallback = bool(local_context_insufficient and not online_available)
        state.usage["local_context_assessment"] = {
            "top_document_lexical": top_document_lexical,
            "top_document_topic": top_document_topic,
            "top_document_score": top_document_score,
            "top_memory_lexical": top_memory_lexical,
            "top_memory_topic": top_memory_topic,
            "threshold": threshold,
            "entity_coverage_failed": entity_coverage_failed,
            "active_source_ref_count": len(state.active_source_refs),
            "active_source_files": state.usage.get("active_source_files", []),
            "has_relevant_local_context": has_relevant_local_context,
            "online_search_available": online_search_available,
            "online_available": online_available,
            "need_online_search": need_online_search,
            "allow_general_knowledge_fallback": allow_general_knowledge_fallback,
        }
        state.usage["allow_general_knowledge_fallback"] = allow_general_knowledge_fallback
        if need_online_search:
            state.query_plan.need_mcp = True
            if "brave-search" not in state.query_plan.allowed_tools:
                state.query_plan.allowed_tools.append("brave-search")
            state.usage["online_search_fallback"] = True

    def decide_next_action(self, state: AgentState, include_mcp: bool) -> LoopDecision:
        if state.stop_reason == "max_loop_steps":
            return LoopDecision(action="build_context", reason="已达到最大循环步数。")
        if self.config.agent_loop_planner_enabled:
            planner_decision = self._llm_decide_next_action(state, include_mcp=include_mcp)
            if planner_decision is not None:
                return planner_decision
        if (
            include_mcp
            and state.query_plan
            and state.query_plan.need_mcp
            and self.agent.mcp_manager is not None
            and state.tool_call_count < self.config.agent_loop_max_tool_calls
            and not state.tool_observations
        ):
            if state.usage.get("online_search_fallback"):
                return LoopDecision(
                    action="call_tools",
                    tool_requests=[{
                        "server": "brave-search",
                        "tool": "brave_web_search",
                        "arguments": {"query": state.question, "count": 5},
                    }],
                    reason="本地 RAG/记忆相关性不足，调用线上搜索 MCP 补充上下文。",
                )
            return LoopDecision(action="call_tools", reason="QueryPlan 判断需要 MCP 工具补充上下文。")
        return LoopDecision(action="build_context", reason="已有上下文或无需工具，进入回答生成。")

    def call_mcp_tools(self, state: AgentState, decision: LoopDecision) -> None:
        if self.agent.mcp_manager is None:
            return
        if self._agentic_mcp_enabled():
            self._call_agentic_mcp_tools(state)
        else:
            self._call_rule_mcp_tools(state, decision)
        if state.usage.get("online_search_fallback") and not self._has_successful_tool_observation(state):
            state.usage["allow_general_knowledge_fallback"] = True
            state.add_warning("call_mcp_tools", "线上搜索 MCP 不可用或未返回有效结果，将允许模型使用通用知识回答。")

    def build_context(self, state: AgentState, context_top_k: Optional[int]) -> None:
        memories = self._build_context_memories(state)
        state.usage["context_memories"] = memories
        max_tokens = min(self.config.rag_context_max_tokens, self.config.agent_loop_context_max_tokens)
        try:
            state.built_context = self.agent.context_builder.build_context(
                question=state.question,
                ranked_chunks=state.ranked_chunks,
                memories=memories,
                top_k=context_top_k if context_top_k is not None else self.config.rag_context_top_k,
                max_tokens=max_tokens,
                allow_general_knowledge_fallback=bool(state.usage.get("allow_general_knowledge_fallback")),
                evidence_status=self._context_evidence_status(state),
            )
            state.built_context = self._fit_context_budget(state, context_top_k=context_top_k, max_tokens=max_tokens)
        except Exception as exc:
            state.add_error("build_context", exc, recoverable=False)
            raise

    def prepare_general_knowledge_fallback(self, state: AgentState) -> None:
        if not state.usage.get("allow_general_knowledge_fallback"):
            return
        if state.usage.get("general_knowledge_fallback_used"):
            return
        if any(item.get("type") == "general_knowledge" for item in state.tool_observations):
            state.usage["general_knowledge_fallback_used"] = True
            return

        try:
            draft = self.agent.generate_general_knowledge_draft(state.question)
        except Exception as exc:
            state.add_error("general_knowledge_fallback", exc)
            draft = None

        if not draft:
            state.add_warning("general_knowledge_fallback", "LLM 通用知识草稿生成失败，将仅使用最终回答 prompt 兜底。")
            state.usage["general_knowledge_fallback_used"] = False
            return

        content = (
            "Local documents/memories did not provide enough evidence. The following draft was generated from model general knowledge "
            "for final-answer organization only; it is not private document or memory evidence:\n"
            f"{draft}"
        )
        state.tool_observations.append({
            "role": "llm:general_knowledge_draft",
            "type": "general_knowledge",
            "content": content,
            "metadata": {"source": "llm_general_knowledge_fallback", "citation_allowed": False},
        })
        state.usage["general_knowledge_fallback_used"] = True
        state.usage["general_knowledge_draft_chars"] = len(draft)

    def generate_answer(
        self,
        state: AgentState,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        try:
            state.answer = self.agent.generate_answer(state.built_context, stream_callback=stream_callback)
            state.usage["answer_generation"] = getattr(self.agent, "_last_answer_generation_metadata", {})
        except Exception as exc:
            state.add_error("generate_answer", exc)
            self.agent._last_answer_generation_metadata = {"fallback": True, "error": str(exc)}
            state.answer = self.agent._fallback_answer(state.built_context)
            state.usage["answer_generation"] = getattr(self.agent, "_last_answer_generation_metadata", {"fallback": True, "error": str(exc)})
        self.post_check_answer(state)

    def post_check_answer(self, state: AgentState) -> None:
        if not self._sentinel_enabled() or not state.answer:
            return
        try:
            output_decision = self.agent.security_manager.post_check_output(state.answer)
            cascade_decision = self.agent.security_manager.check_cascade_answer(
                state.answer,
                context_trace=self._build_cascade_trace(state),
            )
            decision = self.agent.security_manager._merge_decisions([output_decision, cascade_decision])
            sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
            sentinel["output_decision"] = decision.to_dict()
            if decision.sanitized_text and decision.sanitized_text != state.answer:
                state.answer = decision.sanitized_text
                state.add_warning("sentinel.output_guard", "输出包含疑似敏感泄露内容，已由 Sentinel 净化。")
        except Exception as exc:
            state.add_error("sentinel.post_check_answer", exc)

    def verify_answer(self, state: AgentState) -> None:
        result = self.verifier.verify(state.answer, state.built_context, query_plan=state.query_plan)
        state.verification = result.to_dict()
        state.citations = [f"D{match}" for match in DOCUMENT_CITATION_RE.findall(state.answer or "")]

    def write_memory(self, state: AgentState) -> None:
        if self.agent.memory_manager is None or not state.answer:
            return
        if self._sentinel_enabled():
            try:
                allowed, decisions = self.agent.security_manager.should_write_memory_with_trace(
                    state.question,
                    state.answer,
                    context_trace=self._build_cascade_trace(state, memory_write_requested=True),
                )
                rogue_decision = self.agent.security_manager.check_rogue_trace(self._build_behavior_trace(state, memory_write_requested=True))
                if rogue_decision.findings:
                    decisions.append(rogue_decision.to_dict())
                    allowed = allowed and rogue_decision.allowed
                sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
                sentinel.setdefault("memory_write_decisions", []).extend(decisions)
                if not allowed:
                    state.add_warning("sentinel.memory_safety", "本轮问答包含疑似安全策略污染或级联风险，已跳过记忆写入。")
                    state.writeback_events.append({
                        "type": "memory_safety",
                        "result": {"skipped": True, "decisions": decisions},
                    })
                    return
            except Exception as exc:
                state.add_error("sentinel.memory_write_check", exc)
        try:
            consolidations = self.agent.memory_manager.add_interaction(
                state.session_id,
                state.question,
                state.answer,
            )
            state.writeback_events.extend({
                "type": "working_memory",
                "result": item,
            } for item in consolidations)
        except Exception as exc:
            state.add_error("write_memory.add_interaction", exc)

        if state.query_plan and not state.query_plan.high_value_candidate:
            return
        try:
            fact_result = self.agent.memory_manager.process_high_value_qa(
                state.question,
                state.answer,
                session_id=state.session_id,
                provenance=self._answer_references(state),
            )
            state.writeback_events.append({"type": "semantic_fact", "result": fact_result})
        except Exception as exc:
            state.add_error("write_memory.process_high_value_qa", exc)

    def finalize(self, state: AgentState) -> None:
        if not state.stop_reason:
            state.stop_reason = "final"
        state.usage.update({
            "loop_steps": state.loop_step,
            "tool_call_count": state.tool_call_count,
            "candidate_count": len(state.candidates),
            "ranked_count": len(state.ranked_chunks),
            "document_count": len(getattr(state.built_context, "documents", []) or []),
            "memory_count": len(state.usage.get("context_memories", [])),
            "skill_context_count": len(state.skill_contexts),
            "mcp_context_count": len([item for item in state.tool_observations if item.get("type") == "mcp"]),
            "context_tokens": getattr(state.built_context, "token_estimate", 0),
            "stop_reason": state.stop_reason,
            "errors": list(state.errors),
            "warnings": list(state.warnings),
            "verification": dict(state.verification),
            "decisions": list(state.decisions),
            "sentinel": state.usage.get("sentinel", {}),
            "resumed_transcript_count": len(state.resumed_transcript),
            "checkpoint_enabled": self.checkpoint_store.enabled,
        })
        if self._sentinel_enabled():
            try:
                behavior_trace = self._build_behavior_trace(state)
                profile = self.agent.security_manager.update_behavior_profile(state.session_id, behavior_trace)
                sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
                sentinel["behavior_profile"] = profile
                sentinel["session_risk_score"] = profile.get("session_risk_score", 0.0)
                state.usage["sentinel"] = sentinel
                if float(profile.get("session_risk_score", 0.0) or 0.0) >= 0.85:
                    state.add_warning("sentinel.rogue_agent_guard", "当前 session 风险较高，后续自治工具调用应进入人工复核。")
                    state.usage["warnings"] = list(state.warnings)
            except Exception as exc:
                state.add_error("sentinel.behavior_profile", exc)
        if state.answer:
            self.record_transcript(
                state,
                role="assistant",
                content=state.answer,
                metadata={
                    "stop_reason": state.stop_reason,
                    "verification": state.verification,
                    "references": self._answer_references(state),
                },
            )

    def _answer_references(self, state: AgentState) -> Dict[str, Any]:
        built_context = getattr(state, "built_context", None)
        documents = list(getattr(built_context, "documents", []) or [])
        return self._jsonable({
            "sources": [
                {
                    "doc_id": document.doc_id,
                    "source_file": document.source_file,
                    "chunk_index": document.chunk_index,
                    "score": document.score,
                    "heading_paths": document.heading_paths,
                    "metadata": document.metadata,
                }
                for document in documents
            ],
            "memories": list(state.usage.get("context_memories", []) or []),
            "citations": list(state.citations or []),
        })

    def _jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return self._jsonable({field: getattr(value, field) for field in value.__dataclass_fields__})
        if hasattr(value, "__dict__"):
            return self._jsonable(vars(value))
        return str(value)

    def _build_context_memories(self, state: AgentState) -> List[Dict[str, Any]]:
        memories = list(state.explicit_memories)
        memories.extend(state.skill_contexts)
        if state.retrieved_memories:
            memories.extend(self.agent._flatten_retrieved_memories(state.retrieved_memories))
        memories.extend(state.tool_observations)
        compacted = []
        for memory in memories:
            item = dict(memory)
            if item.get("content"):
                item["content"] = self._trim_text(str(item["content"]), self.config.memory_summary_max_chars)
            compacted.append(item)
        compacted = self._filter_unsafe_memory_contexts(state, compacted)
        max_items = max(1, self.config.agent_loop_max_memory_items)
        if len(compacted) <= max_items:
            return compacted
        tool_items = [item for item in compacted if str(item.get("type", "")).startswith("mcp")]
        non_tool_items = [item for item in compacted if not str(item.get("type", "")).startswith("mcp")]
        kept_tool_items = tool_items[-min(len(tool_items), max_items):]
        remaining = max_items - len(kept_tool_items)
        return non_tool_items[:remaining] + kept_tool_items

    def _filter_unsafe_memory_contexts(self, state: AgentState, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._sentinel_enabled() or not contexts:
            return contexts
        memory_like = []
        passthrough = []
        for item in contexts:
            item_type = str(item.get("type") or item.get("memory_type") or "").lower()
            role = str(item.get("role") or "").lower()
            if (
                item_type in {"working", "episodic", "semantic", "sensory", "memory", "semantic_fact", "working_memory"}
                or "memory" in role
                or role in {"working", "episodic", "semantic", "sensory"}
            ):
                memory_like.append(item)
            else:
                passthrough.append(item)
        if not memory_like:
            return contexts
        try:
            filtered, decisions = self.agent.security_manager.filter_memory_contexts(memory_like)
            if decisions:
                sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
                sentinel.setdefault("memory_context_decisions", []).extend(decisions)
                taint = sentinel.setdefault("taint_trace", {})
                taint.setdefault("risky_memory_contexts", []).extend(
                    item.get("index") for item in decisions if item.get("index") is not None
                )
                taint["answer_used_risky_context"] = True
                state.add_warning("sentinel.memory_safety", f"已过滤 {len(decisions)} 条疑似污染记忆上下文。")
            return passthrough + filtered
        except Exception as exc:
            state.add_error("sentinel.memory_context_filter", exc)
            return contexts

    def _sanitize_tool_observation(self, state: AgentState, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not self._sentinel_enabled():
            return observation
        try:
            sanitized, decision = self.agent.security_manager.sanitize_tool_observation(observation)
            if decision.findings:
                sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
                sentinel.setdefault("tool_output_decisions", []).append(decision.to_dict())
                taint = sentinel.setdefault("taint_trace", {})
                taint.setdefault("risky_tool_outputs", []).append(observation.get("role") or observation.get("type") or "tool_output")
                taint["answer_used_risky_context"] = True
                state.add_warning("sentinel.tool_output", "MCP 工具输出包含疑似指令注入内容，已净化后再进入上下文。")
            return sanitized
        except Exception as exc:
            state.add_error("sentinel.tool_output_sanitize", exc)
            return observation

    def _build_cascade_trace(self, state: AgentState, memory_write_requested: bool = False) -> Dict[str, Any]:
        sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
        taint = dict(sentinel.get("taint_trace") or {})
        local_assessment = state.usage.get("local_context_assessment") or {}
        verification = state.verification or {}
        taint.update({
            "risky_input": bool((sentinel.get("input_decision") or {}).get("findings")),
            "local_context_insufficient": bool(
                local_assessment
                and not local_assessment.get("has_relevant_local_context", True)
            ),
            "general_knowledge_fallback_used": bool(state.usage.get("general_knowledge_fallback_used")),
            "allow_general_knowledge_fallback": bool(state.usage.get("allow_general_knowledge_fallback")),
            "repair_attempts": state.repair_attempts,
            "verification_passed": verification.get("passed"),
            "memory_write_requested": memory_write_requested,
            "tool_call_requested": bool(state.tool_call_count or state.usage.get("mcp_tool_call_count")),
            "low_confidence_count": len([
                item for item in state.decisions
                if str(item.get("confidence") or "").lower() in {"low", "weak"}
            ]),
        })
        if sentinel.get("rag_sanitize_decisions"):
            taint.setdefault("risky_rag_chunks", [])
        if sentinel.get("tool_output_decisions"):
            taint.setdefault("risky_tool_outputs", [])
        if sentinel.get("memory_context_decisions"):
            taint.setdefault("risky_memory_contexts", [])
        sentinel["taint_trace"] = taint
        return taint

    def _build_behavior_trace(self, state: AgentState, memory_write_requested: bool = False) -> Dict[str, Any]:
        high_risk_tool_names = {
            "send_email",
            "deploy_service",
            "delete_resource",
            "file_write",
            "github_write",
            "run_command",
            "filesystem",
            "sqlite",
        }
        tool_actions = []
        for observation in state.tool_observations:
            role = str(observation.get("role") or "")
            tool_name = str(observation.get("tool") or observation.get("name") or "")
            if not tool_name and role.startswith("mcp:"):
                tool_name = role.split(".")[-1]
            if tool_name:
                tool_actions.append({"tool_name": tool_name, "content": observation.get("content", "")})
        sentinel = state.usage.get("sentinel") or {}
        blocked_attempts = len([
            item for item in sentinel.get("tool_decisions", []) or []
            if not item.get("allowed", True)
        ])
        high_risk_count = sum(1 for item in tool_actions if str(item.get("tool_name", "")).lower() in high_risk_tool_names)
        return {
            "session_id": state.session_id,
            "user_goal": state.original_question or state.question,
            "agent_actions": tool_actions,
            "tool_calls": tool_actions,
            "tool_call_count": state.tool_call_count,
            "high_risk_tool_count": high_risk_count,
            "blocked_attempts": blocked_attempts,
            "memory_writes": len(state.writeback_events) + (1 if memory_write_requested else 0),
            "memory_write_count": len(state.writeback_events) + (1 if memory_write_requested else 0),
            "external_actions": sum(
                1 for item in tool_actions
                if str(item.get("tool_name", "")).lower() in {"send_email", "deploy_service", "github_write", "run_command"}
            ),
            "user_context": {"user_explicitly_requested": False, "confirmed": False},
        }

    def _fit_context_budget(self, state: AgentState, context_top_k: Optional[int], max_tokens: int):
        built_context = state.built_context
        if getattr(built_context, "token_estimate", 0) <= max_tokens:
            return built_context

        memories = list(state.usage.get("context_memories", []))
        top_k = context_top_k if context_top_k is not None else self.config.rag_context_top_k
        unlimited_documents = top_k <= 0
        while (memories or unlimited_documents or top_k > 1) and getattr(built_context, "token_estimate", 0) > max_tokens:
            if memories:
                memories = memories[: max(0, len(memories) - 1)]
            elif unlimited_documents:
                top_k = max(1, min(len(state.ranked_chunks), self.config.rag_context_top_k))
                unlimited_documents = False
            elif top_k > 1:
                top_k -= 1
            built_context = self.agent.context_builder.build_context(
                question=state.question,
                ranked_chunks=state.ranked_chunks,
                memories=memories,
                top_k=top_k,
                max_tokens=max_tokens,
                allow_general_knowledge_fallback=bool(state.usage.get("allow_general_knowledge_fallback")),
                evidence_status=self._context_evidence_status(state),
            )
        state.usage["context_memories"] = memories
        if getattr(built_context, "token_estimate", 0) > max_tokens:
            state.add_warning("build_context", "上下文仍超过预算，已保留最小可用上下文。")
        else:
            state.add_warning("build_context", "上下文超过预算，已裁剪记忆或文档数量。")
        return built_context

    def _infer_intent_and_tools(self, question: str) -> tuple[str, bool, List[str]]:
        allowed_tools: List[str] = []
        lowered = question.lower()
        has_url = re.search(r"https?://[^\s，。)）]+", question) is not None
        if has_url:
            allowed_tools.append("fetch")
        if any(marker in question for marker in ["列出文件", "文件列表", "目录", "本地文件", "有哪些文件"]):
            allowed_tools.append("filesystem")
        if "sqlite" in lowered or "数据库表" in question or "有哪些表" in question:
            allowed_tools.append("sqlite")
        if "github" in lowered or "仓库" in question:
            allowed_tools.append("github")

        if "fetch" in allowed_tools:
            return "web_tool", True, allowed_tools
        if "filesystem" in allowed_tools:
            return "file_tool", True, allowed_tools
        if "sqlite" in allowed_tools:
            return "sqlite_tool", True, allowed_tools
        if "github" in allowed_tools:
            return "github_tool", True, allowed_tools
        return "private_qa", False, allowed_tools

    def _call_rule_mcp_tools(self, state: AgentState, decision: Optional[LoopDecision] = None) -> None:
        try:
            if state.usage.get("online_search_fallback") and hasattr(self.agent.mcp_manager, "retrieve_online_context"):
                contexts = self.agent.mcp_manager.retrieve_online_context(state.question)
            else:
                contexts = self.agent.mcp_manager.retrieve_context(state.question)
        except Exception as exc:
            state.add_error("call_mcp_tools", exc)
            contexts = [{
                "role": "mcp_error",
                "type": "mcp",
                "content": f"MCP 上下文获取失败：{exc}",
                "metadata": {"success": False, "error": str(exc)},
            }]
        for context in contexts:
            observation = dict(context)
            observation.setdefault("type", "mcp")
            observation = self._sanitize_tool_observation(state, observation)
            state.tool_observations.append(observation)
        state.tool_call_count += len(contexts)

    def _call_agentic_mcp_tools(self, state: AgentState) -> None:
        client = self.agent._get_llm_client()
        if client is None:
            self._call_rule_mcp_tools(state)
            return

        try:
            tools = self.agent.mcp_manager.get_openai_tools()
        except Exception as exc:
            state.add_error("call_mcp_tools.list_tools", exc)
            return
        if not tools:
            return

        messages = [
            {
                "role": "system",
                "content": "You are the MCP tool planner for a private-domain QA assistant. Select MCP tools only when external files, web pages, or database information are needed.",
            },
            {
                "role": "user",
                "content": (
                    f"Original user question: {state.original_question or state.raw_input}\n"
                    f"English retrieval question: {state.question}\n"
                    f"Allowed tool types: {', '.join(state.query_plan.allowed_tools if state.query_plan else []) or 'auto'}\n"
                    f"Retrieved document count: {len(state.ranked_chunks)}\n"
                    f"Retrieved memory types: {', '.join(state.retrieved_memories.keys()) if state.retrieved_memories else 'none'}\n"
                    f"Local context assessment: {json.dumps(state.usage.get('local_context_assessment', {}), ensure_ascii=False)}\n"
                    "If local context relevance is insufficient, prefer an available web-search MCP tool and use the original user question as the search query.\n"
                    "If no tool is needed, respond with a short no-tool-needed note."
                ),
            },
        ]
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=min(512, self.config.qa_max_answer_tokens),
                **self.config.chat_completion_kwargs(),
            )
        except Exception as exc:
            state.add_error("call_mcp_tools.plan", exc)
            return

        message = response.choices[0].message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        for tool_call in tool_calls[: self.config.agent_loop_max_tool_calls - state.tool_call_count]:
            function = getattr(tool_call, "function", None)
            function_name = getattr(function, "name", "")
            arguments = getattr(function, "arguments", "{}")
            if self._sentinel_enabled():
                decision = self.agent.security_manager.check_tool_call(
                    function_name,
                    args=arguments,
                    user_context={"session_id": state.session_id},
                )
                sentinel = state.usage.setdefault("sentinel", self.agent.security_manager.trace_base())
                sentinel.setdefault("tool_decisions", []).append(decision.to_dict())
                rogue_decision = self.agent.security_manager.check_rogue_action(
                    state.question,
                    {"tool_name": function_name, "tool_args": arguments},
                    user_context={"session_id": state.session_id},
                )
                if rogue_decision.findings:
                    sentinel.setdefault("rogue_tool_decisions", []).append(rogue_decision.to_dict())
                    state.add_warning("sentinel.rogue_agent_guard", "工具调用存在目标漂移或自治越权风险。")
                    decision = self.agent.security_manager._merge_decisions([decision, rogue_decision])
                if self.agent.security_manager.compute_session_risk_score(state.session_id) >= 0.85:
                    session_decision = self.agent.security_manager.check_rogue_trace({
                        "user_goal": state.question,
                        "tool_call_count": state.tool_call_count + 1,
                        "high_risk_tool_count": 1,
                        "agent_actions": [{"tool_name": function_name, "tool_args": arguments}],
                    })
                    sentinel.setdefault("rogue_session_decisions", []).append(session_decision.to_dict())
                    decision = self.agent.security_manager._merge_decisions([decision, session_decision])
                cascade_decision = self.agent.security_manager.check_cascade_tool_decision(
                    {"tool_name": function_name, "arguments": arguments},
                    upstream_risks=sentinel.get("taint_trace", {}),
                )
                if cascade_decision.findings:
                    sentinel.setdefault("cascade_tool_decisions", []).append(cascade_decision.to_dict())
                    state.add_warning("sentinel.cascade_guard", "工具调用受到上游风险影响，已要求重新确认或阻断。")
                    decision = self.agent.security_manager._merge_decisions([decision, cascade_decision])
                if not decision.allowed:
                    blocked_observation = {
                        "role": "sentinel:tool_policy",
                        "type": "sentinel",
                        "content": f"LLM-Sentinel 已阻止工具调用 {function_name}，策略动作：{decision.action}",
                        "metadata": decision.to_dict(),
                    }
                    state.tool_observations.append(blocked_observation)
                    continue
            result = self.agent.mcp_manager.call_openai_tool(function_name, arguments)
            state.tool_call_count += 1
            observation = {
                "role": f"mcp:{result.server}.{result.tool}",
                "type": "mcp_tool",
                "content": result.content if result.success else f"MCP 工具调用失败：{result.error or result.content}",
                "metadata": {
                    "server": result.server,
                    "tool": result.tool,
                    "function": function_name,
                    "arguments": self._parse_tool_arguments(arguments),
                    "success": result.success,
                    "error": result.error,
                },
            }
            observation = self._sanitize_tool_observation(state, observation)
            state.tool_observations.append(observation)

    def _llm_decide_next_action(self, state: AgentState, include_mcp: bool) -> Optional[LoopDecision]:
        client = self.agent._get_llm_client()
        if client is None:
            return None

        prompt = {
            "question": state.question,
            "intent": state.query_plan.intent if state.query_plan else "unknown",
            "need_mcp": bool(include_mcp and state.query_plan and state.query_plan.need_mcp),
            "allowed_tools": state.query_plan.allowed_tools if state.query_plan else [],
            "document_count": len(state.ranked_chunks),
            "memory_types": list(state.retrieved_memories.keys()),
            "skill_context_count": len(state.skill_contexts),
            "tool_observation_count": len(state.tool_observations),
            "loop_step": state.loop_step,
            "max_loop_steps": state.max_loop_steps,
            "remaining_tool_calls": max(0, self.config.agent_loop_max_tool_calls - state.tool_call_count),
        }
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the next-step planner for a private-domain QA assistant. Return JSON only. "
                            "The action must be one of: call_tools, build_context, retrieve_more_rag, retrieve_more_memory, finalize_insufficient_context, stop_max_steps."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Decide the next step from the state. Return this shape: "
                            "{\"action\":\"build_context\",\"tool_requests\":[],\"reason\":\"...\"}\n"
                            f"State: {json.dumps(prompt, ensure_ascii=False)}"
                        ),
                    },
                ],
                temperature=0,
                max_tokens=300,
                **self.config.chat_completion_kwargs(),
            )
            raw = response.choices[0].message.content
            data = self._parse_json_object(raw)
        except Exception as exc:
            state.add_error("decide_next_action.llm_planner", exc)
            return None

        action = self._normalize_planner_action(data.get("action", "build_context"))
        allowed_actions = {
            "call_tools",
            "build_context",
            "retrieve_more_rag",
            "retrieve_more_memory",
            "finalize_insufficient_context",
            "stop_max_steps",
        }
        if action not in allowed_actions:
            action = "build_context"
        if action == "call_tools" and (
            not include_mcp
            or not state.query_plan
            or not state.query_plan.need_mcp
            or state.tool_call_count >= self.config.agent_loop_max_tool_calls
        ):
            action = "build_context"
        return LoopDecision(
            action=action,
            tool_requests=self._normalize_tool_requests(data.get("tool_requests", [])),
            reason=str(data.get("reason", "LLM planner decision")),
        )

    def _parse_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            return {"input": str(arguments)}
        return parsed if isinstance(parsed, dict) else {"input": parsed}

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}

    def _context_evidence_status(self, state: AgentState) -> Dict[str, Any]:
        status = dict(state.usage.get("local_context_assessment", {}) or {})
        retrieval_diagnostics = state.usage.get("retrieval_diagnostics", {}) or {}
        status.update({
            "original_question": state.original_question or state.raw_input,
            "english_question": state.question,
            "response_language": state.response_language or "Chinese",
            "retrieval_diagnostics": retrieval_diagnostics,
            "entity_profile": retrieval_diagnostics.get("entity_profile", {}),
            "entity_coverage": state.usage.get("entity_coverage") or retrieval_diagnostics.get("entity_coverage", {}),
            "active_source_refs": state.active_source_refs,
            "active_source_files": state.usage.get("active_source_files", []),
        })
        return status

    def _normalize_planner_action(self, action: Any) -> str:
        raw = str(action or "build_context").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "tool": "call_tools",
            "tools": "call_tools",
            "call_tool": "call_tools",
            "answer": "build_context",
            "generate_answer": "build_context",
            "context": "build_context",
            "rag": "retrieve_more_rag",
            "retrieve_rag": "retrieve_more_rag",
            "memory": "retrieve_more_memory",
            "retrieve_memory": "retrieve_more_memory",
            "insufficient": "finalize_insufficient_context",
            "stop": "stop_max_steps",
        }
        return aliases.get(raw, raw)

    def _normalize_tool_requests(self, tool_requests: Any) -> List[Dict[str, Any]]:
        if isinstance(tool_requests, dict):
            tool_requests = [tool_requests]
        if not isinstance(tool_requests, list):
            return []
        normalized = []
        for item in tool_requests:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

    def _trim_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _agentic_mcp_enabled(self) -> bool:
        return bool(
            self.agent.mcp_manager is not None
            and self.config.mcp_enabled
            and self.config.mcp_tool_calling_enabled
        )

    def _online_search_available(self) -> bool:
        mcp_manager = getattr(self.agent, "mcp_manager", None)
        if mcp_manager is None:
            return False
        if hasattr(mcp_manager, "online_search_available"):
            try:
                return bool(mcp_manager.online_search_available())
            except Exception:
                return False
        return hasattr(mcp_manager, "retrieve_online_context")

    def _has_successful_tool_observation(self, state: AgentState) -> bool:
        for observation in state.tool_observations:
            metadata = observation.get("metadata", {}) or {}
            if metadata.get("success", True):
                content = (observation.get("content") or "").strip()
                if content and not content.startswith("MCP 调用失败"):
                    return True
        return False

    def _should_repair(self, state: AgentState) -> bool:
        if state.repair_attempts >= self.config.agent_loop_max_repair_attempts:
            return False
        verification = state.verification or {}
        return bool(verification.get("need_repair"))

    def _fuse_session_summary_into_working_memory(self, state: AgentState, summary_memory: Dict[str, Any]) -> None:
        if not self.config.session_summary_fuse_memory:
            return
        memory_manager = getattr(self.agent, "memory_manager", None)
        working_memory = getattr(memory_manager, "working_memory", None)
        if working_memory is None:
            return
        content = summary_memory.get("content", "")
        if not content:
            return
        try:
            existing = []
            if hasattr(working_memory, "get_session_history"):
                existing = working_memory.get_session_history(state.session_id)
            for item in existing:
                if item.get("role") == "transcript_summary" and item.get("content") == content:
                    return
            working_memory.add_memory(
                state.session_id,
                "transcript_summary",
                content,
                importance=6,
            )
            state.usage["session_summary_fused"] = True
        except Exception as exc:
            state.add_error("session_summary_fusion", exc)

    def save_checkpoint(self, state: AgentState, node: str) -> None:
        try:
            checkpoint_id = self.checkpoint_store.save(
                session_id=state.session_id,
                node=node,
                snapshot=self._checkpoint_snapshot(state, node=node),
            )
            if checkpoint_id is not None:
                state.usage["last_checkpoint_id"] = checkpoint_id
        except Exception as exc:
            state.add_error("save_checkpoint", exc)

    def _checkpoint_snapshot(self, state: AgentState, node: str) -> Dict[str, Any]:
        return {
            "node": node,
            "session_id": state.session_id,
            "raw_input": state.raw_input,
            "question": state.question,
            "intent": state.query_plan.intent if state.query_plan else None,
            "loop_step": state.loop_step,
            "stop_reason": state.stop_reason,
            "candidate_count": len(state.candidates),
            "ranked_count": len(state.ranked_chunks),
            "memory_types": list(state.retrieved_memories.keys()),
            "skill_context_count": len(state.skill_contexts),
            "tool_observation_count": len(state.tool_observations),
            "answer": state.answer,
            "verification": state.verification,
            "errors": state.errors,
            "warnings": state.warnings,
            "decisions": state.decisions,
        }
