from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List

from agent.agent_state import AgentState, QueryPlan
from agent_dag.dag_context import AgentContext
from agent_dag.dag_policy import node_policy
from agent_dag.edge_guard import EdgeGuard
from agent_dag.evidence_store import SharedEvidenceStore
from agent_dag.node_registry import build_default_registry
from agent_dag.trace import DAGTrace
from rag.provenance import extract_source_refs_from_events, looks_like_follow_up, source_files_from_refs


class DAGAgentRuntime:
    def __init__(self, agent: Any, config: Any):
        self.agent = agent
        self.config = config
        self.nodes = build_default_registry()

    def run_sync(self, payload: Dict[str, Any]) -> AgentState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self.run(payload))
            finally:
                new_loop.close()
        return asyncio.run(self.run(payload))

    async def run(self, payload: Dict[str, Any]) -> AgentState:
        task_id = str(uuid.uuid4())
        question = str(payload.get("question") or "")
        session_id = str(payload.get("session_id") or "default")
        state = AgentState(
            session_id=session_id,
            raw_input=question,
            question=question.strip(),
            original_question=question.strip(),
            max_loop_steps=self.config.agent_loop_max_steps,
        )
        trace = DAGTrace(task_id=task_id)
        evidence_store = SharedEvidenceStore()
        edge_guard = EdgeGuard(self.agent.security_manager, task_id=task_id)

        input_decision = self.agent.security_manager.pre_check_user_input(question)
        state.usage["sentinel"] = self.agent.security_manager.trace_base()
        state.usage["sentinel"]["input_decision"] = input_decision.to_dict()
        if not input_decision.allowed:
            state.answer = "LLM-Sentinel 已阻止该请求。"
            state.stop_reason = "sentinel_blocked"
            state.usage["dag"] = trace.to_dict(evidence_count=0)
            return state

        privileged = {
            "agent": self.agent,
            "evidence_store": evidence_store,
            "include_memory_manager": payload.get("include_memory_manager", True),
            "include_mcp": payload.get("include_mcp", True),
            "include_skills": payload.get("include_skills", True),
            "memory_top_k": payload.get("memory_top_k", 3),
            "modes": payload.get("modes"),
            "top_k": payload.get("top_k"),
            "per_query_limit": payload.get("per_query_limit"),
            "mqe_count": payload.get("mqe_count"),
            "rerank_top_k": payload.get("rerank_top_k"),
            "context_top_k": payload.get("context_top_k"),
            "stream_callback": payload.get("stream_callback"),
            "write_memory": payload.get("write_memory", False),
            "original_question": question,
            "response_language": "Chinese",
        }
        self._load_follow_up_source_refs(payload, state, privileged)
        self._record_transcript(payload, state, role="user", content=question)
        base_context = AgentContext(
            task_id=task_id,
            node_id="root",
            role="root",
            user_query=question,
            user_context={"session_id": session_id},
            privileged_context=privileged,
        )

        planner_result = await self._run_node("planner", base_context, privileged_keys=list(privileged))
        trace.add_node(planner_result)
        plan = planner_result.output or {}
        understanding = plan.get("query_understanding") or self.agent.understand_query(question)
        privileged["query_understanding"] = understanding
        state.query_plan = QueryPlan(
            intent="dag_private_qa",
            need_rag="rag" in plan.get("required_nodes", []),
            need_memory="memory" in plan.get("required_nodes", []),
            need_mcp="web" in plan.get("required_nodes", []),
            need_skills="skill" in plan.get("required_nodes", []),
            retrieval_modes=tuple(payload.get("modes") or understanding.modes),
            answer_style="grounded",
            high_value_candidate=True,
            reason="DAG planner selected retrieval and writing nodes.",
        )
        state.usage["query_understanding"] = understanding

        parallel_ids = [node for node in plan.get("required_nodes", []) if node in {"rag", "memory", "skill", "web", "security"}]
        trace.parallel_groups.append(parallel_ids)
        parallel_results = await asyncio.gather(*[
            self._run_node(node_id, base_context, privileged_keys=list(privileged))
            for node_id in parallel_ids
        ])
        for result in parallel_results:
            allowed, decision, evidence_items = edge_guard.check_result(result, receiver="writer")
            trace.add_edge(result.node_id, "writer", decision)
            if allowed:
                for item in evidence_items:
                    evidence_store.add(item)
            trace.add_node(result, evidence_written=len(evidence_items), output_risk_score=getattr(decision, "risk_score", 0.0))
            self._merge_node_output(state, result)

        if self._merge_memory_provenance_refs(state, privileged):
            rag_result = await self._run_node("rag", base_context, privileged_keys=list(privileged))
            allowed, decision, evidence_items = edge_guard.check_result(rag_result, receiver="writer")
            trace.add_edge("memory", "rag", {"reason": "memory provenance rehydration"})
            trace.add_edge(rag_result.node_id, "writer", decision)
            if allowed:
                for item in evidence_items:
                    evidence_store.add(item)
            trace.add_node(rag_result, evidence_written=len(evidence_items), output_risk_score=getattr(decision, "risk_score", 0.0))
            self._merge_node_output(state, rag_result)

        writer_result = await self._run_node("writer", base_context, privileged_keys=list(privileged))
        trace.add_node(writer_result)
        writer_output = writer_result.output or {}
        state.answer = str(writer_output.get("answer") or "")
        state.built_context = writer_output.get("built_context")
        state.ranked_chunks = list(writer_output.get("ranked_chunks") or state.ranked_chunks)
        state.usage["context_memories"] = list(writer_output.get("memories") or state.usage.get("context_memories", []))
        state.usage["answer_generation"] = dict(writer_output.get("answer_generation") or {})
        privileged["writer_output"] = writer_output

        output_decision = self.agent.security_manager.post_check_output(state.answer)
        state.usage["sentinel"]["output_decision"] = output_decision.to_dict()
        if output_decision.sanitized_text and output_decision.sanitized_text != state.answer:
            state.answer = output_decision.sanitized_text

        verifier_result = await self._run_node("verifier", base_context, privileged_keys=list(privileged))
        trace.add_node(verifier_result)
        verification = (verifier_result.output or {}).get("verification") or {"passed": bool(state.answer)}
        state.verification = verification
        privileged["verification"] = verification

        if payload.get("write_memory"):
            memory_writer_result = await self._run_node("memory_writer", base_context, privileged_keys=list(privileged))
            trace.add_node(memory_writer_result)
            state.writeback_events.append({"type": "dag_memory_writer", "result": memory_writer_result.output})

        trace.risk_summary = {
            "blocked_messages": len(trace.blocked_messages),
            "evidence_count": len(evidence_store.all()),
            "sentinel": state.usage.get("sentinel", {}),
        }
        state.stop_reason = "final"
        state.usage["dag"] = trace.to_dict(evidence_count=len(evidence_store.all()))
        state.usage["runtime"] = "dag"
        if state.answer:
            self._record_transcript(
                payload,
                state,
                role="assistant",
                content=state.answer,
                metadata={
                    "stop_reason": state.stop_reason,
                    "verification": state.verification,
                    "references": self.agent._answer_references_from_built_context(state.built_context),
                },
            )
        return state

    def _record_transcript(
        self,
        payload: Dict[str, Any],
        state: AgentState,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        transcript_store = payload.get("transcript_store")
        if transcript_store is None:
            return
        try:
            event = transcript_store.append(
                session_id=state.session_id,
                role=role,
                content=content,
                metadata=metadata,
            )
            if event:
                state.messages.append(event)
        except Exception as exc:
            state.add_error("dag.record_transcript", exc)

    def _load_follow_up_source_refs(
        self,
        payload: Dict[str, Any],
        state: AgentState,
        privileged: Dict[str, Any],
    ) -> None:
        transcript_store = payload.get("transcript_store")
        if transcript_store is None or not getattr(self.config, "transcript_resume_enabled", False):
            return
        try:
            events = transcript_store.load(state.session_id)
        except Exception as exc:
            state.add_error("dag.load_session_state", exc)
            return
        recent_events = events[-max(1, self.config.transcript_resume_max_events):]
        state.resumed_transcript = recent_events
        source_refs = extract_source_refs_from_events(recent_events)
        if source_refs and looks_like_follow_up(state.raw_input):
            refs = [ref.to_dict() for ref in source_refs]
            files = source_files_from_refs(source_refs)
            state.active_source_refs = refs
            state.usage["active_source_refs"] = refs
            state.usage["active_source_files"] = files
            privileged["active_source_refs"] = refs
            privileged["active_source_files"] = files

    def _merge_memory_provenance_refs(self, state: AgentState, privileged: Dict[str, Any]) -> bool:
        refs = list(privileged.get("active_source_refs") or state.active_source_refs or [])
        seen = {
            (item.get("source_file"), item.get("chunk_index"))
            for item in refs
            if isinstance(item, dict)
        }
        before = len(refs)
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
        if len(refs) <= before:
            return False
        files = sorted({
            str(item.get("source_file"))
            for item in refs
            if isinstance(item, dict) and item.get("source_file")
        })
        state.active_source_refs = refs
        state.usage["active_source_refs"] = refs
        state.usage["active_source_files"] = files
        privileged["active_source_refs"] = refs
        privileged["active_source_files"] = files
        return True

    async def _run_node(self, node_id: str, base_context: AgentContext, privileged_keys: List[str]) -> Any:
        node = self.nodes[node_id]
        policy = node_policy(node_id)
        context = base_context.copy_for_node(
            node_id=node_id,
            role=policy.role,
            permissions=policy.permissions,
            allowed_tools=policy.allowed_tools,
            privileged_keys=privileged_keys,
        )
        return await node.run(context)

    def _merge_node_output(self, state: AgentState, result: Any) -> None:
        output = result.output or {}
        if result.node_id == "rag":
            state.candidates = list(output.get("candidates") or [])
            state.ranked_chunks = list(output.get("ranked_chunks") or [])
            state.usage["retrieval_diagnostics"] = dict(output.get("retrieval_diagnostics") or {})
            state.usage["entity_coverage"] = dict(output.get("entity_coverage") or {})
            if output.get("sentinel_decisions"):
                state.usage.setdefault("sentinel", {}).setdefault("rag_sanitize_decisions", []).extend(output["sentinel_decisions"])
        elif result.node_id == "memory":
            state.retrieved_memories = dict(output.get("retrieved_memories") or {})
            state.usage["context_memories"] = list(output.get("memories") or [])
            if output.get("sentinel_decisions"):
                state.usage.setdefault("sentinel", {}).setdefault("memory_context_decisions", []).extend(output["sentinel_decisions"])
        elif result.node_id == "skill":
            state.skill_contexts = list(output.get("skill_contexts") or [])
            current = list(state.usage.get("context_memories", []))
            current.extend(state.skill_contexts)
            state.usage["context_memories"] = current
        elif result.node_id == "web":
            observations = list(output.get("web_contexts") or [])
            state.tool_observations.extend(observations)
            current = list(state.usage.get("context_memories", []))
            current.extend(observations)
            state.usage["context_memories"] = current
