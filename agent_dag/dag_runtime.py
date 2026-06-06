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

        writer_result = await self._run_node("writer", base_context, privileged_keys=list(privileged))
        trace.add_node(writer_result)
        writer_output = writer_result.output or {}
        state.answer = str(writer_output.get("answer") or "")
        state.built_context = writer_output.get("built_context")
        state.ranked_chunks = list(writer_output.get("ranked_chunks") or state.ranked_chunks)
        state.usage["context_memories"] = list(writer_output.get("memories") or state.usage.get("context_memories", []))
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
        return state

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
