from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from agent.answer_verifier import AnswerVerifier
from agent_dag.dag_context import AgentContext
from agent_dag.dag_node import DAGNode, NodeResult
from agent_dag.evidence_store import EvidenceItem, SharedEvidenceStore
from rag.rag_query import QUERY_MODE_BASIC
from sentinel.runtime.evidence_policy import ShareLevel


class PlannerNode(DAGNode):
    async def _run(self, context: AgentContext) -> Any:
        agent = context.privileged_context["agent"]
        understanding = agent.understand_query(context.user_query)
        required = ["rag"]
        if context.privileged_context.get("include_memory_manager", True):
            required.append("memory")
        if context.privileged_context.get("include_skills", True):
            required.append("skill")
        if context.privileged_context.get("include_mcp", True) and agent.mcp_manager is not None:
            required.append("web")
        required.append("security")
        return {
            "query_understanding": understanding,
            "required_nodes": required,
            "node_dependencies": {
                "writer": [node for node in required if node != "security"],
                "verifier": ["writer"],
                "memory_writer": ["verifier"],
            },
        }


class RAGRetrieverNode(DAGNode):
    async def run(self, context: AgentContext) -> NodeResult:
        result = await super().run(context)
        result.metadata.setdefault("share_level", ShareLevel.CITED_EVIDENCE.value)
        result.metadata.setdefault("trust_level", "medium")
        return result

    async def _run(self, context: AgentContext) -> Any:
        agent = context.privileged_context["agent"]
        understanding = context.privileged_context.get("query_understanding") or agent.understand_query(context.user_query)
        modes = context.privileged_context.get("modes") or understanding.modes or (QUERY_MODE_BASIC,)
        candidates = agent.query_engine.search(
            question=context.user_query,
            modes=modes,
            top_k=context.privileged_context.get("top_k"),
            per_query_limit=context.privileged_context.get("per_query_limit"),
            mqe_count=context.privileged_context.get("mqe_count"),
        )
        ranked = agent.reranker.rerank(
            question=context.user_query,
            candidates=candidates,
            top_k=context.privileged_context.get("rerank_top_k") or agent.config.rag_rerank_top_k,
        )
        decisions = []
        if agent.security_manager is not None:
            ranked, decisions = agent.security_manager.sanitize_ranked_chunks(ranked)
        return {
            "candidates": candidates,
            "ranked_chunks": ranked,
            "sentinel_decisions": decisions,
            "content": f"Retrieved {len(ranked)} ranked RAG chunks.",
        }


class MemoryRetrieverNode(DAGNode):
    async def run(self, context: AgentContext) -> NodeResult:
        result = await super().run(context)
        result.metadata.setdefault("share_level", ShareLevel.PRIVILEGED.value)
        result.metadata.setdefault("trust_level", "medium")
        return result

    async def _run(self, context: AgentContext) -> Any:
        agent = context.privileged_context["agent"]
        if agent.memory_manager is None:
            return {"memories": [], "content": "No memory manager is configured."}
        retrieved = agent.memory_manager.retrieve_memories(
            query=context.user_query,
            session_id=context.user_context.get("session_id"),
            top_k=context.privileged_context.get("memory_top_k", 3),
        )
        memories = agent._flatten_retrieved_memories(retrieved)
        decisions = []
        if agent.security_manager is not None:
            memories, decisions = agent.security_manager.filter_memory_contexts(memories)
        return {
            "retrieved_memories": retrieved,
            "memories": memories,
            "sentinel_decisions": decisions,
            "content": f"Retrieved {len(memories)} memory items.",
        }


class SkillNode(DAGNode):
    async def run(self, context: AgentContext) -> NodeResult:
        result = await super().run(context)
        result.metadata.setdefault("share_level", ShareLevel.SANITIZED_FACTS.value)
        result.metadata.setdefault("trust_level", "medium")
        return result

    async def _run(self, context: AgentContext) -> Any:
        agent = context.privileged_context["agent"]
        if agent.skill_manager is None:
            return {"skill_contexts": [], "content": "No skills configured."}
        skill_contexts = list(agent.skill_manager.retrieve_context(context.user_query))
        return {
            "skill_contexts": skill_contexts,
            "content": f"Retrieved {len(skill_contexts)} skill contexts.",
        }


class WebSearchNode(DAGNode):
    async def run(self, context: AgentContext) -> NodeResult:
        result = await super().run(context)
        result.metadata.setdefault("share_level", ShareLevel.SANITIZED_FACTS.value)
        result.metadata.setdefault("trust_level", "low")
        result.metadata.setdefault("taint_labels", ["external_web", "tool_output_untrusted"])
        return result

    async def _run(self, context: AgentContext) -> Any:
        agent = context.privileged_context["agent"]
        if agent.mcp_manager is None:
            return {"web_contexts": [], "content": "No MCP manager configured."}
        if hasattr(agent.mcp_manager, "retrieve_online_context"):
            contexts = agent.mcp_manager.retrieve_online_context(context.user_query)
        elif hasattr(agent.mcp_manager, "retrieve_context"):
            contexts = agent.mcp_manager.retrieve_context(context.user_query)
        else:
            contexts = []
        sanitized = []
        decisions = []
        for item in contexts or []:
            if agent.security_manager is not None:
                clean, decision = agent.security_manager.sanitize_tool_observation(dict(item))
                sanitized.append(clean)
                if decision.findings:
                    decisions.append(decision.to_dict())
            else:
                sanitized.append(item)
        return {
            "web_contexts": sanitized,
            "sentinel_decisions": decisions,
            "content": f"Retrieved {len(sanitized)} web contexts.",
        }


class SecurityNode(DAGNode):
    async def _run(self, context: AgentContext) -> Any:
        return {
            "risk_summary": {
                "taint_labels": list(context.taint_labels),
                "permissions": list(context.permissions),
                "allowed_tools": list(context.allowed_tools),
            },
            "content": "Security audit completed.",
        }


class WriterNode(DAGNode):
    async def _run(self, context: AgentContext) -> Any:
        agent = context.privileged_context["agent"]
        evidence_store: SharedEvidenceStore = context.privileged_context["evidence_store"]
        ranked_chunks = []
        memories = []
        for item in evidence_store.visible_to("writer"):
            if item.source_type == "retriever":
                ranked_chunks.extend(item.metadata.get("ranked_chunks", []))
            elif item.source_type in {"memory", "skill", "external_retriever"}:
                memories.extend(item.metadata.get("memories", []) or item.metadata.get("skill_contexts", []) or item.metadata.get("web_contexts", []))
        built_context = agent.context_builder.build_context(
            question=context.user_query,
            ranked_chunks=ranked_chunks,
            memories=memories,
            top_k=context.privileged_context.get("context_top_k"),
            allow_general_knowledge_fallback=True,
            evidence_status={
                "original_question": context.privileged_context.get("original_question") or context.user_query,
                "response_language": context.privileged_context.get("response_language") or "Chinese",
                "runtime": "dag",
            },
        )
        try:
            answer = agent.generate_answer(built_context, stream_callback=context.privileged_context.get("stream_callback"))
        except Exception:
            answer = agent._fallback_answer(built_context)
        return {
            "answer": answer,
            "built_context": built_context,
            "ranked_chunks": ranked_chunks,
            "memories": memories,
            "content": answer,
        }


class VerifierNode(DAGNode):
    async def _run(self, context: AgentContext) -> Any:
        writer_output = context.privileged_context.get("writer_output") or {}
        answer = str(writer_output.get("answer") or "")
        built_context = writer_output.get("built_context")
        verifier = AnswerVerifier()
        verification = verifier.verify(answer, built_context, query_plan=None).to_dict() if built_context is not None else {"passed": bool(answer)}
        return {
            "verification": verification,
            "content": "Verification completed.",
        }


class MemoryWriterNode(DAGNode):
    async def _run(self, context: AgentContext) -> Any:
        agent = context.privileged_context["agent"]
        if agent.memory_manager is None or not context.privileged_context.get("write_memory"):
            return {"skipped": True, "reason": "memory writing disabled"}
        writer_output = context.privileged_context.get("writer_output") or {}
        verification = context.privileged_context.get("verification") or {}
        answer = str(writer_output.get("answer") or "")
        if not verification.get("passed", False):
            return {"skipped": True, "reason": "verification failed"}
        allowed, decisions = agent.security_manager.should_write_memory_with_trace(
            context.user_query,
            answer,
            context_trace={"memory_write_requested": True, "verification_passed": True},
        )
        if not allowed:
            return {"skipped": True, "reason": "sentinel blocked memory write", "decisions": decisions}
        consolidations = agent.memory_manager.add_interaction(
            context.user_context.get("session_id") or "default",
            context.user_query,
            answer,
        )
        fact_result = agent.memory_manager.process_high_value_qa(
            context.user_query,
            answer,
            session_id=context.user_context.get("session_id") or "default",
        )
        return {
            "skipped": False,
            "working_memory": [
                asdict(item) if is_dataclass(item) else item
                for item in (consolidations or [])
            ],
            "semantic_fact": asdict(fact_result) if is_dataclass(fact_result) else fact_result,
        }
