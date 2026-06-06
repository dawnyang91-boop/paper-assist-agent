from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from sentinel.config import SentinelConfig
from sentinel.defenses.cascade_guard import CascadeGuard
from sentinel.defenses.code_execution_guard import CodeExecutionGuard
from sentinel.defenses.human_trust_guard import HumanTrustGuard
from sentinel.defenses.identity_guard import IdentityGuard
from sentinel.defenses.inter_agent_guard import InterAgentGuard
from sentinel.defenses.input_guard import InputGuard
from sentinel.defenses.memory_safety import MemorySafetyChecker
from sentinel.defenses.output_guard import OutputGuard
from sentinel.defenses.rag_sanitizer import RAGSanitizer
from sentinel.defenses.rogue_agent_guard import RogueAgentGuard
from sentinel.defenses.supply_chain_guard import SupplyChainGuard
from sentinel.defenses.tool_output_sanitizer import ToolOutputSanitizer
from sentinel.defenses.tool_policy import ToolPolicyChecker
from sentinel.detectors.llm_judge import LLMJudge
from sentinel.schemas import SecurityDecision


class SecurityManager:
    def __init__(self, config: SentinelConfig | None = None):
        self.config = config or SentinelConfig()
        self.input_guard = InputGuard(self.config)
        self.rag_sanitizer = RAGSanitizer(self.config)
        self.tool_policy = ToolPolicyChecker(self.config)
        self.tool_policy.security_manager = self
        self.tool_output_sanitizer = ToolOutputSanitizer(self.config)
        self.memory_safety = MemorySafetyChecker(self.config)
        self.output_guard = OutputGuard(self.config)
        self.identity_guard = IdentityGuard(self.config)
        self.code_execution_guard = CodeExecutionGuard(self.config)
        self.human_trust_guard = HumanTrustGuard(self.config)
        self.supply_chain_guard = SupplyChainGuard(self.config)
        self.cascade_guard = CascadeGuard(self.config)
        self.inter_agent_guard = InterAgentGuard(self.config)
        self.rogue_agent_guard = RogueAgentGuard(self.config)
        self.llm_judge = LLMJudge(config=self.config)

    @classmethod
    def from_app_config(cls, app_config: Any) -> "SecurityManager":
        return cls(SentinelConfig.from_app_config(app_config))

    def pre_check_user_input(self, text: str) -> SecurityDecision:
        input_decision = self.input_guard.check(text)
        identity_decision = self.identity_guard.check_text(text, source="user_input")
        return self._merge_decisions([input_decision, identity_decision])

    def sanitize_retrieved_docs(self, docs: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self.rag_sanitizer.sanitize_docs(docs)

    def sanitize_ranked_chunks(self, ranked_chunks: Iterable[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
        return self.rag_sanitizer.sanitize_ranked_chunks(ranked_chunks)

    def check_tool_call(self, tool_name: str, args: Any = None, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        return self.tool_policy.check(tool_name, args=args, user_context=user_context)

    def check_identity_context(self, text: str, user_context: Dict[str, Any] | None = None, source: str = "untrusted") -> SecurityDecision:
        return self.identity_guard.check_text(text, user_context=user_context, source=source)

    def check_privileged_tool_call(self, tool_name: str, args: Any = None, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        return self.check_tool_call(tool_name, args=args, user_context=user_context)

    def check_code_execution(self, tool_name: str, args: Any = None, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        parsed_args = args if isinstance(args, dict) else {"input": args}
        return self.code_execution_guard.check_tool_args(tool_name, parsed_args)

    def check_confirmation_text(self, tool_call: Dict[str, Any], confirmation_text: str) -> SecurityDecision:
        return self.human_trust_guard.check_confirmation_text(tool_call, confirmation_text)

    def check_human_trust_output(self, answer: str, proposed_action: Dict[str, Any] | None = None) -> SecurityDecision:
        return self.human_trust_guard.check_risk_downplaying(answer, proposed_action=proposed_action)

    def check_tool_descriptor(self, tool_schema: Dict[str, Any]) -> SecurityDecision:
        return self.supply_chain_guard.check_tool_descriptor(tool_schema)

    def check_skill_output(self, skill_name: str, output: str) -> SecurityDecision:
        return self.supply_chain_guard.check_skill_output(skill_name, output)

    def check_supply_chain_artifact(self, text: str, artifact_type: str = "supply_chain_artifact") -> SecurityDecision:
        return self.supply_chain_guard.check_text(text, artifact_type=artifact_type)

    def sanitize_tool_output(self, text: str) -> SecurityDecision:
        return self._merge_decisions([
            self.tool_output_sanitizer.sanitize_text(text),
            self.cascade_guard.check_tool_output_cascade(text),
        ])

    def sanitize_tool_observation(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], SecurityDecision]:
        return self.tool_output_sanitizer.sanitize_observation(observation)

    def check_cascade_answer(self, answer: str, context_trace: Dict[str, Any] | None = None) -> SecurityDecision:
        return self.cascade_guard.check_answer_taint(answer, context_trace=context_trace)

    def check_cascade_memory_write(
        self,
        question: str,
        answer: str,
        context_trace: Dict[str, Any] | None = None,
    ) -> SecurityDecision:
        return self.cascade_guard.check_memory_write_taint(question, answer, context_trace=context_trace)

    def check_cascade_tool_decision(
        self,
        tool_call: Dict[str, Any] | str,
        upstream_risks: Iterable[Dict[str, Any]] | Dict[str, Any] | None = None,
    ) -> SecurityDecision:
        return self.cascade_guard.check_tool_decision_taint(tool_call, upstream_risks=upstream_risks)

    def check_cascade_trace(self, trace: Dict[str, Any] | None = None) -> SecurityDecision:
        return self.cascade_guard.check_trace(trace)

    def check_agent_message(self, message: Dict[str, Any]) -> SecurityDecision:
        return self.inter_agent_guard.check_message(message)

    def check_node_output(self, node_id: str, output: str, target_receivers: Iterable[str] | None = None) -> SecurityDecision:
        return self._merge_decisions([
            self.rag_sanitizer.sanitize_text(output),
            self.supply_chain_guard.check_text(output, artifact_type=f"dag_node_output:{node_id}"),
            self.cascade_guard.check_answer_taint(
                output,
                context_trace={"answer_used_risky_context": False, "target_receivers": list(target_receivers or [])},
            ),
        ])

    def check_evidence_write(self, evidence_item: Dict[str, Any]) -> SecurityDecision:
        content = str(evidence_item.get("sanitized_content") or evidence_item.get("content") or "")
        share_level = str(evidence_item.get("share_level") or "")
        if share_level == "never_share":
            return SecurityDecision(
                allowed=False,
                action="block",
                risk_score=1.0,
                findings=[],
                metadata={"reason": "never_share evidence cannot be written to SharedEvidenceStore"},
            )
        return self.rag_sanitizer.sanitize_text(content)

    def check_dag_tool_action(
        self,
        node_id: str,
        tool_name: str,
        tool_args: Any = None,
        context: Dict[str, Any] | None = None,
    ) -> SecurityDecision:
        context = context or {}
        return self._merge_decisions([
            self.check_tool_call(tool_name, args=tool_args, user_context=context.get("user_context", {})),
            self.check_rogue_action(context.get("user_goal", ""), {"tool_name": tool_name, "tool_args": tool_args}, user_context=context.get("user_context", {})),
            self.check_cascade_tool_decision({"tool_name": tool_name, "arguments": tool_args}, upstream_risks=context.get("upstream_risks", {})),
        ])

    def check_dag_memory_write(self, question: str, answer: str, trace: Dict[str, Any] | None = None) -> SecurityDecision:
        return self._merge_decisions([
            self.check_memory_candidate(answer),
            self.check_cascade_memory_write(question, answer, context_trace=trace),
            self.check_rogue_trace(trace),
        ])

    def sanitize_agent_message_output(self, message: Dict[str, Any]) -> SecurityDecision:
        return self.inter_agent_guard.sanitize_agent_output(message)

    def check_rogue_action(self, user_goal: str, action: Dict[str, Any] | str, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        return self._merge_decisions([
            self.rogue_agent_guard.check_goal_alignment(user_goal, action),
            self.rogue_agent_guard.check_autonomy_violation(action, user_context=user_context),
            self.rogue_agent_guard.check_self_modification(action),
        ])

    def check_rogue_trace(self, trace: Dict[str, Any] | None = None) -> SecurityDecision:
        return self.rogue_agent_guard.check_behavior_trace(trace)

    def update_behavior_profile(self, session_id: str, trace: Dict[str, Any]) -> Dict[str, Any]:
        return self.rogue_agent_guard.update_behavior_profile(session_id, trace)

    def compute_session_risk_score(self, session_id: str) -> float:
        return self.rogue_agent_guard.compute_session_risk_score(session_id)

    def filter_memory_contexts(self, memories: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self.memory_safety.filter_contexts(memories)

    def check_memory_candidate(self, text: str) -> SecurityDecision:
        return self.memory_safety.check_text(text)

    def should_write_memory_interaction(self, question: str, answer: str) -> Tuple[bool, List[Dict[str, Any]]]:
        return self.memory_safety.should_write_interaction(question, answer)

    def should_write_memory_with_trace(
        self,
        question: str,
        answer: str,
        context_trace: Dict[str, Any] | None = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        memory_allowed, memory_decisions = self.memory_safety.should_write_interaction(question, answer)
        cascade_decision = self.check_cascade_memory_write(question, answer, context_trace=context_trace)
        decisions = list(memory_decisions)
        if cascade_decision.findings:
            decisions.append(cascade_decision.to_dict())
        allowed = memory_allowed and cascade_decision.allowed
        return allowed, decisions

    def post_check_output(self, answer: str) -> SecurityDecision:
        return self._merge_decisions([
            self.output_guard.check(answer),
            self.human_trust_guard.check_risk_downplaying(answer),
            self.cascade_guard.check_answer_taint(answer),
        ])

    def judge_case(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        return self.llm_judge.judge(payload)

    def trace_base(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "input_guard_enabled": self.config.input_guard_enabled,
            "rag_guard_enabled": self.config.rag_guard_enabled,
            "tool_guard_enabled": self.config.tool_guard_enabled,
            "output_guard_enabled": self.config.output_guard_enabled,
            "tool_output_guard_enabled": self.config.tool_guard_enabled,
            "memory_guard_enabled": self.config.enabled,
            "cascade_guard_enabled": self.config.enabled,
            "inter_agent_guard_enabled": self.config.enabled,
            "rogue_agent_guard_enabled": self.config.enabled,
            "llm_judge_enabled": self.config.llm_judge_enabled,
            "blocked": False,
        }

    def _merge_decisions(self, decisions: Iterable[SecurityDecision]) -> SecurityDecision:
        decisions = list(decisions)
        findings = [finding for decision in decisions for finding in decision.findings]
        risk_score = max((decision.risk_score for decision in decisions), default=0.0)
        action_priority = {
            "allow": 0,
            "warn": 1,
            "sanitize": 2,
            "redact": 2,
            "reduce_trust": 2,
            "require_confirmation": 3,
            "skip_memory_write": 4,
            "quarantine": 4,
            "block": 5,
        }
        action = max((decision.action for decision in decisions), key=lambda item: action_priority.get(item, 0), default="allow")
        allowed = all(decision.allowed for decision in decisions) and action not in {"block", "quarantine", "skip_memory_write"}
        sanitized = next((decision.sanitized_text for decision in decisions if decision.sanitized_text), None)
        metadata: Dict[str, Any] = {}
        for decision in decisions:
            metadata.update(decision.metadata or {})
        return SecurityDecision(
            allowed=allowed,
            action=action,
            risk_score=risk_score,
            findings=findings,
            sanitized_text=sanitized,
            metadata=metadata,
        )
