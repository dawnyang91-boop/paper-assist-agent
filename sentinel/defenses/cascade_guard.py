from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding


class CascadeGuard:
    """Detect when upstream risk is being amplified into tools, answers, or memory."""

    HIGH_RISK_ACTION_WORDS = (
        "delete",
        "remove",
        "drop",
        "deploy",
        "rollback",
        "payment",
        "send",
        "write",
        "update",
        "rm -rf",
        "删除",
        "移除",
        "部署",
        "回滚",
        "付款",
        "发送",
        "写入",
        "更新",
    )
    UNSUPPORTED_CERTAINTY_WORDS = (
        "must",
        "guaranteed",
        "definitely",
        "absolutely",
        "无需验证",
        "一定",
        "必须",
        "完全确定",
    )
    LOCAL_CITATION_RE = re.compile(r"\[(?:D|M)\d+\]")

    def __init__(self, config: SentinelConfig):
        self.config = config

    def mark_taint(self, source: str, risk_type: str, object_id: str = "", severity: str = "high") -> Dict[str, Any]:
        return {
            "source": source,
            "risk_type": risk_type,
            "object_id": object_id,
            "severity": severity,
        }

    def check_answer_taint(self, answer: str, context_trace: Dict[str, Any] | None = None) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

        trace = dict(context_trace or {})
        findings: List[SecurityFinding] = []
        upstream_risks = self._upstream_risks(trace)
        answer_text = str(answer or "")
        answer_lower = answer_text.lower()

        if upstream_risks and self._uses_risky_context(answer_text, trace):
            findings.append(self._finding(
                "cascading_failure",
                "high",
                0.82,
                self._preview(answer_text),
                "Do not let risky upstream context become final answer without caveats or verification.",
                {"propagation_path": self.build_propagation_path(trace, sink="final_answer")},
            ))

        local_insufficient = bool(trace.get("local_context_insufficient") or trace.get("allow_general_knowledge_fallback"))
        fallback_used = bool(trace.get("general_knowledge_fallback_used") or trace.get("fallback_used"))
        has_local_citation = bool(self.LOCAL_CITATION_RE.search(answer_text))
        has_disclaimer = any(marker in answer_text for marker in (
            "本地文档",
            "本地知识",
            "私域知识",
            "没有找到足够依据",
            "以下基于通用知识",
            "not found in local",
            "general knowledge",
        ))
        if (local_insufficient or fallback_used) and has_local_citation:
            findings.append(self._finding(
                "general_knowledge_overtrust",
                "high",
                0.84,
                self._preview(answer_text),
                "General-knowledge fallback must not be cited as local document or memory evidence.",
                {"local_citations": self.LOCAL_CITATION_RE.findall(answer_text)},
            ))
        if (local_insufficient or fallback_used) and not has_disclaimer:
            findings.append(self._finding(
                "general_knowledge_overtrust",
                "medium",
                0.68,
                self._preview(answer_text),
                "When local evidence is insufficient, explicitly state that the answer uses general knowledge.",
                {"fallback_used": fallback_used},
            ))

        repair_attempts = int(trace.get("repair_attempts") or 0)
        if repair_attempts > 0 and (trace.get("memory_write_requested") or trace.get("write_memory")):
            findings.append(self._finding(
                "repair_loop_amplification",
                "high",
                0.81,
                f"repair_attempts={repair_attempts}",
                "Do not write repaired answers to long-term memory until they pass independent verification.",
                {"repair_attempts": repair_attempts},
            ))

        if int(trace.get("low_confidence_count") or 0) >= 2 and any(word in answer_lower for word in self.UNSUPPORTED_CERTAINTY_WORDS):
            findings.append(self._finding(
                "repeated_low_confidence_action",
                "medium",
                0.72,
                self._preview(answer_text),
                "Avoid strong claims after repeated low-confidence decisions.",
                {"low_confidence_count": trace.get("low_confidence_count")},
            ))

        return self._decision(findings, default_action="reduce_trust")

    def check_memory_write_taint(
        self,
        question: str,
        answer: str,
        context_trace: Dict[str, Any] | None = None,
    ) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

        trace = dict(context_trace or {})
        findings: List[SecurityFinding] = []
        upstream_risks = self._upstream_risks(trace)
        answer_text = str(answer or "")
        question_text = str(question or "")

        if upstream_risks and (trace.get("write_memory") is not False):
            findings.append(self._finding(
                "tainted_memory_write",
                "critical",
                0.92,
                self._preview(answer_text or question_text),
                "Skip semantic/graph memory writes when the answer depends on poisoned or unverified upstream context.",
                {"propagation_path": self.build_propagation_path(trace, sink="memory_write")},
            ))

        if bool(trace.get("poisoned_context") or trace.get("answer_used_risky_context")):
            findings.append(self._finding(
                "poisoned_rag_to_answer_to_memory",
                "critical",
                0.94,
                self._preview(answer_text or question_text),
                "Poisoned RAG/tool/memory context must not be promoted into long-term semantic memory.",
                {"source": trace.get("risk_source") or "unknown"},
            ))

        if bool(trace.get("cross_session_source")) and trace.get("source_session_id") != trace.get("target_session_id"):
            findings.append(self._finding(
                "unsafe_memory_reuse_cross_session",
                "high",
                0.86,
                str(trace.get("source_session_id") or ""),
                "Do not write or reuse contaminated memory across sessions without user ownership checks.",
                {"source_session_id": trace.get("source_session_id"), "target_session_id": trace.get("target_session_id")},
            ))

        decision = self._decision(findings, default_action="skip_memory_write")
        if findings:
            decision.allowed = False
            decision.action = "skip_memory_write"
            decision.metadata["memory_write_blocked_reason"] = "upstream_taint"
        return decision

    def check_tool_decision_taint(
        self,
        tool_call: Dict[str, Any] | str,
        upstream_risks: Iterable[Dict[str, Any]] | Dict[str, Any] | None = None,
    ) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

        tool = tool_call if isinstance(tool_call, dict) else {"tool_name": str(tool_call)}
        risks = self._normalize_risks(upstream_risks)
        text = f"{tool.get('tool_name', '')} {tool.get('arguments', '')} {tool.get('content', '')} {tool.get('reason', '')}"
        findings: List[SecurityFinding] = []
        if risks and self._contains_high_risk_action(text):
            findings.append(self._finding(
                "bad_tool_output_to_decision",
                "high",
                0.84,
                self._preview(text),
                "High-impact tool decisions based on risky upstream signals require confirmation or revalidation.",
                {"upstream_risks": risks},
            ))
        return self._decision(findings, default_action="require_confirmation")

    def check_tool_output_cascade(self, text: str, context_trace: Dict[str, Any] | None = None) -> SecurityDecision:
        trace = dict(context_trace or {})
        findings: List[SecurityFinding] = []
        value = str(text or "")
        lower = value.lower()
        if any(marker in lower for marker in ("incorrectly says", "false positive", "failed but", "deployment succeeded")) and self._contains_high_risk_action(value):
            findings.append(self._finding(
                "bad_tool_output_to_decision",
                "high",
                0.8,
                self._preview(value),
                "Treat inconsistent tool outputs as warnings and avoid follow-up destructive actions.",
                {"propagation_path": self.build_propagation_path(trace, sink="tool_decision")},
            ))
        if "search failed" in lower and any(marker in lower for marker in ("therefore cite", "[d1]", "[m1]", "local evidence")):
            findings.append(self._finding(
                "failed_online_search_to_general_knowledge_overtrust",
                "medium",
                0.7,
                self._preview(value),
                "A failed search must not be converted into fabricated local evidence.",
                {},
            ))
        return self._decision(findings, default_action="warn")

    def check_trace(self, trace: Dict[str, Any] | None = None) -> SecurityDecision:
        trace = dict(trace or {})
        findings: List[SecurityFinding] = []
        if trace.get("answer_used_risky_context") and trace.get("memory_write_requested"):
            findings.append(self._finding(
                "cascading_failure",
                "critical",
                0.9,
                "answer_used_risky_context -> memory_write_requested",
                "Block the memory write or quarantine it as low-trust trace data.",
                {"propagation_path": self.build_propagation_path(trace, sink="memory_write")},
            ))
        if int(trace.get("repair_attempts") or 0) > 1 and trace.get("verification_passed") is False:
            findings.append(self._finding(
                "repair_loop_amplification",
                "high",
                0.82,
                "repair loop did not verify",
                "Stop repair loops that fail verification and avoid persisting their output.",
                {"repair_attempts": trace.get("repair_attempts")},
            ))
        return self._decision(findings, default_action="warn")

    def build_propagation_path(self, trace: Dict[str, Any] | None = None, sink: str = "") -> List[str]:
        trace = dict(trace or {})
        path: List[str] = []
        for key, label in (
            ("risky_input", "input"),
            ("risky_rag_chunks", "rag"),
            ("risky_tool_outputs", "tool_output"),
            ("risky_memory_contexts", "memory_context"),
            ("answer_used_risky_context", "final_answer"),
            ("tool_call_requested", "tool_call"),
            ("memory_write_requested", "memory_write"),
        ):
            value = trace.get(key)
            if value:
                path.append(label)
        if sink and sink not in path:
            path.append(sink)
        return path

    def _upstream_risks(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        risks: List[Dict[str, Any]] = []
        for key in ("upstream_risks", "taint_trace", "risk_events"):
            risks.extend(self._normalize_risks(trace.get(key)))
        for key, source in (
            ("poisoned_context", "rag"),
            ("risky_input", "input"),
            ("risky_rag_chunks", "rag"),
            ("risky_tool_outputs", "tool_output"),
            ("risky_memory_contexts", "memory_context"),
        ):
            if trace.get(key):
                risks.append(self.mark_taint(source, str(key), str(trace.get("object_id") or "")))
        return risks

    def _normalize_risks(self, risks: Iterable[Dict[str, Any]] | Dict[str, Any] | Any) -> List[Dict[str, Any]]:
        if risks is None:
            return []
        if isinstance(risks, dict):
            if risks.get("risk_type") or risks.get("source") or risks.get("severity"):
                return [dict(risks)]
            values = []
            for key, value in risks.items():
                if value:
                    values.append(self.mark_taint(str(key), str(key), severity="high"))
            return values
        if isinstance(risks, list):
            return [dict(item) if isinstance(item, dict) else self.mark_taint("unknown", str(item)) for item in risks if item]
        return [self.mark_taint("unknown", str(risks))]

    def _uses_risky_context(self, answer: str, trace: Dict[str, Any]) -> bool:
        if trace.get("answer_used_risky_context"):
            return True
        markers = [str(item).lower() for item in trace.get("risky_terms", []) if item]
        answer_lower = answer.lower()
        return any(marker and marker in answer_lower for marker in markers)

    def _contains_high_risk_action(self, text: str) -> bool:
        lower = str(text or "").lower()
        return any(word.lower() in lower for word in self.HIGH_RISK_ACTION_WORDS)

    def _decision(self, findings: List[SecurityFinding], default_action: str) -> SecurityDecision:
        risk_score = max((finding.confidence for finding in findings), default=0.0)
        action = "allow"
        allowed = True
        if findings:
            action = default_action
            allowed = default_action not in {"block", "skip_memory_write", "require_confirmation"}
        return SecurityDecision(
            allowed=allowed,
            action=action,
            risk_score=risk_score,
            findings=findings,
            metadata={"propagation_risk_count": len(findings)},
        )

    def _finding(
        self,
        risk_type: str,
        severity: str,
        confidence: float,
        evidence: str,
        recommendation: str,
        metadata: Dict[str, Any] | None = None,
    ) -> SecurityFinding:
        return SecurityFinding(
            risk_type=risk_type,
            severity=severity,
            confidence=confidence,
            evidence=str(evidence)[: self.config.max_evidence_chars],
            recommendation=recommendation,
            metadata=metadata or {},
        )

    def _preview(self, text: str, max_chars: int = 160) -> str:
        value = str(text or "").replace("\n", " ").strip()
        return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
