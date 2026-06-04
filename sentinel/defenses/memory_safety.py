from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from sentinel.config import SentinelConfig
from sentinel.detectors.rule_detector import MEMORY_SAFETY_RULES, detect_rules
from sentinel.schemas import SecurityDecision, decision_from_findings


class MemorySafetyChecker:
    """Reject memories that try to become safety policy or tool permissions."""

    def __init__(self, config: SentinelConfig):
        self.config = config

    def check_text(self, text: str) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, sanitized_text=text)

        findings = detect_rules(text, MEMORY_SAFETY_RULES, max_evidence_chars=self.config.max_evidence_chars)
        decision = decision_from_findings(
            findings,
            sanitized_text=text,
            warn_threshold=self.config.warn_threshold,
            block_threshold=0.85,
        )
        if findings:
            decision.allowed = False
            decision.action = "block"
        decision.metadata["memory_context_untrusted"] = True
        return decision

    def filter_contexts(self, memories: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        for index, memory in enumerate(memories):
            item = dict(memory)
            content = str(item.get("content") or item.get("page_content") or "")
            decision = self.check_text(content)
            if decision.findings:
                decisions.append({"index": index, "content_preview": self._preview(content), **decision.to_dict()})
            if not decision.allowed:
                continue
            metadata = dict(item.get("metadata") or {})
            metadata["untrusted_memory_context"] = True
            item["metadata"] = metadata
            kept.append(item)
        return kept, decisions

    def should_write_interaction(self, question: str, answer: str) -> Tuple[bool, List[Dict[str, Any]]]:
        decisions: List[Dict[str, Any]] = []
        allowed = True
        for label, text in (("question", question), ("answer", answer)):
            decision = self.check_text(text)
            if decision.findings:
                decisions.append({"field": label, **decision.to_dict()})
            if not decision.allowed:
                allowed = False
        return allowed, decisions

    def strip_unsafe_memory_lines(self, text: str) -> str:
        kept = []
        for line in (text or "").splitlines():
            if detect_rules(line, MEMORY_SAFETY_RULES, max_evidence_chars=self.config.max_evidence_chars):
                continue
            if re.match(r"\s*[-*>#]*\s*(security|policy|system|developer|tool permission)\s*[:：]", line, flags=re.IGNORECASE):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def _preview(self, text: str, max_chars: int = 120) -> str:
        value = (text or "").replace("\n", " ").strip()
        return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
