from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from sentinel.config import SentinelConfig
from sentinel.detectors.rule_detector import TOOL_OUTPUT_RULES, detect_rules
from sentinel.schemas import SecurityDecision, decision_from_findings


class ToolOutputSanitizer:
    """Sanitize MCP/tool observations before they enter the LLM context."""

    def __init__(self, config: SentinelConfig):
        self.config = config

    def sanitize_text(self, text: str) -> SecurityDecision:
        if not self.config.enabled or not self.config.tool_guard_enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, sanitized_text=text)

        findings = detect_rules(text, TOOL_OUTPUT_RULES, max_evidence_chars=self.config.max_evidence_chars)
        sanitized = self._strip_instruction_lines(text) if findings else text
        decision = decision_from_findings(
            findings,
            sanitized_text=sanitized,
            warn_threshold=self.config.warn_threshold,
            block_threshold=1.1,
        )
        decision.allowed = True
        decision.action = "sanitize" if findings and sanitized != text else decision.action
        decision.metadata["removed_chars"] = max(0, len(text or "") - len(sanitized or ""))
        decision.metadata["untrusted_context"] = True
        return decision

    def sanitize_observation(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], SecurityDecision]:
        item = dict(observation)
        content = str(item.get("content") or "")
        decision = self.sanitize_text(content)
        if decision.sanitized_text is not None:
            item["content"] = decision.sanitized_text
        metadata = dict(item.get("metadata") or {})
        metadata["untrusted_tool_output"] = True
        if decision.findings:
            metadata["sentinel_sanitized"] = True
            metadata["sentinel_findings"] = [finding.to_dict() for finding in decision.findings]
        item["metadata"] = metadata
        return item, decision

    def sanitize_observations(self, observations: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sanitized = []
        decisions = []
        for index, observation in enumerate(observations):
            item, decision = self.sanitize_observation(observation)
            sanitized.append(item)
            if decision.findings:
                decisions.append({"index": index, **decision.to_dict()})
        return sanitized, decisions

    def _strip_instruction_lines(self, text: str) -> str:
        kept = []
        for line in (text or "").splitlines():
            findings = detect_rules(line, TOOL_OUTPUT_RULES, max_evidence_chars=self.config.max_evidence_chars)
            if findings:
                continue
            if re.match(r"\s*[-*>#]*\s*(system|developer|assistant|instruction)\s*[:：]", line, flags=re.IGNORECASE):
                continue
            kept.append(line)
        sanitized = "\n".join(kept).strip()
        return sanitized or "[该工具输出包含疑似指令注入内容，已被 Sentinel 净化。]"
