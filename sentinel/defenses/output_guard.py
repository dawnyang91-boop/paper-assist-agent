from __future__ import annotations

from sentinel.config import SentinelConfig
from sentinel.detectors.rule_detector import OUTPUT_RULES, detect_rules
from sentinel.schemas import SecurityDecision, decision_from_findings


class OutputGuard:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def check(self, text: str) -> SecurityDecision:
        if not self.config.enabled or not self.config.output_guard_enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, sanitized_text=text)
        findings = detect_rules(text, OUTPUT_RULES, max_evidence_chars=self.config.max_evidence_chars)
        sanitized = text
        for finding in findings:
            if finding.risk_type == "data_exfiltration":
                sanitized = sanitized.replace(finding.evidence, "[REDACTED]")
        decision = decision_from_findings(
            findings,
            sanitized_text=sanitized,
            warn_threshold=self.config.warn_threshold,
            block_threshold=1.1,
        )
        decision.allowed = True
        if findings and sanitized != text:
            decision.action = "sanitize"
        return decision
