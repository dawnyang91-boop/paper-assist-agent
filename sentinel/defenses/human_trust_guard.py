from __future__ import annotations

import re
from typing import Any, Dict, List

from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding, decision_from_findings


TRUST_EXPLOITATION_PATTERNS = (
    re.compile(r"\b(completely safe|no risk|harmless cleanup|just click confirm|no need to review)\b", re.I),
    re.compile(r"\b(official policy says|admin approved|owner approved)\b", re.I),
    re.compile(r"(完全安全|没有风险|只是清理|无需审查|官方政策|管理员已批准|紧急|必须立刻)"),
)

OVERCONFIDENCE_PATTERNS = (
    re.compile(r"\b(100%\s*safe|definitely correct|certainly safe|guaranteed)\b", re.I),
    re.compile(r"(一定|必然|完全正确|百分之百安全|绝对安全)"),
)

DESTRUCTIVE_WORDS = ("delete", "remove", "drop", "rm -rf", "删除", "清空", "移除", "销毁")
EXTERNAL_WORDS = ("send", "email", "external", "deploy", "publish", "发送", "外发", "部署", "发布")


class HumanTrustGuard:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def check_risk_downplaying(self, answer: str, proposed_action: Dict[str, Any] | None = None) -> SecurityDecision:
        findings = []
        for pattern in TRUST_EXPLOITATION_PATTERNS:
            match = pattern.search(answer or "")
            if match:
                findings.append(SecurityFinding(
                    risk_type="human_trust_exploitation",
                    severity="medium",
                    confidence=0.72,
                    evidence=match.group(0),
                    recommendation="Rewrite risk communication to clearly state uncertainty, affected resources, and consequences.",
                    metadata={"owasp_id": "ASI09", "detector": "TrustExploitationDetector"},
                ))
        findings.extend(self.check_overconfident_claims(answer, {}).findings)
        return decision_from_findings(findings, warn_threshold=self.config.warn_threshold, block_threshold=self.config.block_threshold)

    def check_confirmation_text(self, tool_call: Dict[str, Any], confirmation_text: str) -> SecurityDecision:
        tool_name = str(tool_call.get("tool_name") or tool_call.get("name") or "")
        args = tool_call.get("tool_args") or tool_call.get("args") or {}
        actual = f"{tool_name} {args}".lower()
        visible = (confirmation_text or "").lower()
        findings: List[SecurityFinding] = []
        destructive = any(word in actual for word in DESTRUCTIVE_WORDS)
        external = any(word in actual for word in EXTERNAL_WORDS)
        if destructive and not any(word in visible for word in DESTRUCTIVE_WORDS):
            findings.append(self._mismatch("destructive action hidden", confirmation_text))
        if external and not any(word in visible for word in EXTERNAL_WORDS):
            findings.append(self._mismatch("external action hidden", confirmation_text))
        if findings:
            return SecurityDecision(
                allowed=False,
                action="block",
                risk_score=max(f.confidence for f in findings),
                findings=findings,
            )
        return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

    def build_safe_confirmation(self, tool_call: Dict[str, Any], risk_decision: SecurityDecision) -> Dict[str, Any]:
        args = tool_call.get("tool_args") or tool_call.get("args") or {}
        return {
            "tool_name": tool_call.get("tool_name") or tool_call.get("name"),
            "tool_level": risk_decision.metadata.get("level"),
            "exact_target": args.get("path") or args.get("target") or args.get("to") or args.get("repo"),
            "arguments_summary": {key: self._shorten(value) for key, value in args.items()},
            "destructive_effect": any(word in str(args).lower() for word in DESTRUCTIVE_WORDS),
            "external_recipient": args.get("to") or args.get("url"),
            "risk_explanation": [finding.recommendation for finding in risk_decision.findings],
        }

    def check_overconfident_claims(self, answer: str, evidence_status: Dict[str, Any] | None = None) -> SecurityDecision:
        findings = []
        for pattern in OVERCONFIDENCE_PATTERNS:
            match = pattern.search(answer or "")
            if match:
                findings.append(SecurityFinding(
                    risk_type="human_trust_exploitation",
                    severity="medium",
                    confidence=0.62,
                    evidence=match.group(0),
                    recommendation="Avoid absolute certainty when local evidence is weak or the action is high risk.",
                    metadata={"owasp_id": "ASI09", "detector": "OverconfidenceDetector"},
                ))
        return decision_from_findings(findings, warn_threshold=self.config.warn_threshold, block_threshold=self.config.block_threshold)

    def _mismatch(self, evidence: str, confirmation_text: str) -> SecurityFinding:
        return SecurityFinding(
            risk_type="misleading_confirmation",
            severity="high",
            confidence=0.9,
            evidence=f"{evidence}: {confirmation_text}",
            recommendation="Confirmation text must explicitly name the real tool action and affected resources.",
            metadata={"owasp_id": "ASI09", "detector": "ConfirmationMismatchDetector"},
        )

    def _shorten(self, value: Any) -> str:
        text = str(value)
        return text if len(text) <= 180 else text[:177] + "..."
