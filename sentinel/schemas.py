from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from sentinel.owasp_agentic_top10 import normalize_owasp_ids


@dataclass
class SecurityFinding:
    risk_type: str
    severity: str
    confidence: float
    evidence: str
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityDecision:
    allowed: bool
    action: str
    risk_score: float
    findings: List[SecurityFinding] = field(default_factory=list)
    sanitized_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["findings"] = [finding.to_dict() for finding in self.findings]
        return data


@dataclass
class EvalCase:
    case_id: str
    category: str
    attack_type: str
    input: str
    expected_behavior: str
    owasp_id: List[str] = field(default_factory=list)
    runtime_stage: str = "user_input"
    expected_action: str = ""
    target_mode: str = "chat"
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalCase":
        category = str(data.get("category", ""))
        expected_action = str(data.get("expected_action", ""))
        expected_behavior = str(data.get("expected_behavior", "")) or expected_action
        return cls(
            case_id=str(data.get("case_id", "")),
            category=category,
            attack_type=str(data.get("attack_type", "")),
            input=str(data.get("input", "")),
            expected_behavior=expected_behavior,
            owasp_id=normalize_owasp_ids(data.get("owasp_id"), category),
            runtime_stage=str(data.get("runtime_stage", "user_input")),
            expected_action=expected_action,
            target_mode=str(data.get("target_mode", "chat")),
            labels=[str(item) for item in data.get("labels", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    score: float
    target_answer: str
    findings: List[SecurityFinding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["findings"] = [finding.to_dict() for finding in self.findings]
        return data


def decision_from_findings(
    findings: List[SecurityFinding],
    sanitized_text: Optional[str] = None,
    warn_threshold: float = 0.55,
    block_threshold: float = 0.85,
) -> SecurityDecision:
    risk_score = max((finding.confidence for finding in findings), default=0.0)
    if risk_score >= block_threshold:
        action = "block"
        allowed = False
    elif findings and sanitized_text is not None and sanitized_text != "":
        action = "sanitize"
        allowed = True
    elif risk_score >= warn_threshold:
        action = "warn"
        allowed = True
    else:
        action = "allow"
        allowed = True
    return SecurityDecision(
        allowed=allowed,
        action=action,
        risk_score=risk_score,
        findings=findings,
        sanitized_text=sanitized_text,
    )
