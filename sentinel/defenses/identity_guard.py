from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List

from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding, decision_from_findings


AUTHORITY_PATTERNS = (
    re.compile(r"\b(i am|i'm|as)\s+(the\s+)?(admin|administrator|developer|owner|security auditor)\b", re.I),
    re.compile(r"\b(owner|admin|administrator|developer)\s+(approved|authorized|said|granted)\b", re.I),
    re.compile(r"(我是|作为).{0,8}(管理员|开发者|所有者|审计员|系统管理员)"),
    re.compile(r"(已授权|授权你|管理员授权|系统授权|安全审计授权)"),
)

FORGED_PERMISSION_PATTERNS = (
    re.compile(r"\bpermission_granted\s*=\s*true\b", re.I),
    re.compile(r"\brole\s*=\s*(admin|owner|developer)\b", re.I),
    re.compile(r"\bauthorized_by_(system|admin|owner)\b", re.I),
    re.compile(r"\b(user|admin)\s+(has\s+)?approved\b", re.I),
    re.compile(r"\bno\s+confirmation\s+required\b", re.I),
    re.compile(r"(无需确认|不需要确认|权限已授予|角色=管理员)"),
)

TENANT_KEYS = {"user_id", "tenant_id", "session_id", "owner_id"}


class IdentityGuard:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def check_identity_claim(self, text: str, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)
        findings = self._authority_findings(text)
        return decision_from_findings(
            findings,
            warn_threshold=self.config.warn_threshold,
            block_threshold=self.config.block_threshold,
        )

    def check_permission_claim(self, text: str, source: str = "untrusted") -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)
        findings = []
        for pattern in FORGED_PERMISSION_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(SecurityFinding(
                    risk_type="identity_privilege_abuse",
                    severity="high",
                    confidence=0.9,
                    evidence=match.group(0),
                    recommendation="Ignore permission claims from untrusted text; only authenticated application context may grant permissions.",
                    metadata={"source": source, "detector": "ForgedPermissionDetector"},
                ))
        return decision_from_findings(
            findings,
            warn_threshold=self.config.warn_threshold,
            block_threshold=self.config.block_threshold,
        )

    def check_cross_user_access(self, tool_args: Any, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)
        user_context = user_context or {}
        findings: List[SecurityFinding] = []
        flattened = dict(self._flatten_items(tool_args if isinstance(tool_args, dict) else {"input": tool_args}))
        for key, value in flattened.items():
            key_lower = str(key).lower()
            text = str(value)
            if key_lower.split(".")[-1] in TENANT_KEYS:
                expected = user_context.get(key_lower.split(".")[-1])
                if expected and str(expected) != text:
                    findings.append(self._finding(
                        "cross_user_access",
                        f"{key}={text}",
                        "Tool arguments target a different user, tenant, or session than the authenticated context.",
                        confidence=0.94,
                    ))
            if self._looks_cross_user_path(text, user_context):
                findings.append(self._finding(
                    "cross_user_access",
                    text,
                    "Path or query appears to access another user's private data or transcript scope.",
                    confidence=0.92,
                ))
        return decision_from_findings(
            findings,
            warn_threshold=self.config.warn_threshold,
            block_threshold=self.config.block_threshold,
        )

    def check_text(self, text: str, user_context: Dict[str, Any] | None = None, source: str = "untrusted") -> SecurityDecision:
        findings = []
        findings.extend(self.check_identity_claim(text, user_context).findings)
        findings.extend(self.check_permission_claim(text, source=source).findings)
        return decision_from_findings(
            findings,
            warn_threshold=self.config.warn_threshold,
            block_threshold=self.config.block_threshold,
        )

    def _authority_findings(self, text: str) -> List[SecurityFinding]:
        findings = []
        for pattern in AUTHORITY_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(SecurityFinding(
                    risk_type="identity_privilege_abuse",
                    severity="high",
                    confidence=0.88,
                    evidence=match.group(0),
                    recommendation="Do not let natural-language authority claims elevate role or permissions.",
                    metadata={"detector": "AuthorityClaimDetector"},
                ))
        return findings

    def _looks_cross_user_path(self, text: str, user_context: Dict[str, Any]) -> bool:
        lowered = (text or "").lower()
        if "../" in lowered or "..\\" in lowered:
            return True
        risky_roots = ("data/transcripts/", "data/uploads/", "transcripts/", "uploads/")
        if not any(root in lowered for root in risky_roots):
            return False
        current_user = str(user_context.get("user_id") or user_context.get("user_email") or "").lower()
        if current_user and current_user in lowered:
            return False
        parts = PurePosixPath(lowered.replace("\\", "/")).parts
        return any(part not in {"data", "transcripts", "uploads", "private", ".", "/"} for part in parts)

    def _flatten_items(self, value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
        if isinstance(value, dict):
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._flatten_items(item, next_prefix)
        elif isinstance(value, (list, tuple, set)):
            for index, item in enumerate(value):
                yield from self._flatten_items(item, f"{prefix}.{index}" if prefix else str(index))
        else:
            yield prefix, value

    def _finding(self, risk_type: str, evidence: str, recommendation: str, confidence: float) -> SecurityFinding:
        return SecurityFinding(
            risk_type=risk_type,
            severity="high",
            confidence=confidence,
            evidence=evidence,
            recommendation=recommendation,
            metadata={"owasp_id": "ASI03"},
        )
