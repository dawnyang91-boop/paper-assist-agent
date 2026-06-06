from __future__ import annotations

from enum import StrEnum
from typing import Iterable


class ShareLevel(StrEnum):
    NO_SHARE = "no_share"
    SANITIZED_FACTS = "sanitized_facts"
    CITED_EVIDENCE = "cited_evidence"
    PRIVILEGED = "privileged"
    NEVER_SHARE = "never_share"


SHARE_RANK = {
    ShareLevel.NO_SHARE: 0,
    ShareLevel.SANITIZED_FACTS: 1,
    ShareLevel.CITED_EVIDENCE: 2,
    ShareLevel.PRIVILEGED: 3,
    ShareLevel.NEVER_SHARE: 99,
}


def normalize_share_level(value: str | ShareLevel | None) -> ShareLevel:
    if isinstance(value, ShareLevel):
        return value
    text = str(value or "").strip().lower()
    for level in ShareLevel:
        if text == level.value:
            return level
    return ShareLevel.NO_SHARE


def can_share_with(level: str | ShareLevel, receiver: str, allowed_receivers: Iterable[str] | None = None) -> bool:
    normalized = normalize_share_level(level)
    if normalized in {ShareLevel.NO_SHARE, ShareLevel.NEVER_SHARE}:
        return False
    allowed = {str(item) for item in (allowed_receivers or [])}
    return not allowed or receiver in allowed or "*" in allowed


def is_high_risk_taint(labels: Iterable[str] | None = None) -> bool:
    taints = {str(item).lower() for item in (labels or [])}
    return bool(taints.intersection({
        "high_risk",
        "prompt_injection_detected",
        "private_data",
        "permission_sensitive",
        "tool_output_untrusted",
        "rag_untrusted",
    }))
