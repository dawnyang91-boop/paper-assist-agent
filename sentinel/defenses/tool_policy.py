from __future__ import annotations

import re
from typing import Any, Dict

from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding


TOOL_LEVELS = {
    "fetch": "safe_read",
    "brave": "safe_read",
    "brave_search": "safe_read",
    "search": "safe_read",
    "sqlite": "sensitive_read",
    "filesystem": "sensitive_read",
    "file_read": "sensitive_read",
    "write": "write",
    "update": "write",
    "delete": "external_action",
    "email": "external_action",
    "payment": "external_action",
}


class ToolPolicyChecker:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def check(self, tool_name: str, args: Any = None, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        if not self.config.enabled or not self.config.tool_guard_enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, metadata={"tool_name": tool_name})

        level = self._infer_level(tool_name)
        if level == "safe_read":
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, metadata={"tool_name": tool_name, "level": level})
        if level == "sensitive_read":
            finding = SecurityFinding(
                risk_type="tool_misuse",
                severity="medium",
                confidence=0.65,
                evidence=tool_name,
                recommendation="Check user permission and scope before reading private resources.",
            )
            return SecurityDecision(
                allowed=True,
                action="warn",
                risk_score=0.65,
                findings=[finding],
                metadata={"tool_name": tool_name, "level": level},
            )
        if level == "write":
            finding = SecurityFinding(
                risk_type="tool_misuse",
                severity="high",
                confidence=0.86,
                evidence=tool_name,
                recommendation="Require explicit user confirmation before write actions.",
            )
            return SecurityDecision(
                allowed=False,
                action="require_confirmation",
                risk_score=0.86,
                findings=[finding],
                metadata={"tool_name": tool_name, "level": level},
            )

        finding = SecurityFinding(
            risk_type="tool_misuse",
            severity="critical",
            confidence=0.95,
            evidence=tool_name,
            recommendation="Block external actions unless an explicit approval workflow exists.",
        )
        return SecurityDecision(
            allowed=False,
            action="block",
            risk_score=0.95,
            findings=[finding],
            metadata={"tool_name": tool_name, "level": level},
        )

    def _infer_level(self, tool_name: str) -> str:
        name = (tool_name or "").lower()
        for keyword, level in TOOL_LEVELS.items():
            if re.search(re.escape(keyword), name):
                return level
        return "safe_read"
