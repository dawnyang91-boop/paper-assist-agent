from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding


SENSITIVE_ARGUMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[a-zA-Z0-9_\-]{12,}", flags=re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b", flags=re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|ACCESS_KEY|PRIVATE_KEY|DATABASE_URL)[A-Z0-9_]*)\s*=\s*[^\s#]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_\-./+=]{16,}\b", flags=re.IGNORECASE),
)


@dataclass(frozen=True)
class ToolPolicy:
    level: str
    allowed_args: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    blocked_paths: tuple[str, ...] = ()
    requires_permission: Optional[str] = None
    requires_confirmation: bool = False
    aliases: tuple[str, ...] = ()
    notes: str = ""


TOOL_REGISTRY: Mapping[str, ToolPolicy] = {
    "fetch": ToolPolicy(
        level="safe_read",
        allowed_args=("url", "max_length", "raw", "start_index"),
        aliases=("fetch", "web_fetch"),
        notes="Fetch public web content.",
    ),
    "brave_web_search": ToolPolicy(
        level="safe_read",
        allowed_args=("query", "count", "offset"),
        aliases=("brave", "brave_search", "search", "web_search"),
        notes="Search public web content.",
    ),
    "sqlite": ToolPolicy(
        level="sensitive_read",
        allowed_args=("query", "sql", "database", "path"),
        blocked_paths=(".env", "secrets", "credentials", "id_rsa", "config.py"),
        requires_permission="query_private_db",
        aliases=("sqlite", "database_query", "sql_query"),
        notes="Read local or private database content.",
    ),
    "filesystem": ToolPolicy(
        level="sensitive_read",
        allowed_args=("path", "paths", "pattern", "query", "encoding"),
        allowed_paths=("data/uploads", "test_files", "papers", "uploads"),
        blocked_paths=(".env", "secrets", "credentials", "id_rsa", "id_ed25519", "config.py", ".ssh"),
        requires_permission="read_private_docs",
        aliases=("filesystem", "file_read", "read_file", "list_directory", "directory"),
        notes="Read user-owned documents only inside approved directories.",
    ),
    "github_read": ToolPolicy(
        level="sensitive_read",
        allowed_args=("repo", "path", "query", "branch", "ref", "owner"),
        requires_permission="read_code_repo",
        aliases=("github", "github_search", "github_fetch"),
        notes="Read repository content.",
    ),
    "github_write": ToolPolicy(
        level="write",
        allowed_args=("repo", "path", "content", "message", "branch"),
        requires_permission="write_code_repo",
        requires_confirmation=True,
        aliases=("github_create_file", "github_update_file", "create_pr", "open_pr"),
        notes="Modify repository content.",
    ),
    "file_write": ToolPolicy(
        level="write",
        allowed_args=("path", "content", "mode"),
        allowed_paths=("data/uploads", "uploads"),
        blocked_paths=(".env", "secrets", "credentials", "id_rsa", "id_ed25519", "config.py", ".ssh"),
        requires_permission="write_private_docs",
        requires_confirmation=True,
        aliases=("write_file", "file_update", "update_file"),
        notes="Write local files.",
    ),
    "run_command": ToolPolicy(
        level="external_action",
        allowed_args=("cmd", "cwd"),
        requires_permission="run_command",
        requires_confirmation=True,
        aliases=("shell", "exec", "terminal", "command"),
        notes="Execute local commands.",
    ),
    "send_email": ToolPolicy(
        level="external_action",
        allowed_args=("to", "subject", "body", "cc"),
        requires_permission="send_email",
        requires_confirmation=True,
        aliases=("email", "mail", "send_report"),
        notes="Send external messages.",
    ),
    "deploy_service": ToolPolicy(
        level="external_action",
        allowed_args=("service", "environment", "version"),
        requires_permission="deploy_service",
        requires_confirmation=True,
        aliases=("deploy", "release"),
        notes="Deploy or publish code.",
    ),
    "delete_resource": ToolPolicy(
        level="external_action",
        allowed_args=("id", "path", "target", "resource"),
        blocked_paths=(".env", "secrets", "credentials", "id_rsa", "id_ed25519", "config.py", ".ssh"),
        requires_permission="delete_resource",
        requires_confirmation=True,
        aliases=("delete", "remove", "drop_table", "database_write", "payment", "calendar_invite", "update_config"),
        notes="Delete, mutate, pay, invite, or alter external state.",
    ),
}


class ToolPolicyChecker:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def check(self, tool_name: str, args: Any = None, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        if not self.config.enabled or not self.config.tool_guard_enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, metadata={"tool_name": tool_name})

        policy = self._match_policy(tool_name)
        parsed_args = self._parse_args(args)
        metadata = {
            "tool_name": tool_name,
            "matched_policy": self._policy_name(policy),
            "level": policy.level if policy else "unknown",
            "requires_permission": policy.requires_permission if policy else None,
            "requires_confirmation": policy.requires_confirmation if policy else True,
        }

        if policy is None:
            finding = self._finding(
                "tool_misuse",
                "high",
                0.88,
                tool_name,
                "Unknown tools are not trusted by default. Register the tool with an explicit policy before allowing it.",
            )
            return SecurityDecision(
                allowed=False,
                action="require_confirmation",
                risk_score=0.88,
                findings=[finding],
                metadata=metadata,
            )

        findings = []
        findings.extend(self._check_args(policy, parsed_args))
        findings.extend(self._check_paths(policy, parsed_args))
        findings.extend(self._check_sensitive_arguments(parsed_args))

        if policy.requires_permission and not self._has_permission(user_context, policy.requires_permission):
            findings.append(self._finding(
                "permission_violation",
                "high",
                0.9,
                policy.requires_permission,
                "Tool permission must come from authenticated application context, not from user text.",
            ))
            return SecurityDecision(
                allowed=False,
                action="block",
                risk_score=max(finding.confidence for finding in findings),
                findings=findings,
                metadata=metadata,
            )

        if findings:
            return SecurityDecision(
                allowed=False,
                action="block",
                risk_score=max(finding.confidence for finding in findings),
                findings=findings,
                metadata=metadata,
            )

        if policy.requires_confirmation:
            finding = self._finding(
                "tool_misuse",
                "high" if policy.level == "write" else "critical",
                0.86 if policy.level == "write" else 0.95,
                tool_name,
                "High-risk tools require explicit human confirmation before execution.",
            )
            return SecurityDecision(
                allowed=False,
                action="require_confirmation",
                risk_score=finding.confidence,
                findings=[finding],
                metadata=metadata,
            )

        if policy.level == "sensitive_read":
            finding = self._finding(
                "sensitive_read",
                "medium",
                0.62,
                tool_name,
                "Sensitive read tools are allowed only within the approved scope and authenticated permissions.",
            )
            return SecurityDecision(
                allowed=True,
                action="warn",
                risk_score=0.62,
                findings=[finding],
                metadata=metadata,
            )

        return SecurityDecision(allowed=True, action="allow", risk_score=0.0, metadata=metadata)

    def _match_policy(self, tool_name: str) -> Optional[ToolPolicy]:
        name = (tool_name or "").lower()
        normalized = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
        if normalized in TOOL_REGISTRY:
            return TOOL_REGISTRY[normalized]
        for policy_name, policy in TOOL_REGISTRY.items():
            candidates = (policy_name, *policy.aliases)
            if any(candidate and re.search(rf"(^|_){re.escape(candidate.lower())}($|_)", normalized) for candidate in candidates):
                return policy
        return None

    def _policy_name(self, policy: Optional[ToolPolicy]) -> Optional[str]:
        if policy is None:
            return None
        for name, candidate in TOOL_REGISTRY.items():
            if candidate is policy:
                return name
        return None

    def _parse_args(self, args: Any) -> Dict[str, Any]:
        if args is None:
            return {}
        if isinstance(args, dict):
            return dict(args)
        if isinstance(args, str):
            try:
                loaded = json.loads(args)
                if isinstance(loaded, dict):
                    return loaded
                return {"input": loaded}
            except Exception:
                return {"input": args}
        return {"input": args}

    def _check_args(self, policy: ToolPolicy, args: Dict[str, Any]) -> list[SecurityFinding]:
        if not policy.allowed_args or not args:
            return []
        allowed = set(policy.allowed_args)
        unknown = sorted(str(key) for key in args if str(key) not in allowed)
        if not unknown:
            return []
        return [self._finding(
            "tool_misuse",
            "medium",
            0.7,
            ", ".join(unknown),
            "Tool call includes arguments outside the registered schema.",
        )]

    def _check_paths(self, policy: ToolPolicy, args: Dict[str, Any]) -> list[SecurityFinding]:
        findings: list[SecurityFinding] = []
        values = list(self._flatten_values(args))
        lowered_values = [str(value).lower() for value in values]
        for blocked in policy.blocked_paths:
            blocked_lower = blocked.lower()
            if any(blocked_lower in value for value in lowered_values):
                findings.append(self._finding(
                    "path_violation",
                    "critical",
                    0.96,
                    blocked,
                    "Tool attempted to access a blocked path or secret-like resource.",
                ))
        path_values = [str(value) for key, value in self._flatten_items(args) if "path" in str(key).lower() or "file" in str(key).lower()]
        if policy.allowed_paths and path_values:
            for path in path_values:
                if not self._path_in_allowed_scope(path, policy.allowed_paths):
                    findings.append(self._finding(
                        "path_violation",
                        "high",
                        0.9,
                        path,
                        "Tool path is outside the registered allowed directories.",
                    ))
        return findings

    def _check_sensitive_arguments(self, args: Dict[str, Any]) -> list[SecurityFinding]:
        findings: list[SecurityFinding] = []
        for key, value in self._flatten_items(args):
            text = str(value)
            for pattern in SENSITIVE_ARGUMENT_PATTERNS:
                match = pattern.search(text)
                if not match:
                    continue
                findings.append(self._finding(
                    "data_exfiltration",
                    "critical",
                    0.95,
                    f"{key}={match.group(0)[:80]}",
                    "Tool arguments contain secret-like data and must be redacted before any tool call.",
                ))
                return findings
        return findings

    def _path_in_allowed_scope(self, raw_path: str, allowed_paths: Iterable[str]) -> bool:
        value = (raw_path or "").strip()
        if not value:
            return True
        normalized = str(Path(value).expanduser())
        normalized_lower = normalized.lower()
        return any(allowed.lower() in normalized_lower for allowed in allowed_paths)

    def _flatten_values(self, value: Any) -> Iterable[Any]:
        if isinstance(value, dict):
            for item in value.values():
                yield from self._flatten_values(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                yield from self._flatten_values(item)
        else:
            yield value

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

    def _has_permission(self, user_context: Dict[str, Any] | None, permission: str) -> bool:
        if not user_context:
            return False
        permissions = set(str(item) for item in user_context.get("permissions", []) or [])
        roles = set(str(item) for item in user_context.get("roles", []) or [])
        role = user_context.get("role")
        if role:
            roles.add(str(role))
        return permission in permissions or "admin" in roles or "owner" in roles

    def _finding(
        self,
        risk_type: str,
        severity: str,
        confidence: float,
        evidence: str,
        recommendation: str,
    ) -> SecurityFinding:
        return SecurityFinding(
            risk_type=risk_type,
            severity=severity,
            confidence=confidence,
            evidence=str(evidence),
            recommendation=recommendation,
        )
