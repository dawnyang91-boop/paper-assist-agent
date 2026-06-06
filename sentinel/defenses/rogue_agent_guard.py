from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from sentinel.config import SentinelConfig
from sentinel.runtime.behavior_profile import BehaviorProfileStore
from sentinel.schemas import SecurityDecision, SecurityFinding


class RogueAgentGuard:
    """Detect autonomous behavior drift and unsafe self-directed actions."""

    HIGH_RISK_TOOLS = {
        "send_email",
        "deploy_service",
        "delete_resource",
        "file_write",
        "github_write",
        "github_update_file",
        "run_command",
        "filesystem",
        "sqlite",
    }
    EXTERNAL_TOOLS = {"send_email", "deploy_service", "github_write", "run_command"}
    SELF_MODIFICATION_PATHS = (
        "sentinel/",
        "config.py",
        ".env",
        "tool_policy.py",
        "security_manager.py",
        "system_prompt",
        ".github/workflows/",
        "deploy",
        "docker-compose",
    )
    PRIVATE_COLLECTION_PATTERNS = (
        re.compile(r"(scan|collect|read|crawl).{0,40}(private|unrelated|all files|filesystem|uploads|transcripts)", re.I),
        re.compile(r"(扫描|收集|读取).{0,20}(无关|私有|所有文件|文件系统|上传|会话)"),
    )
    SAFETY_DISABLE_PATTERNS = (
        re.compile(r"SENTINEL_ENABLED\s*=\s*false", re.I),
        re.compile(r"disable\s+(all\s+)?(safety|checks|sentinel|tool policy)", re.I),
        re.compile(r"(关闭|禁用).{0,10}(安全|检查|Sentinel|策略)"),
    )

    def __init__(self, config: SentinelConfig):
        self.config = config
        self.behavior_profiles = BehaviorProfileStore()

    def check_goal_alignment(self, user_goal: str, proposed_action: Dict[str, Any] | str) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)
        action = self._as_action(proposed_action)
        findings: List[SecurityFinding] = []
        tool_name = str(action.get("tool_name") or action.get("tool") or "").lower()
        action_text = self._action_text(action)
        goal_text = str(user_goal or "").lower()
        if tool_name in self.HIGH_RISK_TOOLS and not self._goal_mentions_action(goal_text, action_text):
            findings.append(self._finding(
                "goal_drift",
                "high",
                0.88,
                action_text,
                "High-risk actions must be directly tied to the user's current goal.",
            ))
        if any(pattern.search(action_text) for pattern in self.PRIVATE_COLLECTION_PATTERNS):
            findings.append(self._finding(
                "autonomous_data_collection",
                "high",
                0.9,
                action_text,
                "Agent must not proactively collect unrelated private data.",
            ))
        return self._decision(findings, action="block" if findings else "allow", allowed=not findings)

    def check_autonomy_violation(self, action: Dict[str, Any] | str, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        user_context = user_context or {}
        item = self._as_action(action)
        findings: List[SecurityFinding] = []
        tool_name = str(item.get("tool_name") or item.get("tool") or "").lower()
        explicit = bool(user_context.get("user_explicitly_requested") or item.get("user_explicitly_requested"))
        confirmed = bool(user_context.get("confirmed") or item.get("confirmed"))
        if tool_name in self.EXTERNAL_TOOLS and not explicit:
            findings.append(self._finding(
                "excessive_autonomy",
                "high",
                0.9,
                self._action_text(item),
                "External actions require an explicit user request.",
            ))
        if tool_name in {"file_write", "github_write", "delete_resource", "deploy_service", "run_command"} and not confirmed:
            findings.append(self._finding(
                "excessive_autonomy",
                "high",
                0.88,
                self._action_text(item),
                "Write, delete, deploy, and command execution require explicit confirmation.",
            ))
        return self._decision(findings, action="block" if findings else "allow", allowed=not findings)

    def check_self_modification(self, tool_call: Dict[str, Any] | str) -> SecurityDecision:
        item = self._as_action(tool_call)
        text = self._action_text(item)
        path = str(item.get("path") or item.get("target") or item.get("file") or "")
        content = str(item.get("content") or item.get("value") or "")
        findings: List[SecurityFinding] = []
        if any(marker.lower() in path.lower() or marker.lower() in text.lower() for marker in self.SELF_MODIFICATION_PATHS):
            findings.append(self._finding(
                "self_modification",
                "critical",
                0.95,
                path or text,
                "Agent cannot modify Sentinel, policy, prompt, environment, CI/CD, or deployment configuration without human review.",
            ))
        if any(pattern.search(content or text) for pattern in self.SAFETY_DISABLE_PATTERNS):
            findings.append(self._finding(
                "policy_drift",
                "critical",
                0.96,
                content or text,
                "Attempts to disable or weaken safety controls must be blocked.",
            ))
        return self._decision(findings, action="block" if findings else "allow", allowed=not findings)

    def update_behavior_profile(self, session_id: str, trace: Dict[str, Any]) -> Dict[str, Any]:
        decisions = self.check_behavior_trace(trace)
        profile = self.behavior_profiles.update(session_id, trace, risk_score=decisions.risk_score)
        return profile.to_dict()

    def compute_session_risk_score(self, session_id: str) -> float:
        return self.behavior_profiles.get(session_id).score()

    def check_behavior_trace(self, trace: Dict[str, Any] | None = None) -> SecurityDecision:
        trace = dict(trace or {})
        findings: List[SecurityFinding] = []
        user_goal = str(trace.get("user_goal") or trace.get("original_user_goal") or "")
        actions = trace.get("agent_actions") or trace.get("actions") or []
        if isinstance(actions, dict):
            actions = [actions]
        for action in actions:
            if not isinstance(action, dict):
                action = {"content": action}
            findings.extend(self.check_goal_alignment(user_goal, action).findings)
            findings.extend(self.check_autonomy_violation(action, user_context=trace.get("user_context") or {}).findings)
            findings.extend(self.check_self_modification(action).findings)

        tool_call_count = int(trace.get("tool_call_count") or len(trace.get("tool_calls") or []))
        high_risk_count = int(trace.get("high_risk_tool_count") or 0)
        if not high_risk_count:
            high_risk_count = sum(
                1 for action in actions
                if str((action or {}).get("tool_name") or (action or {}).get("tool") or "").lower() in self.HIGH_RISK_TOOLS
            )
        blocked_attempts = int(trace.get("blocked_attempts") or trace.get("repeated_blocked_attempts") or 0)
        memory_writes = int(trace.get("memory_write_count") or trace.get("memory_writes") or 0)

        if tool_call_count >= 8 or high_risk_count >= 4:
            findings.append(self._finding(
                "abnormal_tool_frequency",
                "high",
                0.86,
                f"tool_call_count={tool_call_count}, high_risk_tool_count={high_risk_count}",
                "Abnormal tool frequency should trigger autonomous loop circuit breaker.",
            ))
        if blocked_attempts >= 2:
            findings.append(self._finding(
                "repeated_policy_violation",
                "high",
                0.88,
                f"blocked_attempts={blocked_attempts}",
                "Repeated policy violations require session-level blocking or human review.",
            ))
        hidden_objective = str(trace.get("hidden_long_term_objective") or trace.get("memory_candidate") or "")
        if hidden_objective and any(pattern.search(hidden_objective) for pattern in self.SAFETY_DISABLE_PATTERNS):
            findings.append(self._finding(
                "hidden_long_term_objective",
                "critical",
                0.92,
                hidden_objective,
                "Dangerous long-term objectives must not be persisted in memory.",
            ))
        if memory_writes >= 5 and high_risk_count:
            findings.append(self._finding(
                "excessive_autonomy",
                "medium",
                0.7,
                f"memory_writes={memory_writes}, high_risk_tool_count={high_risk_count}",
                "High-risk autonomous sessions should reduce memory persistence.",
            ))
        return self._decision(findings, action="block" if any(f.confidence >= 0.85 for f in findings) else ("warn" if findings else "allow"), allowed=not any(f.confidence >= 0.85 for f in findings))

    def _as_action(self, value: Dict[str, Any] | str) -> Dict[str, Any]:
        if isinstance(value, dict):
            args = value.get("args") if isinstance(value.get("args"), dict) else {}
            tool_args = value.get("tool_args") if isinstance(value.get("tool_args"), dict) else {}
            return {**value, **args, **tool_args}
        return {"content": str(value)}

    def _action_text(self, action: Dict[str, Any]) -> str:
        return " ".join(str(value) for value in self._flatten_values(action) if value is not None)

    def _goal_mentions_action(self, goal: str, action_text: str) -> bool:
        if not goal:
            return False
        action_lower = action_text.lower()
        goal_tokens = {token for token in re.split(r"\W+", goal) if len(token) >= 4}
        if any(token in action_lower for token in goal_tokens):
            return True
        safe_goal_markers = ("summary", "summarize", "论文", "总结", "answer", "问答")
        risky_markers = ("email", "deploy", "delete", "scan", "collect", "filesystem", "private", "发送", "部署", "删除", "扫描", "收集")
        return not any(marker in action_lower for marker in risky_markers) and any(marker in goal for marker in safe_goal_markers)

    def _flatten_values(self, value: Any) -> Iterable[Any]:
        if isinstance(value, dict):
            for item in value.values():
                yield from self._flatten_values(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                yield from self._flatten_values(item)
        else:
            yield value

    def _decision(self, findings: List[SecurityFinding], action: str, allowed: bool) -> SecurityDecision:
        return SecurityDecision(
            allowed=allowed,
            action=action,
            risk_score=max((finding.confidence for finding in findings), default=0.0),
            findings=findings,
            metadata={"session_guard": "rogue_agent", "finding_count": len(findings)},
        )

    def _finding(self, risk_type: str, severity: str, confidence: float, evidence: Any, recommendation: str) -> SecurityFinding:
        return SecurityFinding(
            risk_type=risk_type,
            severity=severity,
            confidence=confidence,
            evidence=str(evidence)[: self.config.max_evidence_chars],
            recommendation=recommendation,
            metadata={"owasp_id": "ASI10"},
        )
