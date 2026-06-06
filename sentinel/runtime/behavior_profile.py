from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class BehaviorProfile:
    session_id: str
    user_goal: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    blocked_attempts: int = 0
    confirmations: int = 0
    memory_writes: int = 0
    external_actions: int = 0
    risk_score_history: List[float] = field(default_factory=list)

    def add_trace(self, trace: Dict[str, Any]) -> None:
        if trace.get("user_goal") and not self.user_goal:
            self.user_goal = str(trace.get("user_goal"))
        for action in trace.get("actions", []) or trace.get("agent_actions", []) or []:
            if isinstance(action, dict):
                self.actions.append(dict(action))
        for call in trace.get("tool_calls", []) or []:
            if isinstance(call, dict):
                self.tool_calls.append(dict(call))
        self.blocked_attempts += int(trace.get("blocked_attempts") or 0)
        self.confirmations += int(trace.get("confirmations") or 0)
        self.memory_writes += int(trace.get("memory_writes") or 0)
        self.external_actions += int(trace.get("external_actions") or 0)

    def record_risk(self, risk_score: float) -> None:
        self.risk_score_history.append(max(0.0, min(1.0, float(risk_score or 0.0))))

    def score(self) -> float:
        score = 0.0
        score += min(0.35, 0.06 * len(self.tool_calls))
        score += min(0.25, 0.08 * self.external_actions)
        score += min(0.25, 0.08 * self.blocked_attempts)
        score += min(0.15, 0.04 * self.memory_writes)
        if self.risk_score_history:
            score += max(self.risk_score_history) * 0.25
        return round(min(1.0, score), 4)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["session_risk_score"] = self.score()
        return data


class BehaviorProfileStore:
    def __init__(self):
        self._profiles: Dict[str, BehaviorProfile] = {}

    def get(self, session_id: str) -> BehaviorProfile:
        key = session_id or "default"
        if key not in self._profiles:
            self._profiles[key] = BehaviorProfile(session_id=key)
        return self._profiles[key]

    def update(self, session_id: str, trace: Dict[str, Any], risk_score: float = 0.0) -> BehaviorProfile:
        profile = self.get(session_id)
        profile.add_trace(trace)
        profile.record_risk(risk_score)
        return profile
