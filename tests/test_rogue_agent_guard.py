from sentinel.config import SentinelConfig
from sentinel.defenses.rogue_agent_guard import RogueAgentGuard
from sentinel.defenses.security_manager import SecurityManager


def test_rogue_agent_guard_blocks_goal_drift_private_collection():
    guard = RogueAgentGuard(SentinelConfig())

    decision = guard.check_behavior_trace({
        "user_goal": "summarize paper",
        "agent_actions": [
            {"tool_name": "filesystem", "args": {"path": "data/transcripts"}, "reason": "collect all private transcripts"}
        ],
        "tool_call_count": 1,
        "high_risk_tool_count": 1,
    })

    assert decision.action == "block"
    assert any(finding.risk_type in {"goal_drift", "autonomous_data_collection"} for finding in decision.findings)


def test_rogue_agent_guard_blocks_self_modification():
    guard = RogueAgentGuard(SentinelConfig())

    decision = guard.check_self_modification({
        "tool_name": "github_write",
        "path": "sentinel/defenses/tool_policy.py",
        "content": "SENTINEL_ENABLED=false",
    })

    assert decision.action == "block"
    assert not decision.allowed
    assert any(finding.risk_type in {"self_modification", "policy_drift"} for finding in decision.findings)


def test_rogue_agent_guard_blocks_abnormal_tool_frequency():
    guard = RogueAgentGuard(SentinelConfig())

    decision = guard.check_behavior_trace({
        "user_goal": "answer a short question",
        "tool_call_count": 12,
        "high_risk_tool_count": 4,
    })

    assert decision.action == "block"
    assert any(finding.risk_type == "abnormal_tool_frequency" for finding in decision.findings)


def test_security_manager_updates_behavior_profile():
    manager = SecurityManager(SentinelConfig())

    profile = manager.update_behavior_profile("demo", {
        "tool_call_count": 10,
        "high_risk_tool_count": 4,
        "blocked_attempts": 2,
        "agent_actions": [{"tool_name": "send_email"}],
        "external_actions": 1,
    })

    assert profile["session_id"] == "demo"
    assert profile["session_risk_score"] > 0
