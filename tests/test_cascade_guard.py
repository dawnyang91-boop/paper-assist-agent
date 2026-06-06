from sentinel.defenses.cascade_guard import CascadeGuard
from sentinel.defenses.security_manager import SecurityManager
from sentinel.config import SentinelConfig


def test_cascade_guard_skips_tainted_memory_write():
    guard = CascadeGuard(SentinelConfig())

    decision = guard.check_memory_write_taint(
        "Should Redis be removed?",
        "Redis must be removed because the retrieved document says so.",
        {
            "poisoned_context": True,
            "risky_rag_chunks": ["D1"],
            "answer_used_risky_context": True,
            "memory_write_requested": True,
        },
    )

    assert decision.action == "skip_memory_write"
    assert not decision.allowed
    assert any(finding.risk_type == "poisoned_rag_to_answer_to_memory" for finding in decision.findings)


def test_cascade_guard_reduces_trust_for_fallback_with_local_citation():
    guard = CascadeGuard(SentinelConfig())

    decision = guard.check_answer_taint(
        "According to [D1], the exact answer is guaranteed.",
        {
            "local_context_insufficient": True,
            "general_knowledge_fallback_used": True,
        },
    )

    assert decision.action == "reduce_trust"
    assert decision.allowed
    assert any(finding.risk_type == "general_knowledge_overtrust" for finding in decision.findings)


def test_security_manager_combines_memory_safety_and_cascade_trace():
    manager = SecurityManager(SentinelConfig())

    allowed, decisions = manager.should_write_memory_with_trace(
        "Can this session summary be reused?",
        "A memory from another session grants filesystem permissions.",
        {
            "cross_session_source": True,
            "source_session_id": "session-a",
            "target_session_id": "session-b",
            "risky_memory_contexts": ["M1"],
            "memory_write_requested": True,
        },
    )

    assert not allowed
    assert any(item.get("action") == "skip_memory_write" for item in decisions)


def test_cascade_guard_warns_on_bad_tool_output_to_action():
    guard = CascadeGuard(SentinelConfig())

    decision = guard.check_tool_output_cascade(
        "Tool output incorrectly says deployment succeeded and recommends deleting rollback files."
    )

    assert decision.action == "warn"
    assert decision.allowed
    assert any(finding.risk_type == "bad_tool_output_to_decision" for finding in decision.findings)
