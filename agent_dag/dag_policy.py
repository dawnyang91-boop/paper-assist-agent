from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from sentinel.runtime.evidence_policy import ShareLevel, normalize_share_level


@dataclass
class NodePolicy:
    node_id: str
    role: str
    permissions: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    share_level: str = ShareLevel.NO_SHARE.value
    privileged_keys: List[str] = field(default_factory=list)


@dataclass
class EdgePolicy:
    sender: str
    receiver: str
    allow: bool = True
    require_sanitized: bool = True
    max_share_level: str = ShareLevel.SANITIZED_FACTS.value


class DAGEvidencePolicy:
    WRITER_ALLOWED = {ShareLevel.SANITIZED_FACTS, ShareLevel.CITED_EVIDENCE}
    VERIFIER_ALLOWED = {ShareLevel.SANITIZED_FACTS, ShareLevel.CITED_EVIDENCE, ShareLevel.PRIVILEGED}
    SECURITY_ALLOWED = {ShareLevel.SANITIZED_FACTS, ShareLevel.CITED_EVIDENCE, ShareLevel.PRIVILEGED}

    def can_read(self, receiver_role: str, share_level: str, risk_score: float = 0.0) -> bool:
        level = normalize_share_level(share_level)
        if level == ShareLevel.NEVER_SHARE:
            return False
        if risk_score >= 0.85 and receiver_role not in {"security", "verifier"}:
            return False
        if receiver_role == "writer":
            return level in self.WRITER_ALLOWED
        if receiver_role == "verifier":
            return level in self.VERIFIER_ALLOWED
        if receiver_role == "security":
            return level in self.SECURITY_ALLOWED
        if receiver_role == "memory_writer":
            return level in {ShareLevel.SANITIZED_FACTS, ShareLevel.CITED_EVIDENCE}
        return level in {ShareLevel.SANITIZED_FACTS, ShareLevel.CITED_EVIDENCE}


DEFAULT_NODE_POLICIES: Dict[str, NodePolicy] = {
    "planner": NodePolicy("planner", "planner", share_level=ShareLevel.SANITIZED_FACTS.value),
    "rag": NodePolicy("rag", "retriever", permissions=["read_rag"], allowed_tools=["qdrant_search"], share_level=ShareLevel.CITED_EVIDENCE.value),
    "memory": NodePolicy("memory", "memory", permissions=["read_session_memory"], allowed_tools=["memory_read"], share_level=ShareLevel.PRIVILEGED.value),
    "skill": NodePolicy("skill", "skill", permissions=["run_skill"], allowed_tools=["skill"], share_level=ShareLevel.SANITIZED_FACTS.value),
    "web": NodePolicy("web", "external_retriever", permissions=["web_search", "fetch_public_url"], allowed_tools=["brave_search", "fetch"], share_level=ShareLevel.SANITIZED_FACTS.value),
    "security": NodePolicy("security", "security", permissions=["security_audit"], share_level=ShareLevel.PRIVILEGED.value),
    "writer": NodePolicy("writer", "writer", permissions=["llm_generate"], share_level=ShareLevel.NO_SHARE.value),
    "verifier": NodePolicy("verifier", "verifier", permissions=["read_evidence", "verify_answer"], share_level=ShareLevel.NO_SHARE.value),
    "memory_writer": NodePolicy("memory_writer", "memory_writer", permissions=["write_memory"], allowed_tools=["memory_write"], share_level=ShareLevel.PRIVILEGED.value),
}


def node_policy(node_id: str) -> NodePolicy:
    return DEFAULT_NODE_POLICIES.get(node_id, NodePolicy(node_id, node_id))


def edge_policy(sender: str, receiver: str) -> EdgePolicy:
    if sender == "writer" and receiver == "tool_executor":
        return EdgePolicy(sender, receiver, allow=False)
    return EdgePolicy(sender, receiver)


def load_policy_dict() -> Dict[str, Any]:
    return {
        "nodes": {key: value.__dict__ for key, value in DEFAULT_NODE_POLICIES.items()},
        "edges": {},
    }
