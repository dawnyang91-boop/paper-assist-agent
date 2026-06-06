from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class DAGNodeTrace:
    node_id: str
    role: str
    status: str
    latency_ms: float
    input_risk_score: float = 0.0
    output_risk_score: float = 0.0
    evidence_written: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DAGEdgeTrace:
    sender: str
    receiver: str
    allowed: bool
    decision: str
    risk_types: List[str] = field(default_factory=list)


@dataclass
class DAGTrace:
    task_id: str
    started_at: float = field(default_factory=time.time)
    nodes: List[DAGNodeTrace] = field(default_factory=list)
    edges: List[DAGEdgeTrace] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    blocked_messages: List[Dict[str, Any]] = field(default_factory=list)
    taint_trace: List[Dict[str, Any]] = field(default_factory=list)
    risk_summary: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, result: Any, evidence_written: int = 0, output_risk_score: float = 0.0) -> None:
        self.nodes.append(DAGNodeTrace(
            node_id=result.node_id,
            role=result.role,
            status=result.status,
            latency_ms=result.latency_ms,
            output_risk_score=output_risk_score,
            evidence_written=evidence_written,
            errors=list(result.errors or []),
        ))

    def add_edge(self, sender: str, receiver: str, decision: Any) -> None:
        risk_types = [finding.risk_type for finding in getattr(decision, "findings", [])]
        self.edges.append(DAGEdgeTrace(
            sender=sender,
            receiver=receiver,
            allowed=bool(getattr(decision, "allowed", True)),
            decision=str(getattr(decision, "action", "allow")),
            risk_types=risk_types,
        ))
        if not getattr(decision, "allowed", True):
            self.blocked_messages.append({
                "sender": sender,
                "receiver": receiver,
                "decision": decision.to_dict() if hasattr(decision, "to_dict") else str(decision),
            })

    def to_dict(self, evidence_count: int = 0) -> Dict[str, Any]:
        finished = time.time()
        return {
            "nodes": [asdict(item) for item in self.nodes],
            "edges": [asdict(item) for item in self.edges],
            "parallel_groups": list(self.parallel_groups),
            "execution_time_ms": round((finished - self.started_at) * 1000, 2),
            "blocked_messages": list(self.blocked_messages),
            "taint_trace": list(self.taint_trace),
            "evidence_count": evidence_count,
            "risk_summary": dict(self.risk_summary),
        }
