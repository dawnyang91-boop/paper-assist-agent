from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentMessage:
    message_id: str = ""
    task_id: str = ""
    sender_id: str = ""
    sender_role: str = ""
    receiver_id: str = ""
    receiver_role: str = ""
    content: str = ""
    content_type: str = "text"
    trust_level: str = "low"
    taint_labels: List[str] = field(default_factory=list)
    allowed_receivers: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    source_trace: Dict[str, Any] = field(default_factory=dict)
    claimed_role: str = ""
    origin: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any] | None, content: str = "") -> "AgentMessage":
        data = dict(value or {})
        return cls(
            message_id=str(data.get("message_id") or ""),
            task_id=str(data.get("task_id") or ""),
            sender_id=str(data.get("sender_id") or data.get("agent_id") or ""),
            sender_role=str(data.get("sender_role") or data.get("role") or data.get("claimed_role") or ""),
            receiver_id=str(data.get("receiver_id") or ""),
            receiver_role=str(data.get("receiver_role") or ""),
            content=str(data.get("content") or content or ""),
            content_type=str(data.get("content_type") or "text"),
            trust_level=str(data.get("trust_level") or "low"),
            taint_labels=[str(item) for item in data.get("taint_labels", [])],
            allowed_receivers=[str(item) for item in data.get("allowed_receivers", [])],
            allowed_actions=[str(item) for item in data.get("allowed_actions", [])],
            signature=data.get("signature"),
            source_trace=dict(data.get("source_trace") or {}),
            claimed_role=str(data.get("claimed_role") or ""),
            origin=str(data.get("origin") or ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
