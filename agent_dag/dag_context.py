from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List


REFERENCE_PRIVILEGED_KEYS = {
    "agent",
    "evidence_store",
    "stream_callback",
}


@dataclass
class ContextItem:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    taint_labels: List[str] = field(default_factory=list)
    trust_level: str = "low"


@dataclass
class AgentContext:
    task_id: str
    node_id: str
    role: str
    user_query: str
    user_context: Dict[str, Any] = field(default_factory=dict)
    private_context: List[ContextItem] = field(default_factory=list)
    shared_evidence_refs: List[str] = field(default_factory=list)
    privileged_context: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    trust_level: str = "low"
    taint_labels: List[str] = field(default_factory=lambda: ["user_untrusted"])

    def copy_for_node(
        self,
        node_id: str,
        role: str,
        permissions: List[str] | None = None,
        allowed_tools: List[str] | None = None,
        privileged_keys: List[str] | None = None,
    ) -> "AgentContext":
        allowed_privileged = set(privileged_keys or [])
        privileged = {
            key: value if key in REFERENCE_PRIVILEGED_KEYS or callable(value) else deepcopy(value)
            for key, value in self.privileged_context.items()
            if key in allowed_privileged
        }
        return AgentContext(
            task_id=self.task_id,
            node_id=node_id,
            role=role,
            user_query=self.user_query,
            user_context=deepcopy(self.user_context),
            private_context=deepcopy(self.private_context),
            shared_evidence_refs=list(self.shared_evidence_refs),
            privileged_context=privileged,
            permissions=list(permissions or []),
            allowed_tools=list(allowed_tools or []),
            trust_level=self.trust_level,
            taint_labels=list(self.taint_labels),
        )
