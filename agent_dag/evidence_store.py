from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List

from sentinel.runtime.evidence_policy import ShareLevel, can_share_with, is_high_risk_taint, normalize_share_level


@dataclass
class EvidenceItem:
    evidence_id: str
    task_id: str
    source_node: str
    source_type: str
    content: str
    sanitized_content: str
    trust_level: str = "low"
    risk_score: float = 0.0
    taint_labels: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    allowed_receivers: List[str] = field(default_factory=lambda: ["writer", "verifier", "security"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    share_level: str = ShareLevel.SANITIZED_FACTS.value
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SharedEvidenceStore:
    def __init__(self):
        self._items: Dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> EvidenceItem:
        level = normalize_share_level(item.share_level)
        if level == ShareLevel.NEVER_SHARE:
            return item
        self._items[item.evidence_id] = item
        return item

    def create(
        self,
        task_id: str,
        source_node: str,
        source_type: str,
        content: str,
        sanitized_content: str | None = None,
        **kwargs: Any,
    ) -> EvidenceItem:
        item = EvidenceItem(
            evidence_id=str(kwargs.pop("evidence_id", uuid.uuid4())),
            task_id=task_id,
            source_node=source_node,
            source_type=source_type,
            content=content,
            sanitized_content=sanitized_content if sanitized_content is not None else content,
            **kwargs,
        )
        return self.add(item)

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._items.get(evidence_id)

    def all(self) -> List[EvidenceItem]:
        return list(self._items.values())

    def visible_to(self, receiver: str, include_high_risk: bool = False) -> List[EvidenceItem]:
        visible = []
        for item in self._items.values():
            if not can_share_with(item.share_level, receiver, item.allowed_receivers):
                continue
            if receiver == "writer":
                level = normalize_share_level(item.share_level)
                if level not in {ShareLevel.SANITIZED_FACTS, ShareLevel.CITED_EVIDENCE}:
                    continue
            if not include_high_risk and (item.risk_score >= 0.85 or is_high_risk_taint(item.taint_labels)):
                continue
            visible.append(item)
        return visible

    def refs_for(self, receiver: str) -> List[str]:
        return [item.evidence_id for item in self.visible_to(receiver)]

    def to_trace(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self._items.values()]

    def extend(self, items: Iterable[EvidenceItem]) -> None:
        for item in items:
            self.add(item)
