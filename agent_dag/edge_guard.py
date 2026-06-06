from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Tuple

from agent_dag.dag_node import NodeResult
from agent_dag.evidence_store import EvidenceItem
from sentinel.agent_protocol import AgentMessage
from sentinel.runtime.evidence_policy import ShareLevel, normalize_share_level


class EdgeGuard:
    def __init__(self, security_manager: Any, task_id: str):
        self.security_manager = security_manager
        self.task_id = task_id

    def result_to_message(self, result: NodeResult, receiver: str) -> AgentMessage:
        content = self._message_content(result.output)
        return AgentMessage(
            message_id=str(uuid.uuid4()),
            task_id=self.task_id,
            sender_id=f"{result.node_id}_agent",
            sender_role=result.role,
            receiver_id=receiver,
            receiver_role=receiver,
            content=content,
            content_type="node_result",
            trust_level="medium" if result.ok else "low",
            taint_labels=list(result.metadata.get("taint_labels", [])),
            allowed_receivers=[receiver, "security", "verifier"],
            signature=f"dag:{self.task_id}:{result.node_id}",
            source_trace={"node_id": result.node_id},
            metadata={"status": result.status},
        )

    def check_result(self, result: NodeResult, receiver: str) -> Tuple[bool, Any, List[EvidenceItem]]:
        message = self.result_to_message(result, receiver)
        decision = self.security_manager.check_agent_message(message.to_dict())
        if not decision.allowed:
            return False, decision, []
        sanitized_text = decision.sanitized_text
        evidence = self._evidence_from_result(result, receiver, sanitized_text=sanitized_text)
        return True, decision, evidence

    def _evidence_from_result(self, result: NodeResult, receiver: str, sanitized_text: str | None = None) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        for item in result.evidence_items:
            if isinstance(item, EvidenceItem):
                items.append(item)
        if items:
            return items
        output = result.output
        if not output:
            return []
        content = sanitized_text if sanitized_text is not None else self._message_content(output)
        share_level = str(result.metadata.get("share_level") or ShareLevel.SANITIZED_FACTS.value)
        if normalize_share_level(share_level) in {ShareLevel.NO_SHARE, ShareLevel.NEVER_SHARE}:
            return []
        items.append(EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            task_id=self.task_id,
            source_node=result.node_id,
            source_type=result.role,
            content=self._message_content(output),
            sanitized_content=content,
            trust_level=str(result.metadata.get("trust_level") or "medium"),
            risk_score=float(result.metadata.get("risk_score", 0.0) or 0.0),
            taint_labels=list(result.metadata.get("taint_labels", [])),
            citations=list(result.metadata.get("citations", [])),
            allowed_receivers=[receiver, "verifier", "security"],
            metadata={**dict(result.metadata), **(output if isinstance(output, dict) else {})},
            share_level=share_level,
        ))
        return items

    def _message_content(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ("content", "answer", "summary", "text"):
                if output.get(key):
                    return str(output.get(key))
        if isinstance(output, Iterable) and not isinstance(output, (str, bytes, dict)):
            return "\n".join(str(item) for item in output)
        return str(output)
