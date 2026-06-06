from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent_dag.dag_context import AgentContext


@dataclass
class NodeResult:
    node_id: str
    role: str
    status: str = "succeeded"
    output: Any = None
    evidence_items: List[Any] = field(default_factory=list)
    messages: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


@dataclass
class DAGNode:
    node_id: str
    role: str
    permissions: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    share_level: str = "no_share"
    input_policy: Dict[str, Any] = field(default_factory=dict)
    output_policy: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 15.0
    max_retries: int = 0

    async def run(self, context: AgentContext) -> NodeResult:
        started = time.perf_counter()
        try:
            output = await self._run(context)
            return NodeResult(
                node_id=self.node_id,
                role=self.role,
                output=output,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return NodeResult(
                node_id=self.node_id,
                role=self.role,
                status="failed",
                errors=[{"node": self.node_id, "error": str(exc)}],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    async def _run(self, context: AgentContext) -> Any:
        return {}
