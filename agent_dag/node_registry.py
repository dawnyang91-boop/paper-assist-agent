from __future__ import annotations

from typing import Dict

from agent_dag.dag_node import DAGNode
from agent_dag.dag_policy import node_policy
from agent_dag.nodes.base_nodes import (
    MemoryRetrieverNode,
    MemoryWriterNode,
    PlannerNode,
    RAGRetrieverNode,
    SecurityNode,
    SkillNode,
    VerifierNode,
    WebSearchNode,
    WriterNode,
)


NODE_CLASSES = {
    "planner": PlannerNode,
    "rag": RAGRetrieverNode,
    "memory": MemoryRetrieverNode,
    "skill": SkillNode,
    "web": WebSearchNode,
    "security": SecurityNode,
    "writer": WriterNode,
    "verifier": VerifierNode,
    "memory_writer": MemoryWriterNode,
}


def build_node(node_id: str) -> DAGNode:
    policy = node_policy(node_id)
    node_cls = NODE_CLASSES.get(node_id, DAGNode)
    return node_cls(
        node_id=policy.node_id,
        role=policy.role,
        permissions=list(policy.permissions),
        allowed_tools=list(policy.allowed_tools),
        share_level=policy.share_level,
    )


def build_default_registry() -> Dict[str, DAGNode]:
    return {node_id: build_node(node_id) for node_id in NODE_CLASSES}
