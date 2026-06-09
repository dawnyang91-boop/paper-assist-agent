from __future__ import annotations

from typing import Any, Dict, Optional


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    """Thin compatibility adapter around the existing AgentGraph.

    The project still keeps the legacy graph as the source of behavior. This adapter
    validates that LangGraph is installed and provides checkpoint-compatible config
    keys, then delegates execution to the legacy AgentGraph runner.
    """

    def __init__(self, agent_graph: Any):
        try:
            __import__("langgraph.graph")
        except ImportError as exc:
            raise LangGraphUnavailable(
                "LangGraph 未安装，无法启用 AGENT_RUNTIME=langgraph。请安装 langgraph 或切回 legacy/dag。"
            ) from exc
        self.agent_graph = agent_graph

    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            key: value
            for key, value in dict(state or {}).items()
            if not str(key).startswith("_")
        }
        checkpoint_config = self._checkpoint_config(payload, config=config)
        agent_state = self.agent_graph.run(**payload)
        payload["_agent_state"] = agent_state
        payload["_langgraph_config"] = checkpoint_config
        return payload

    def _checkpoint_config(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        checkpoint_config = dict(config or {})
        configurable = dict(checkpoint_config.get("configurable") or {})
        if not any(key in configurable for key in ("thread_id", "checkpoint_ns", "checkpoint_id")):
            session_id = str((state or {}).get("session_id") or "default")
            configurable["thread_id"] = session_id
        checkpoint_config["configurable"] = configurable
        return checkpoint_config
