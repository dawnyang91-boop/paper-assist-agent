from typing import Any

from agent.agent_state import AgentState


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    """Optional LangGraph runner that exposes the assistant loop as named nodes."""

    def __init__(self, agent_graph: Any):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise LangGraphUnavailable("未安装 langgraph；请先安装 langgraph 后再启用该适配器。") from exc

        self.agent_graph = agent_graph
        self.StateGraph = StateGraph
        self.END = END
