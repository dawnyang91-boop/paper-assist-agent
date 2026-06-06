from typing import Any


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    """Small LangGraph wrapper around the existing AgentGraph.

    AgentGraph returns an AgentState object, which is not safe for LangGraph's
    msgpack checkpoint serializer. This adapter therefore compiles without a
    checkpointer and keeps the graph state minimal.
    """

    def __init__(self, agent_graph: Any):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise LangGraphUnavailable("未安装 langgraph；请先安装 langgraph 后再启用该适