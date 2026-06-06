from typing import Any


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    """Small LangGraph wrapper around the existing AgentGraph.

    AgentGraph uses AgentState with rich Python objects. Those objects are not
    safe for LangGraph's msgpack checkpoint serializer, so this adapter compiles
    without a checkpointer and keeps the graph state minimal.
    """

    def __init__(self, agent_graph: Any):
        try:
            from