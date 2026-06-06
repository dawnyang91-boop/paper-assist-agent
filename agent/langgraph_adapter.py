from typing import Any


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    """Compatibility runner for AGENT_RUNTIME=langgraph.

    The project AgentGraph stores rich AgentState objects that are not msgpack
    serializable by LangGraph's MemorySaver. This runner keeps the public
    langgraph adapter interface but delegates to AgentGraph.run without using
    LangGraph checkpoint serialization