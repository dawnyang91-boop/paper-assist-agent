from typing import Any


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    def __init__(self, agent_graph: Any):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise LangGraphUnavailable("langgraph is not installed") from exc
        self.agent_graph = agent_graph
        self.END = END
