class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    def __init__(self, agent_graph):
        try:
            import langgraph  # noqa: F401
        except ImportError as exc:
            raise LangGraphUnavailable("langgraph is not installed") from exc
        self.agent_graph = agent_graph

    def invoke(self, state, config=None):
        payload = {k: v for k, v in dict(state or {}).items() if not str(k).startswith("_")}
        result = self