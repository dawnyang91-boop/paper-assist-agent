from typing import Any


class LangGraphUnavailable(RuntimeError):
    pass


class LangGraphAgentRunner:
    """Optional LangGraph runner for the assistant loop.

    The legacy AgentGraph uses an AgentState dataclass with rich Python objects that
    are not msgpack-serializable. Therefore this adapter intentionally does not use
    LangGraph check