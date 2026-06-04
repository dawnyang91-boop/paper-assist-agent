from typing import Any, Dict, List, Optional, Sequence, TypedDict


class LangGraphQAState(TypedDict, total=False):
    question: str
    session_id: str
    user_profile: Dict[str, Any]
    include_memory_manager: bool
    include_mcp: bool
    include_skills: bool
    write_memory: bool
    memory_top_k: int
    modes: Optional[Sequence[str]]
    top_k: Optional[int]
    per_query_limit: Optional[int]
    mqe_count: Optional[int]
    rerank_top_k: Optional[int]
    context_top_k: Optional[int]
    answer: str
    metadata: Dict[str, Any]
    errors: List[Dict[str, Any]]
    stop_reason: str
    _agent_state: Any
    _decision: Any
