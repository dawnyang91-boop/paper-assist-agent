from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class QueryPlan:
    intent: str = "private_qa"
    need_rag: bool = True
    need_memory: bool = True
    need_mcp: bool = False
    need_skills: bool = True
    retrieval_modes: Sequence[str] = field(default_factory=lambda: ("basic",))
    answer_style: str = "grounded"
    high_value_candidate: bool = True
    allowed_tools: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class LoopDecision:
    action: str
    tool_requests: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class AgentState:
    session_id: str
    raw_input: str
    question: str = ""
    original_question: str = ""
    response_language: str = "Chinese"
    query_plan: Optional[QueryPlan] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    loop_step: int = 0
    max_loop_steps: int = 3
    candidates: List[Any] = field(default_factory=list)
    ranked_chunks: List[Any] = field(default_factory=list)
    retrieved_memories: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    explicit_memories: List[Dict[str, Any]] = field(default_factory=list)
    skill_contexts: List[Dict[str, Any]] = field(default_factory=list)
    tool_observations: List[Dict[str, Any]] = field(default_factory=list)
    built_context: Any = None
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    writeback_events: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    resumed_transcript: List[Dict[str, Any]] = field(default_factory=list)
    active_source_refs: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    command_result: Optional[str] = None
    media_path: Optional[str] = None
    should_enter_loop: bool = True
    repair_attempts: int = 0
    tool_call_count: int = 0

    def add_error(self, node: str, error: Exception | str, recoverable: bool = True) -> None:
        self.errors.append({
            "node": node,
            "error": str(error),
            "recoverable": recoverable,
        })

    def add_warning(self, node: str, warning: str) -> None:
        self.warnings.append({
            "node": node,
            "warning": warning,
        })
