from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from sentinel.config import SentinelConfig
from sentinel.defenses.input_guard import InputGuard
from sentinel.defenses.output_guard import OutputGuard
from sentinel.defenses.rag_sanitizer import RAGSanitizer
from sentinel.defenses.tool_policy import ToolPolicyChecker
from sentinel.detectors.llm_judge import LLMJudge
from sentinel.schemas import SecurityDecision


class SecurityManager:
    def __init__(self, config: SentinelConfig | None = None):
        self.config = config or SentinelConfig()
        self.input_guard = InputGuard(self.config)
        self.rag_sanitizer = RAGSanitizer(self.config)
        self.tool_policy = ToolPolicyChecker(self.config)
        self.output_guard = OutputGuard(self.config)
        self.llm_judge = LLMJudge(config=self.config)

    @classmethod
    def from_app_config(cls, app_config: Any) -> "SecurityManager":
        return cls(SentinelConfig.from_app_config(app_config))

    def pre_check_user_input(self, text: str) -> SecurityDecision:
        return self.input_guard.check(text)

    def sanitize_retrieved_docs(self, docs: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return self.rag_sanitizer.sanitize_docs(docs)

    def sanitize_ranked_chunks(self, ranked_chunks: Iterable[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
        return self.rag_sanitizer.sanitize_ranked_chunks(ranked_chunks)

    def check_tool_call(self, tool_name: str, args: Any = None, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        return self.tool_policy.check(tool_name, args=args, user_context=user_context)

    def post_check_output(self, answer: str) -> SecurityDecision:
        return self.output_guard.check(answer)

    def judge_case(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        return self.llm_judge.judge(payload)

    def trace_base(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "input_guard_enabled": self.config.input_guard_enabled,
            "rag_guard_enabled": self.config.rag_guard_enabled,
            "tool_guard_enabled": self.config.tool_guard_enabled,
            "output_guard_enabled": self.config.output_guard_enabled,
            "llm_judge_enabled": self.config.llm_judge_enabled,
            "blocked": False,
        }
