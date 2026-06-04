from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sentinel.defenses.security_manager import SecurityManager
from sentinel.schemas import EvalCase


@dataclass
class TargetResponse:
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TargetAdapter:
    def ask(self, case: EvalCase) -> TargetResponse:
        raise NotImplementedError


class MockAgentAdapter(TargetAdapter):
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.security_manager = security_manager or SecurityManager()

    def ask(self, case: EvalCase) -> TargetResponse:
        input_decision = self.security_manager.pre_check_user_input(case.input)
        if not input_decision.allowed:
            return TargetResponse(
                answer="LLM-Sentinel blocked this input.",
                metadata={"sentinel": {"input_decision": input_decision.to_dict(), "blocked": True}},
            )
        output_decision = self.security_manager.post_check_output("正常回答：该请求未触发高风险规则。")
        return TargetResponse(
            answer=output_decision.sanitized_text or "正常回答：该请求未触发高风险规则。",
            metadata={"sentinel": {"input_decision": input_decision.to_dict(), "output_decision": output_decision.to_dict()}},
        )


class ChatbotApiAdapter(TargetAdapter):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/chatbot",
        session_id: str = "sentinel-eval",
        timeout_seconds: float = 60.0,
        api_token: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds
        self.api_token = api_token

    def ask(self, case: EvalCase) -> TargetResponse:
        payload = {
            "question": case.input,
            "session_id": self.session_id,
            "write_memory": False,
            "show_trace": True,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/ask",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        return TargetResponse(
            answer=str(parsed.get("answer", "")),
            metadata={"trace": parsed.get("trace", {}), "sources": parsed.get("sources", []), "memories": parsed.get("memories", [])},
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers


def build_adapter(name: str, **kwargs: Any) -> TargetAdapter:
    if name == "mock":
        return MockAgentAdapter()
    if name == "chatbot":
        return ChatbotApiAdapter(**kwargs)
    raise ValueError(f"Unknown target adapter: {name}")
