from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SentinelConfig:
    enabled: bool = True
    input_guard_enabled: bool = True
    rag_guard_enabled: bool = True
    tool_guard_enabled: bool = True
    output_guard_enabled: bool = True
    block_threshold: float = 0.85
    warn_threshold: float = 0.55
    strip_rag_instructions: bool = True
    max_evidence_chars: int = 180
    llm_judge_enabled: bool = False
    llm_judge_model: Optional[str] = None
    llm_judge_api_key: Optional[str] = None
    llm_judge_base_url: Optional[str] = None
    llm_judge_max_tokens: int = 700
    llm_judge_temperature: float = 0.0

    @classmethod
    def from_app_config(cls, app_config: Any) -> "SentinelConfig":
        return cls(
            enabled=bool(getattr(app_config, "sentinel_enabled", True)),
            input_guard_enabled=bool(getattr(app_config, "sentinel_input_guard_enabled", True)),
            rag_guard_enabled=bool(getattr(app_config, "sentinel_rag_guard_enabled", True)),
            tool_guard_enabled=bool(getattr(app_config, "sentinel_tool_guard_enabled", True)),
            output_guard_enabled=bool(getattr(app_config, "sentinel_output_guard_enabled", True)),
            block_threshold=float(getattr(app_config, "sentinel_block_threshold", 0.85)),
            warn_threshold=float(getattr(app_config, "sentinel_warn_threshold", 0.55)),
            strip_rag_instructions=bool(getattr(app_config, "sentinel_strip_rag_instructions", True)),
            llm_judge_enabled=bool(getattr(app_config, "sentinel_llm_judge_enabled", False)),
            llm_judge_model=getattr(app_config, "sentinel_llm_judge_model", None) or getattr(app_config, "model_name", None),
            llm_judge_api_key=getattr(app_config, "sentinel_llm_judge_api_key", None) or getattr(app_config, "openai_api_key", None),
            llm_judge_base_url=getattr(app_config, "sentinel_llm_judge_base_url", None) or getattr(app_config, "openai_base_url", None),
            llm_judge_max_tokens=int(getattr(app_config, "sentinel_llm_judge_max_tokens", 700)),
            llm_judge_temperature=float(getattr(app_config, "sentinel_llm_judge_temperature", 0.0)),
        )
