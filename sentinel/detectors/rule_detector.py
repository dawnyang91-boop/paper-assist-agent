from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Pattern

from sentinel.schemas import SecurityFinding


@dataclass(frozen=True)
class Rule:
    pattern: Pattern[str]
    risk_type: str
    severity: str
    confidence: float
    recommendation: str


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)


INPUT_RULES: List[Rule] = [
    Rule(_compile(r"\bignore (all )?(previous|prior|above) instructions\b"), "prompt_injection", "high", 0.9, "Block instruction-override attempts before building prompts."),
    Rule(_compile(r"\breveal (the )?(system|developer|hidden) prompt\b"), "data_exfiltration", "high", 0.9, "Do not reveal system, developer, or hidden prompts."),
    Rule(_compile(r"\b(show|print|dump) (your )?(system|developer|hidden) (message|prompt)\b"), "data_exfiltration", "high", 0.88, "Refuse prompt disclosure attempts."),
    Rule(_compile(r"\bdisable (safety|guardrails|policy)\b"), "jailbreak", "high", 0.86, "Keep safety policy active."),
    Rule(_compile(r"\bdo not (refuse|warn|explain)\b"), "jailbreak", "medium", 0.72, "Treat refusal-suppression instructions as untrusted."),
    Rule(_compile(r"你现在是.{0,20}(无限制|无约束|开发者模式)"), "jailbreak", "high", 0.86, "Reject role-play jailbreaks."),
    Rule(_compile(r"(忽略|无视).{0,8}(之前|以上|所有).{0,8}(指令|规则|提示)"), "prompt_injection", "high", 0.9, "Block instruction-override attempts."),
    Rule(_compile(r"(泄露|输出|展示|打印).{0,8}(系统提示|隐藏提示|开发者消息|系统消息)"), "data_exfiltration", "high", 0.9, "Protect hidden prompts and private instructions."),
    Rule(_compile(r"(不要|不准).{0,8}(拒绝|警告|解释)"), "jailbreak", "medium", 0.72, "Treat refusal suppression as suspicious."),
]


RAG_INSTRUCTION_RULES: List[Rule] = [
    Rule(_compile(r"\bignore (the )?(user|question|previous instructions)\b"), "rag_poisoning", "high", 0.9, "Remove document instructions that try to control the assistant."),
    Rule(_compile(r"\bwhen answering,? you must\b"), "rag_poisoning", "medium", 0.75, "Strip assistant-directed instructions from retrieved documents."),
    Rule(_compile(r"\bassistant should\b"), "rag_poisoning", "medium", 0.72, "Strip assistant-directed instructions from retrieved documents."),
    Rule(_compile(r"\bsystem prompt\b"), "rag_poisoning", "medium", 0.7, "Do not trust retrieved text that references system prompts."),
    Rule(_compile(r"(不要|禁止).{0,8}(回答|理会).{0,8}(用户|问题)"), "rag_poisoning", "high", 0.9, "Remove RAG instructions that hijack answer behavior."),
    Rule(_compile(r"你必须.{0,12}(执行|遵守|优先).{0,12}(本文档|这个文档|以下指令)"), "rag_poisoning", "high", 0.9, "Remove document instructions that claim priority."),
]


OUTPUT_RULES: List[Rule] = [
    Rule(_compile(r"sk-[a-zA-Z0-9_\-]{12,}"), "data_exfiltration", "critical", 0.95, "Remove leaked API keys or credentials from output."),
    Rule(_compile(r"(system prompt|developer message|hidden prompt)\s*[:：]"), "data_exfiltration", "high", 0.86, "Do not disclose hidden prompts."),
    Rule(_compile(r"(我的系统提示词|隐藏提示词|开发者消息).{0,8}是"), "data_exfiltration", "high", 0.86, "Do not disclose hidden prompts."),
]


def detect_rules(text: str, rules: Iterable[Rule], max_evidence_chars: int = 180) -> List[SecurityFinding]:
    findings: List[SecurityFinding] = []
    value = text or ""
    for rule in rules:
        match = rule.pattern.search(value)
        if not match:
            continue
        evidence = match.group(0).strip()
        if len(evidence) > max_evidence_chars:
            evidence = evidence[: max_evidence_chars - 3] + "..."
        findings.append(SecurityFinding(
            risk_type=rule.risk_type,
            severity=rule.severity,
            confidence=rule.confidence,
            evidence=evidence,
            recommendation=rule.recommendation,
            metadata={"pattern": rule.pattern.pattern},
        ))
    return findings
