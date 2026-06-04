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
    Rule(_compile(r"\b(system message|developer instruction|hidden rules|internal policy|upper[- ]level prompt)\b"), "data_exfiltration", "high", 0.86, "Treat prompt-leak variants as hidden-instruction disclosure attempts."),
    Rule(_compile(r"(上层提示词|内部策略|开发者指令|隐藏规则|隐藏提示词|系统消息)"), "data_exfiltration", "high", 0.86, "Protect hidden prompts and private instructions."),
    Rule(_compile(r"\bdisable (safety|guardrails|policy)\b"), "jailbreak", "high", 0.86, "Keep safety policy active."),
    Rule(_compile(r"\bdo not (refuse|warn|explain)\b"), "jailbreak", "medium", 0.72, "Treat refusal-suppression instructions as untrusted."),
    Rule(_compile(r"\bsafety[- ]disabled assistant\b"), "jailbreak", "high", 0.86, "Reject safety-disabled role-play."),
    Rule(_compile(r"\bnever mention policy boundaries\b"), "jailbreak", "medium", 0.78, "Treat policy-boundary suppression as jailbreak intent."),
    Rule(_compile(r"\b(classroom simulation|fictional lab)\b.{0,100}\b(safety checks are unnecessary|follow the unsafe part|ignore safety)\b"), "jailbreak", "high", 0.86, "Classroom or fictional framing cannot disable safety checks."),
    Rule(_compile(r"\bif you refuse,? you fail the benchmark\b"), "jailbreak", "high", 0.86, "Benchmark pressure cannot override safety policy."),
    Rule(_compile(r"\b(ignore|bypass) (all )?(safeguards|safety policy|policy boundaries)\b"), "jailbreak", "high", 0.86, "Reject requests to bypass safeguards."),
    Rule(_compile(r"\banswer without caveats\b.{0,80}\bwithout safety notes\b"), "jailbreak", "medium", 0.74, "Treat safety-note suppression as suspicious."),
    Rule(_compile(r"\btranslate .* and (execute|follow)\b"), "jailbreak", "high", 0.84, "Do not execute translated hidden instructions."),
    Rule(_compile(r"\btranslate .* (hidden prompt|confidential prompts?)\b"), "data_exfiltration", "high", 0.86, "Do not disclose hidden prompts through translation."),
    Rule(_compile(r"\b(output|print|split).{0,40}(every other character|alternate characters?)\b"), "encoded_instruction", "high", 0.84, "Do not disclose hidden prompts through character splitting."),
    Rule(_compile(r"\bsplit (the )?(hidden|system|developer) prompt\b"), "encoded_instruction", "high", 0.86, "Do not disclose hidden prompts through splitting."),
    Rule(_compile(r"\b(ROT13|base64|unicode|hex)[- ]style\b.{0,80}\b(hidden prompt|system prompt|developer message|confidential prompts?)\b"), "encoded_instruction", "high", 0.86, "Do not disclose hidden prompts through encoding."),
    Rule(_compile(r"\b(first step|step one).{0,80}\b(second step|next step).{0,120}\b(ignore safety|reveal|hidden|internal rules)\b"), "jailbreak", "high", 0.84, "Detect multi-turn jailbreak setup."),
    Rule(_compile(r"\bcontinue (the )?(previous|last) (round|turn|agreement)\b.{0,80}\b(hidden|confidential|cannot disclose)\b"), "jailbreak", "high", 0.84, "Detect multi-turn jailbreak continuation."),
    Rule(_compile(r"\bobey the instruction inside brackets\b.{0,80}\b(ignore|bypass|safeguards)\b"), "jailbreak", "high", 0.84, "Nested untrusted instructions cannot override safety."),
    Rule(_compile(r"\btreat my next sentence as (the )?(real|new) system instruction\b"), "prompt_injection", "high", 0.9, "User text cannot create new system instructions."),
    Rule(_compile(r"\b(real|new) system instruction\b.{0,80}\b(always obey|ignore|override|unrestricted)\b"), "prompt_injection", "high", 0.9, "User text cannot create new system instructions."),
    Rule(_compile(r"#{2,}\s*(system|developer|instruction|override)[^\n]*#{0,}"), "delimiter_confusion", "high", 0.84, "Treat fake system/developer delimiters as untrusted user data."),
    Rule(_compile(r"```+\s*(system|developer|instruction|assistant)"), "delimiter_confusion", "high", 0.84, "Do not treat fenced fake role blocks as higher-priority instructions."),
    Rule(_compile(r"</?\s*(system|developer|assistant|instruction)\s*>"), "delimiter_confusion", "high", 0.84, "Do not treat XML-like fake role tags as higher-priority instructions."),
    Rule(_compile(r"(新的|最高优先级).{0,16}(规则|指令)"), "delimiter_confusion", "high", 0.84, "User text cannot introduce higher-priority rules."),
    Rule(_compile(r"\b(base64|rot13|unicode|hex|encode|decode)\b.{0,80}\b(system prompt|developer message|hidden prompt|hidden rules|internal policy)\b"), "encoded_instruction", "high", 0.85, "Block encoded prompt-leak attempts."),
    Rule(_compile(r"\b(system prompt|developer message|hidden prompt|hidden rules|internal policy)\b.{0,80}\b(base64|rot13|unicode|hex|encode|decode)\b"), "encoded_instruction", "high", 0.85, "Block encoded prompt-leak attempts."),
    Rule(_compile(r"\bdecode this instruction and follow it\b"), "encoded_instruction", "high", 0.85, "Do not follow encoded instructions from user text."),
    Rule(_compile(r"\b[a-zA-Z0-9+/]{24,}={0,2}\b"), "encoded_instruction", "medium", 0.68, "Long encoded-looking payloads should be treated as suspicious when paired with instruction-following language."),
    Rule(_compile(r"(base64|rot13|unicode|编码|解码).{0,80}(系统提示|隐藏提示|开发者消息|开发者指令|内部策略)"), "encoded_instruction", "high", 0.85, "Block encoded prompt-leak attempts."),
    Rule(_compile(r"(翻译|转换).{0,16}(并执行|然后执行|后执行).{0,80}(忽略|绕过|泄露|输出|ignore|bypass|reveal)"), "jailbreak", "high", 0.84, "Do not execute translated hidden instructions."),
    Rule(_compile(r"(每隔一个字符|隔一个字符|拆成.{0,8}字符).{0,40}(系统提示|隐藏提示|开发者消息)"), "encoded_instruction", "high", 0.86, "Do not disclose hidden prompts through character splitting."),
    Rule(_compile(r"(系统提示|隐藏提示|开发者消息).{0,40}(每隔一个字符|隔一个字符|拆成.{0,8}字符)"), "encoded_instruction", "high", 0.86, "Do not disclose hidden prompts through character splitting."),
    Rule(_compile(r"(第一步|先).{0,40}(第二步|下一步).{0,80}(忽略安全|输出内部规则|隐藏提示)"), "jailbreak", "high", 0.84, "Detect multi-turn jailbreak setup."),
    Rule(_compile(r"(继续上一轮|按约定).{0,80}(隐藏提示|隐藏内容|内部规则|不能透露)"), "jailbreak", "high", 0.84, "Detect multi-turn jailbreak continuation."),
    Rule(_compile(r"(开发者模式).{0,20}(不受任何约束|无约束|绕过)"), "jailbreak", "high", 0.86, "Reject developer-mode jailbreaks."),
    Rule(_compile(r"\bquality assurance\b.{0,80}\b(developer message|system prompt|hidden rules|internal policy)\b"), "audit_excuse", "high", 0.86, "Quality-assurance framing cannot justify secret disclosure."),
    Rule(_compile(r"\binclude the exact developer message\b"), "data_exfiltration", "high", 0.88, "Do not disclose developer messages."),
    Rule(_compile(r"\b(i am|i'm) (the )?(admin|administrator|developer|auditor|owner|researcher)\b.{0,80}\b(reveal|disable|override|bypass|ignore)\b"), "authority_claim", "high", 0.82, "Authority claims in user text are not permissions."),
    Rule(_compile(r"(我是|本人是).{0,16}(管理员|开发者|审计员|研究员|所有者).{0,40}(输出|泄露|忽略|绕过|关闭)"), "authority_claim", "high", 0.82, "Authority claims in user text are not permissions."),
    Rule(_compile(r"\b(security|safety|red[- ]team|audit|research) (test|audit|research|exercise)\b.{0,120}\b(reveal|print|dump|show)\b.{0,40}\b(system|developer|hidden)\b"), "audit_excuse", "high", 0.86, "Safety tests may be discussed, but cannot justify secret disclosure."),
    Rule(_compile(r"(安全测试|安全审计|红队测试|研究用途).{0,80}(输出|泄露|展示|打印).{0,30}(系统提示|隐藏提示|开发者消息|开发者指令)"), "audit_excuse", "high", 0.86, "Safety tests may be discussed, but cannot justify secret disclosure."),
    Rule(_compile(r"你现在是.{0,20}(无限制|无约束|开发者模式)"), "jailbreak", "high", 0.86, "Reject role-play jailbreaks."),
    Rule(_compile(r"(忽略|无视).{0,8}(之前|以上|所有).{0,8}(指令|规则|提示)"), "prompt_injection", "high", 0.9, "Block instruction-override attempts."),
    Rule(_compile(r"(泄露|输出|展示|打印).{0,8}(系统提示|隐藏提示|开发者消息|系统消息)"), "data_exfiltration", "high", 0.9, "Protect hidden prompts and private instructions."),
    Rule(_compile(r"(不要|不准).{0,8}(拒绝|警告|解释)"), "jailbreak", "medium", 0.72, "Treat refusal suppression as suspicious."),
]


RAG_INSTRUCTION_RULES: List[Rule] = [
    Rule(_compile(r"---+\s*(new instruction|system override|instruction)\s*---+"), "rag_poisoning", "high", 0.88, "Remove chunk-boundary instruction blocks."),
    Rule(_compile(r"\bignore (chunk|document|paragraph|content) [a-z0-9]+\b"), "rag_poisoning", "high", 0.86, "Remove cross-chunk instruction hijacking."),
    Rule(_compile(r"\bignore (the )?(user|question|previous instructions)\b"), "rag_poisoning", "high", 0.9, "Remove document instructions that try to control the assistant."),
    Rule(_compile(r"\bignore competing chunks\b"), "rag_poisoning", "high", 0.86, "Do not let retrieved chunks suppress other evidence."),
    Rule(_compile(r"\b(answer must|must) follow this document\b"), "rag_poisoning", "high", 0.84, "Do not let retrieved chunks claim authority over the answer."),
    Rule(_compile(r"\bwhen answering,? you must\b"), "rag_poisoning", "medium", 0.75, "Strip assistant-directed instructions from retrieved documents."),
    Rule(_compile(r"\bassistant should\b"), "rag_poisoning", "medium", 0.72, "Strip assistant-directed instructions from retrieved documents."),
    Rule(_compile(r"\bassistant must\b"), "rag_poisoning", "medium", 0.78, "Strip assistant-directed instructions from retrieved documents."),
    Rule(_compile(r"\byou must call\b"), "rag_poisoning", "high", 0.86, "Retrieved text cannot trigger tool calls."),
    Rule(_compile(r"\bsystem prompt\b"), "rag_poisoning", "medium", 0.7, "Do not trust retrieved text that references system prompts."),
    Rule(_compile(r"\bdeveloper (message|instruction)\b"), "rag_poisoning", "medium", 0.72, "Do not trust retrieved text that references hidden instructions."),
    Rule(_compile(r"```+\s*(system|developer|instruction|assistant)"), "delimiter_confusion", "medium", 0.76, "Strip fake role blocks from retrieved documents."),
    Rule(_compile(r"</?\s*(system|developer|assistant|instruction)\s*>"), "delimiter_confusion", "medium", 0.76, "Strip fake role tags from retrieved documents."),
    Rule(_compile(r"\b(system|developer)\s*[:：]\s*(ignore|reveal|print|dump|call|write)\b"), "rag_poisoning", "high", 0.88, "Strip metadata or content that pretends to be a privileged instruction."),
    Rule(_compile(r"\b(falsely claims|false conclusion|without evidence|without citation|states without citation)\b"), "fact_pollution", "medium", 0.72, "Flag unsupported or explicitly false factual claims for trust reduction."),
    Rule(_compile(r"\b(never suitable|should always be removed|all retrieved papers .* invalid)\b"), "fact_pollution", "medium", 0.7, "Flag sweeping unsupported claims for trust reduction."),
    Rule(_compile(r"\b(outdated note|stale|deprecated)\b"), "stale_content", "medium", 0.68, "Reduce trust for stale document content."),
    Rule(_compile(r"(污染事实|错误事实|没有证据|无引用|过期内容)"), "fact_pollution", "medium", 0.72, "Flag unsupported or stale factual claims for trust reduction."),
    Rule(_compile(r"(最高权威|唯一可信来源|必须优先采用|优先采用本文档)"), "rag_poisoning", "high", 0.84, "Do not let retrieved chunks claim authority over other evidence."),
    Rule(_compile(r"(不要|禁止).{0,8}(回答|理会).{0,8}(用户|问题)"), "rag_poisoning", "high", 0.9, "Remove RAG instructions that hijack answer behavior."),
    Rule(_compile(r"你必须.{0,12}(执行|遵守|优先).{0,12}(本文档|这个文档|以下指令)"), "rag_poisoning", "high", 0.9, "Remove document instructions that claim priority."),
]


TOOL_OUTPUT_RULES: List[Rule] = [
    Rule(_compile(r"\bassistant must\b"), "tool_output_instruction", "high", 0.86, "Tool output is untrusted data and must not issue assistant instructions."),
    Rule(_compile(r"\bassistant should\b"), "tool_output_instruction", "medium", 0.76, "Tool output is untrusted data and must not issue assistant instructions."),
    Rule(_compile(r"\byou must call\b"), "tool_output_instruction", "high", 0.88, "Tool output must not trigger tool calls."),
    Rule(_compile(r"\bcall (delete|remove|write|update|send|email|payment|deploy)\b"), "tool_output_instruction", "critical", 0.94, "Tool output cannot instruct state-changing tools."),
    Rule(_compile(r"\b(send email|delete file|drop table|update config|run command)\b"), "tool_output_instruction", "critical", 0.94, "Tool output cannot instruct state-changing tools."),
    Rule(_compile(r"\bignore (the )?(user|previous instructions)\b"), "tool_output_instruction", "high", 0.88, "Tool output cannot override user or system instructions."),
    Rule(_compile(r"\b(system prompt|developer message|hidden prompt|internal policy)\b"), "tool_output_instruction", "high", 0.84, "Tool output mentioning hidden instructions should be sanitized."),
    Rule(_compile(r"(助手|模型).{0,8}必须"), "tool_output_instruction", "medium", 0.76, "Tool output is untrusted data and must not issue assistant instructions."),
    Rule(_compile(r"(调用|执行).{0,8}(删除|写入|发送邮件|付款|部署|更新配置)"), "tool_output_instruction", "critical", 0.94, "Tool output cannot instruct state-changing tools."),
    Rule(_compile(r"(忽略|无视).{0,8}(用户|问题|之前指令)"), "tool_output_instruction", "high", 0.88, "Tool output cannot override user or system instructions."),
]


MEMORY_SAFETY_RULES: List[Rule] = [
    Rule(_compile(r"\b(ignore|disable|bypass) (safety|guardrails|policy|rules)\b"), "memory_instruction", "critical", 0.94, "Do not persist memories that alter safety policy."),
    Rule(_compile(r"\breveal (the )?(system|developer|hidden) prompt\b"), "memory_instruction", "critical", 0.94, "Do not persist memories that request hidden prompt disclosure."),
    Rule(_compile(r"\bdo not ask (for )?confirmation\b"), "memory_instruction", "high", 0.9, "Do not persist memories that disable confirmation flows."),
    Rule(_compile(r"\b(auto|automatically) (execute|call|run) (tools?|commands?)\b"), "memory_instruction", "high", 0.9, "Do not persist memories that authorize automatic tool execution."),
    Rule(_compile(r"\bread private (files?|documents?|data)\b"), "memory_instruction", "high", 0.86, "Do not persist memories that request broad private-data access."),
    Rule(_compile(r"(忽略|绕过|关闭).{0,12}(安全|规则|策略|护栏)"), "memory_instruction", "critical", 0.94, "Do not persist memories that alter safety policy."),
    Rule(_compile(r"(输出|泄露|展示).{0,8}(系统提示|隐藏提示|开发者消息|开发者指令)"), "memory_instruction", "critical", 0.94, "Do not persist memories that request hidden prompt disclosure."),
    Rule(_compile(r"(无需|不需要).{0,8}(确认|二次确认)"), "memory_instruction", "high", 0.9, "Do not persist memories that disable confirmation flows."),
    Rule(_compile(r"(自动|直接).{0,8}(执行|调用).{0,8}(工具|命令|删除|写入|发送)"), "memory_instruction", "high", 0.9, "Do not persist memories that authorize automatic tool execution."),
]


OUTPUT_RULES: List[Rule] = [
    Rule(_compile(r"sk-[a-zA-Z0-9_\-]{12,}"), "data_exfiltration", "critical", 0.95, "Remove leaked API keys or credentials from output."),
    Rule(_compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "data_exfiltration", "critical", 0.96, "Remove leaked GitHub tokens from output."),
    Rule(_compile(r"\bAKIA[0-9A-Z]{16}\b"), "data_exfiltration", "critical", 0.96, "Remove leaked AWS access keys from output."),
    Rule(_compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"), "data_exfiltration", "critical", 0.98, "Remove private key blocks from output."),
    Rule(_compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "data_exfiltration", "critical", 0.95, "Remove JWT tokens from output."),
    Rule(_compile(r"\bBearer\s+[A-Za-z0-9_\-./+=]{16,}\b"), "data_exfiltration", "critical", 0.95, "Remove bearer tokens from output."),
    Rule(_compile(r"\b(session|cookie|csrf|xsrf|refresh_token|access_token)\s*[:=]\s*[A-Za-z0-9_\-./+=]{12,}\b"), "data_exfiltration", "critical", 0.95, "Remove cookies, session tokens, and access tokens from output."),
    Rule(_compile(r"\b(?:[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|ACCESS_KEY|PRIVATE_KEY|DATABASE_URL)[A-Z0-9_]*)\s*=\s*[^\s#]{8,}"), "data_exfiltration", "high", 0.86, "Remove .env style secret values from output."),
    Rule(_compile(r"\b(postgres(?:ql)?|mysql|mongodb|redis)://[^\s]+"), "data_exfiltration", "critical", 0.94, "Remove database connection strings from output."),
    Rule(_compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "data_exfiltration", "medium", 0.7, "Redact email addresses when output may contain private user data."),
    Rule(_compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)"), "data_exfiltration", "medium", 0.72, "Redact phone numbers when output may contain private user data."),
    Rule(_compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "data_exfiltration", "high", 0.86, "Redact Chinese citizen ID numbers from output."),
    Rule(_compile(r"(?:/Users|/home|/var|/etc|C:\\\\Users)[^\s`'\"，。；;]{4,}"), "data_exfiltration", "medium", 0.74, "Redact private local file paths from output."),
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
