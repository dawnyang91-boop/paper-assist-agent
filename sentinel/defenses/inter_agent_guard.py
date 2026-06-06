from __future__ import annotations

import re
from typing import Any, Dict, List

from sentinel.agent_protocol import AgentMessage
from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding


class InterAgentGuard:
    """Validate and sanitize messages crossing agent boundaries."""

    ROLE_ACTIONS = {
        "planner": {"assign_task", "ask_question", "read_context"},
        "worker": {"return_result", "read_context"},
        "reviewer": {"critique", "approve_read"},
        "writer": {"write_answer", "read_context"},
        "executor": {"read_context", "call_safe_tool"},
    }
    HIGH_PRIVILEGE_ACTIONS = {
        "deploy",
        "deploy_service",
        "send_email",
        "delete_resource",
        "file_write",
        "github_write",
        "run_command",
        "database_write",
    }
    INSTRUCTION_PATTERNS = (
        re.compile(r"\b(ignore|bypass|disable)\s+(safety|policy|previous instructions)\b", re.I),
        re.compile(r"\b(call|invoke|execute|run)\s+[\w_.-]*(deploy|delete|email|write|shell|command)", re.I),
        re.compile(r"\b(must|should|need to)\b.{0,40}\b(send|delete|deploy|write|email|execute|run)\b", re.I),
        re.compile(r"\b(system|developer)\s*(override|message|prompt)\b", re.I),
        re.compile(r"(忽略|绕过|禁用).{0,12}(安全|策略|规则|指令)"),
        re.compile(r"(调用|执行|运行).{0,16}(部署|删除|邮件|写入|命令)"),
    )

    def __init__(self, config: SentinelConfig):
        self.config = config
        self.registered_agents = {
            "planner_agent": "planner",
            "worker_agent": "worker",
            "reviewer_agent": "reviewer",
            "writer_agent": "writer",
            "executor": "executor",
            "rag_agent": "retriever",
            "memory_agent": "memory",
            "skill_agent": "skill",
            "web_agent": "web",
            "security_agent": "security",
            "verifier_agent": "verifier",
            "memory_writer_agent": "memory_writer",
        }

    def check_message(self, message: Dict[str, Any] | AgentMessage | None) -> SecurityDecision:
        if not self.config.enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)
        agent_message = message if isinstance(message, AgentMessage) else AgentMessage.from_dict(message or {})
        findings: List[SecurityFinding] = []
        findings.extend(self.verify_agent_identity(agent_message).findings)
        findings.extend(self.check_sender_receiver_policy(
            agent_message.sender_role,
            agent_message.receiver_role,
            self._requested_action(agent_message),
        ).findings)

        instruction_decision = self.sanitize_agent_output(agent_message)
        findings.extend(instruction_decision.findings)
        if any(finding.risk_type in {"spoofed_agent_message", "missing_message_authentication", "unauthorized_agent_request"} for finding in findings):
            return self._decision(findings, action="block", allowed=False)
        if instruction_decision.findings:
            return SecurityDecision(
                allowed=True,
                action="sanitize",
                risk_score=max(f.confidence for f in findings),
                findings=findings,
                sanitized_text=instruction_decision.sanitized_text,
                metadata={"message": agent_message.to_dict()},
            )
        return self._decision(findings, action="warn" if findings else "allow", allowed=True)

    def verify_agent_identity(self, message: AgentMessage | Dict[str, Any]) -> SecurityDecision:
        msg = message if isinstance(message, AgentMessage) else AgentMessage.from_dict(message)
        findings: List[SecurityFinding] = []
        if not msg.sender_id or not msg.sender_role or not msg.receiver_id:
            findings.append(self._finding(
                "missing_message_authentication",
                "high",
                0.88,
                msg.to_dict(),
                "Inter-agent messages require sender_id, sender_role, receiver_id, and origin metadata.",
            ))
        if not msg.signature:
            findings.append(self._finding(
                "missing_message_authentication",
                "medium",
                0.7,
                msg.sender_id or "unsigned",
                "Unsigned inter-agent messages must be treated as untrusted.",
            ))
        registered_role = self.registered_agents.get(msg.sender_id)
        claimed = msg.claimed_role or msg.sender_role
        if msg.sender_id and registered_role is None:
            findings.append(self._finding(
                "spoofed_agent_message",
                "high",
                0.9,
                msg.sender_id,
                "Unknown agent_id cannot issue trusted instructions.",
            ))
        if registered_role and claimed and registered_role != claimed:
            findings.append(self._finding(
                "forged_role_channel",
                "high",
                0.9,
                f"{msg.sender_id}:{claimed}",
                "Claimed role must match the registered role for the sender agent.",
            ))
        return self._decision(findings, action="block" if findings else "allow", allowed=not findings)

    def check_sender_receiver_policy(self, sender: str, receiver: str, action: str) -> SecurityDecision:
        findings: List[SecurityFinding] = []
        sender_role = (sender or "").lower()
        requested = (action or "").lower()
        allowed = self.ROLE_ACTIONS.get(sender_role, set())
        if requested in self.HIGH_PRIVILEGE_ACTIONS and requested not in allowed:
            findings.append(self._finding(
                "unauthorized_agent_request",
                "high",
                0.9,
                f"{sender_role}->{receiver}:{requested}",
                "Low-privilege agents cannot request high-impact external actions from another agent.",
            ))
        if sender_role in {"worker", "reviewer"} and requested not in {"", "return_result", "critique", "approve_read", "read_context"}:
            findings.append(self._finding(
                "unauthorized_agent_request",
                "high",
                0.86,
                f"{sender_role}:{requested}",
                "Worker/reviewer messages are data or critique only; they cannot command tool execution.",
            ))
        return self._decision(findings, action="block" if findings else "allow", allowed=not findings)

    def sanitize_agent_output(self, message: Dict[str, Any] | AgentMessage) -> SecurityDecision:
        msg = message if isinstance(message, AgentMessage) else AgentMessage.from_dict(message)
        findings: List[SecurityFinding] = []
        lines = []
        for line in (msg.content or "").splitlines() or [msg.content or ""]:
            if any(pattern.search(line) for pattern in self.INSTRUCTION_PATTERNS):
                findings.append(self._finding(
                    "cross_agent_prompt_injection",
                    "high",
                    0.84,
                    line,
                    "Treat instructions embedded in worker/tool outputs as untrusted data and strip them.",
                ))
                continue
            lines.append(line)
        sanitized = "\n".join(lines).strip() if findings else msg.content
        if findings and not sanitized:
            sanitized = "[InterAgentGuard removed unsafe inter-agent instructions.]"
        return SecurityDecision(
            allowed=True,
            action="sanitize" if findings else "allow",
            risk_score=max((finding.confidence for finding in findings), default=0.0),
            findings=findings,
            sanitized_text=sanitized,
            metadata={"message": msg.to_dict()},
        )

    def _requested_action(self, msg: AgentMessage) -> str:
        explicit = msg.source_trace.get("requested_action") or msg.source_trace.get("tool_name")
        if explicit:
            return str(explicit)
        return "return_result" if msg.sender_role == "worker" else ""

    def _decision(self, findings: List[SecurityFinding], action: str, allowed: bool) -> SecurityDecision:
        return SecurityDecision(
            allowed=allowed,
            action=action,
            risk_score=max((finding.confidence for finding in findings), default=0.0),
            findings=findings,
        )

    def _finding(self, risk_type: str, severity: str, confidence: float, evidence: Any, recommendation: str) -> SecurityFinding:
        return SecurityFinding(
            risk_type=risk_type,
            severity=severity,
            confidence=confidence,
            evidence=str(evidence)[: self.config.max_evidence_chars],
            recommendation=recommendation,
            metadata={"owasp_id": "ASI07"},
        )
