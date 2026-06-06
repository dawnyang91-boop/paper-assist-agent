from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding, decision_from_findings


BLOCK_COMMAND_PATTERNS = (
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\b(curl|wget)\b.+\|\s*(bash|sh)\b", re.I),
    re.compile(r"\bsudo\b", re.I),
    re.compile(r"\bchmod\s+777\b", re.I),
    re.compile(r"\b(nc|netcat)\b.*(-e|/bin/sh|/bin/bash)", re.I),
    re.compile(r"\bdocker\s+run\b.*--privileged\b", re.I),
    re.compile(r"\bkubectl\s+delete\b", re.I),
    re.compile(r"\bdrop\s+table\b", re.I),
    re.compile(r"\bdelete\s+from\b(?!.*\bwhere\b)", re.I | re.S),
)

EXECUTION_INTENT_PATTERNS = (
    re.compile(r"\b(run|execute|launch)\s+(this\s+)?(code|script|command)\b", re.I),
    re.compile(r"\bautomatically\s+(deploy|execute|run)\b", re.I),
    re.compile(r"\bno\s+confirmation\s+(needed|required)\b", re.I),
    re.compile(r"(执行以下命令|运行这段代码|自动部署|无需确认执行)"),
)

CICD_PATH_PATTERNS = (
    re.compile(r"(^|/)\.github/workflows/.*\.(ya?ml)$", re.I),
    re.compile(r"(^|/)Dockerfile$", re.I),
    re.compile(r"(^|/)docker-compose\.ya?ml$", re.I),
    re.compile(r"(^|/)deploy\.sh$", re.I),
    re.compile(r"(^|/)k8s/.*\.ya?ml$", re.I),
    re.compile(r"(^|/)helm/", re.I),
    re.compile(r"(^|/)package\.json$", re.I),
    re.compile(r"(^|/)pyproject\.toml$", re.I),
)


class CodeExecutionGuard:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def check_command(self, cmd: str, user_context: Dict[str, Any] | None = None) -> SecurityDecision:
        findings = self._command_findings(cmd)
        action = "block" if findings and max(f.confidence for f in findings) >= self.config.block_threshold else "allow"
        return SecurityDecision(
            allowed=action != "block",
            action=action,
            risk_score=max((finding.confidence for finding in findings), default=0.0),
            findings=findings,
        )

    def check_generated_code(self, code: str, language: str = "") -> SecurityDecision:
        findings = self._command_findings(code)
        findings.extend(self._intent_findings(code))
        return decision_from_findings(findings, warn_threshold=self.config.warn_threshold, block_threshold=self.config.block_threshold)

    def check_file_write_execution_risk(self, path: str, content: str = "") -> SecurityDecision:
        findings: List[SecurityFinding] = []
        if self._is_cicd_path(path):
            findings.append(SecurityFinding(
                risk_type="unexpected_code_execution",
                severity="high",
                confidence=0.86,
                evidence=path,
                recommendation="CI/CD and deployment file mutations require explicit human review.",
                metadata={"owasp_id": "ASI05", "detector": "CICDMutationDetector"},
            ))
        findings.extend(self._command_findings(content))
        if findings and any(f.confidence >= self.config.block_threshold for f in findings if "curl" in f.evidence.lower() or "rm -rf" in f.evidence.lower()):
            action = "block"
        elif findings:
            action = "require_confirmation"
        else:
            action = "allow"
        return SecurityDecision(
            allowed=action == "allow",
            action=action,
            risk_score=max((finding.confidence for finding in findings), default=0.0),
            findings=findings,
        )

    def check_sql_query(self, sql: str) -> SecurityDecision:
        return self.check_command(sql)

    def check_cicd_change(self, path: str, content: str) -> SecurityDecision:
        return self.check_file_write_execution_risk(path, content)

    def check_tool_args(self, tool_name: str, args: Dict[str, Any]) -> SecurityDecision:
        name = (tool_name or "").lower()
        if name in {"run_command", "shell", "exec", "terminal", "command"}:
            return self.check_command(str(args.get("cmd") or args.get("input") or ""))
        if name in {"file_write", "github_write", "github_create_file", "github_update_file"}:
            return self.check_file_write_execution_risk(str(args.get("path") or ""), str(args.get("content") or ""))
        if name in {"sqlite", "sql_query", "database_query"}:
            return self.check_sql_query(str(args.get("query") or args.get("sql") or ""))
        findings = self._intent_findings(" ".join(str(value) for value in self._flatten_values(args)))
        return decision_from_findings(findings, warn_threshold=self.config.warn_threshold, block_threshold=self.config.block_threshold)

    def _command_findings(self, text: str) -> List[SecurityFinding]:
        findings = []
        for pattern in BLOCK_COMMAND_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(SecurityFinding(
                    risk_type="unexpected_code_execution",
                    severity="critical",
                    confidence=0.96,
                    evidence=match.group(0),
                    recommendation="Block destructive or remotely-sourced command execution by default.",
                    metadata={"owasp_id": "ASI05", "detector": "CommandRiskDetector"},
                ))
        return findings

    def _intent_findings(self, text: str) -> List[SecurityFinding]:
        findings = []
        for pattern in EXECUTION_INTENT_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(SecurityFinding(
                    risk_type="unexpected_code_execution",
                    severity="medium",
                    confidence=0.66,
                    evidence=match.group(0),
                    recommendation="Require explicit confirmation before generated code or scripts are executed.",
                    metadata={"owasp_id": "ASI05", "detector": "CodeExecutionIntentDetector"},
                ))
        return findings

    def _is_cicd_path(self, path: str) -> bool:
        return any(pattern.search(path or "") for pattern in CICD_PATH_PATTERNS)

    def _flatten_values(self, value: Any) -> Iterable[Any]:
        if isinstance(value, dict):
            for item in value.values():
                yield from self._flatten_values(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                yield from self._flatten_values(item)
        else:
            yield value
