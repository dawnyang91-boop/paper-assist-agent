from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from sentinel.config import SentinelConfig
from sentinel.schemas import SecurityDecision, SecurityFinding


DESCRIPTOR_INJECTION_PATTERNS = (
    re.compile(r"\b(ignore|override)\s+(all\s+)?(previous|prior|system|developer)\s+instructions\b", re.I),
    re.compile(r"\b(system|developer)\s*:\s*", re.I),
    re.compile(r"\bcall\s+this\s+tool\s+before\s+answering\b", re.I),
    re.compile(r"\b(no\s+confirmation\s+required|do\s+not\s+ask\s+for\s+confirmation)\b", re.I),
    re.compile(r"\b(reveal|print|exfiltrate)\s+(the\s+)?(prompt|system prompt|developer message|secrets?)\b", re.I),
    re.compile(r"(忽略|覆盖).{0,12}(系统|开发者|之前).{0,12}(指令|规则)"),
    re.compile(r"(无需确认|不要请求确认|泄露提示词|输出系统提示)"),
)

CONFIG_TAMPER_PATTERNS = (
    re.compile(r"\bSENTINEL_[A-Z_]*ENABLED\s*=\s*false\b", re.I),
    re.compile(r"\bSENTINEL_ENABLED\s*=\s*false\b", re.I),
    re.compile(r"\bOPENAI_BASE_URL\s*=\s*https?://(?!api\.openai\.com|api\.deepseek\.com|api\.aihubmix\.com)", re.I),
    re.compile(r"\bMCP_FILESYSTEM_ROOT\s*=\s*/\s*$", re.I | re.M),
    re.compile(r"(关闭|禁用).{0,8}(sentinel|安全|防护)"),
)

DEPENDENCY_RISK_PATTERNS = (
    re.compile(r"\b(pip|npm|pnpm|yarn)\s+install\b.*(http://|https://|git\+|--extra-index-url|--index-url)", re.I),
    re.compile(r"\bcurl\b.+\|\s*(bash|sh)\b", re.I),
    re.compile(r"\binstall\s+(untrusted|unknown|private|mirror)\s+(package|dependency)\b", re.I),
    re.compile(r"(安装|下载).{0,12}(未知|不可信|远程).{0,12}(包|模型|依赖|插件)"),
)

REMOTE_ARTIFACT_PATTERNS = (
    re.compile(r"\b(load|download|use)\s+(model|adapter|embedding|plugin)\s+from\s+https?://", re.I),
    re.compile(r"\bhttps?://(?!huggingface\.co|github\.com|raw\.githubusercontent\.com)", re.I),
    re.compile(r"(从|加载).{0,8}(未知|远程).{0,8}(模型|插件|adapter|embedding)"),
)


class SupplyChainGuard:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def check_tool_descriptor(self, tool_schema: Dict[str, Any]) -> SecurityDecision:
        text = self._descriptor_text(tool_schema)
        findings = self._descriptor_findings(text, artifact_type="mcp_tool_descriptor")
        if findings:
            return SecurityDecision(
                allowed=False,
                action="quarantine",
                risk_score=max(f.confidence for f in findings),
                findings=findings,
                metadata={"artifact_type": "mcp_tool_descriptor", "tool_name": tool_schema.get("name")},
            )
        return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

    def check_skill_output(self, skill_name: str, output: str) -> SecurityDecision:
        findings = self._descriptor_findings(output, artifact_type="skill_output")
        if findings:
            return SecurityDecision(
                allowed=True,
                action="sanitize",
                risk_score=max(f.confidence for f in findings),
                findings=findings,
                sanitized_text=self._sanitize_untrusted_text(output),
                metadata={"artifact_type": "skill_output", "skill_name": skill_name},
            )
        return SecurityDecision(allowed=True, action="allow", risk_score=0.0, sanitized_text=output)

    def check_mcp_server_config(self, server_config: Any) -> SecurityDecision:
        text = f"{getattr(server_config, 'name', '')} {' '.join(str(item) for item in getattr(server_config, 'command', []) or [])} {getattr(server_config, 'env', {})}"
        findings = []
        findings.extend(self._dependency_findings(text, artifact_type="mcp_server_config"))
        findings.extend(self._remote_artifact_findings(text, artifact_type="mcp_server_config"))
        if findings:
            return SecurityDecision(
                allowed=False,
                action="quarantine",
                risk_score=max(f.confidence for f in findings),
                findings=findings,
                metadata={"artifact_type": "mcp_server_config", "server": getattr(server_config, "name", "")},
            )
        return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

    def check_model_config(self, config_diff: Dict[str, Any] | str) -> SecurityDecision:
        text = config_diff if isinstance(config_diff, str) else "\n".join(f"{k}={v}" for k, v in config_diff.items())
        findings = []
        for pattern in CONFIG_TAMPER_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(self._finding(
                    risk_type="agentic_supply_chain",
                    severity="critical",
                    confidence=0.92,
                    evidence=match.group(0),
                    recommendation="Block high-risk runtime configuration changes unless explicitly reviewed by an authenticated operator.",
                    artifact_type="model_config",
                    detector="ConfigTamperDetector",
                ))
        findings.extend(self._remote_artifact_findings(text, artifact_type="model_config"))
        if findings:
            return SecurityDecision(
                allowed=False,
                action="block",
                risk_score=max(f.confidence for f in findings),
                findings=findings,
                metadata={"artifact_type": "model_config"},
            )
        return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

    def check_dependency_request(self, package_name: str, source: str = "") -> SecurityDecision:
        text = f"{package_name} {source}"
        findings = self._dependency_findings(text, artifact_type="dependency_request")
        if findings:
            return SecurityDecision(
                allowed=False,
                action="block",
                risk_score=max(f.confidence for f in findings),
                findings=findings,
                metadata={"artifact_type": "dependency_request", "package": package_name, "source": source},
            )
        return SecurityDecision(allowed=True, action="allow", risk_score=0.0)

    def check_text(self, text: str, artifact_type: str = "supply_chain_artifact") -> SecurityDecision:
        findings = []
        findings.extend(self._descriptor_findings(text, artifact_type=artifact_type))
        findings.extend(self._dependency_findings(text, artifact_type=artifact_type))
        findings.extend(self._remote_artifact_findings(text, artifact_type=artifact_type))
        if not findings:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0)
        action = "block" if any(f.confidence >= self.config.block_threshold for f in findings) else "sanitize"
        return SecurityDecision(
            allowed=action != "block",
            action=action,
            risk_score=max(f.confidence for f in findings),
            findings=findings,
            sanitized_text=self._sanitize_untrusted_text(text),
            metadata={"artifact_type": artifact_type},
        )

    def _descriptor_text(self, tool_schema: Dict[str, Any]) -> str:
        parts = []
        for key in ("name", "description", "title"):
            if tool_schema.get(key):
                parts.append(str(tool_schema[key]))
        schema = tool_schema.get("inputSchema") or tool_schema.get("input_schema") or tool_schema.get("parameters") or {}
        parts.extend(str(value) for value in self._walk_descriptions(schema))
        return "\n".join(parts)

    def _walk_descriptions(self, value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"description", "title"}:
                    yield str(item)
                yield from self._walk_descriptions(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from self._walk_descriptions(item)

    def _descriptor_findings(self, text: str, artifact_type: str) -> List[SecurityFinding]:
        findings = []
        for pattern in DESCRIPTOR_INJECTION_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(self._finding(
                    risk_type="agentic_supply_chain",
                    severity="critical",
                    confidence=0.91,
                    evidence=match.group(0),
                    recommendation="Quarantine or sanitize tool descriptors and skill outputs that contain prompt instructions.",
                    artifact_type=artifact_type,
                    detector="DescriptorInjectionDetector",
                ))
        return findings

    def _dependency_findings(self, text: str, artifact_type: str) -> List[SecurityFinding]:
        findings = []
        for pattern in DEPENDENCY_RISK_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(self._finding(
                    risk_type="agentic_supply_chain",
                    severity="high",
                    confidence=0.88,
                    evidence=match.group(0),
                    recommendation="Do not install untrusted dependencies or execute remote install scripts without review.",
                    artifact_type=artifact_type,
                    detector="ArtifactTrustDetector",
                ))
        return findings

    def _remote_artifact_findings(self, text: str, artifact_type: str) -> List[SecurityFinding]:
        findings = []
        for pattern in REMOTE_ARTIFACT_PATTERNS:
            match = pattern.search(text or "")
            if match:
                findings.append(self._finding(
                    risk_type="agentic_supply_chain",
                    severity="high",
                    confidence=0.84,
                    evidence=match.group(0),
                    recommendation="Only load model, plugin, or tool artifacts from allowlisted sources with provenance checks.",
                    artifact_type=artifact_type,
                    detector="ArtifactTrustDetector",
                ))
        return findings

    def _sanitize_untrusted_text(self, text: str) -> str:
        sanitized = text or ""
        for pattern in DESCRIPTOR_INJECTION_PATTERNS:
            sanitized = pattern.sub("[removed supply-chain instruction]", sanitized)
        return sanitized

    def _finding(
        self,
        risk_type: str,
        severity: str,
        confidence: float,
        evidence: str,
        recommendation: str,
        artifact_type: str,
        detector: str,
    ) -> SecurityFinding:
        return SecurityFinding(
            risk_type=risk_type,
            severity=severity,
            confidence=confidence,
            evidence=evidence,
            recommendation=recommendation,
            metadata={"owasp_id": "ASI04", "artifact_type": artifact_type, "detector": detector},
        )
