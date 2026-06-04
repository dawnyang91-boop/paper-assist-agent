from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from sentinel.config import SentinelConfig
from sentinel.detectors.rule_detector import RAG_INSTRUCTION_RULES, detect_rules
from sentinel.schemas import SecurityDecision, SecurityFinding, decision_from_findings


class RAGSanitizer:
    def __init__(self, config: SentinelConfig):
        self.config = config

    def sanitize_text(self, text: str) -> SecurityDecision:
        if not self.config.enabled or not self.config.rag_guard_enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, sanitized_text=text)

        findings = detect_rules(text, RAG_INSTRUCTION_RULES, max_evidence_chars=self.config.max_evidence_chars)
        sanitized = text
        if findings and self.config.strip_rag_instructions:
            sanitized = self._strip_instruction_lines(text)
        decision = decision_from_findings(
            findings,
            sanitized_text=sanitized,
            warn_threshold=self.config.warn_threshold,
            block_threshold=1.1,
        )
        decision.allowed = True
        decision.action = "sanitize" if findings and sanitized != text else decision.action
        decision.metadata["removed_chars"] = max(0, len(text or "") - len(sanitized or ""))
        return decision

    def sanitize_ranked_chunks(self, ranked_chunks: Iterable[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
        sanitized_chunks = []
        decisions: List[Dict[str, Any]] = []
        for ranked in ranked_chunks:
            content = getattr(ranked, "content", "") or ""
            decision = self.sanitize_text(content)
            if decision.findings:
                decisions.append({
                    "rank": getattr(ranked, "rank", None),
                    "action": decision.action,
                    "risk_score": decision.risk_score,
                    "findings": [finding.to_dict() for finding in decision.findings],
                    "removed_chars": decision.metadata.get("removed_chars", 0),
                })
            sanitized_text = decision.sanitized_text
            if sanitized_text is not None and sanitized_text != content:
                try:
                    ranked.chunk.content = sanitized_text
                    ranked.chunk.payload = dict(ranked.chunk.payload or {})
                    ranked.chunk.payload["sentinel_sanitized"] = True
                    ranked.chunk.payload["sentinel_findings"] = [finding.to_dict() for finding in decision.findings]
                except Exception:
                    pass
            sanitized_chunks.append(ranked)
        return sanitized_chunks, decisions

    def sanitize_docs(self, docs: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sanitized_docs = []
        decisions = []
        for index, doc in enumerate(docs):
            item = dict(doc)
            text = str(item.get("content") or item.get("page_content") or "")
            decision = self.sanitize_text(text)
            if decision.sanitized_text is not None:
                item["content"] = decision.sanitized_text
                if "page_content" in item:
                    item["page_content"] = decision.sanitized_text
            if decision.findings:
                item["sentinel_sanitized"] = True
                item["sentinel_findings"] = [finding.to_dict() for finding in decision.findings]
                decisions.append({"index": index, **decision.to_dict()})
            sanitized_docs.append(item)
        return sanitized_docs, decisions

    def _strip_instruction_lines(self, text: str) -> str:
        kept = []
        for line in (text or "").splitlines():
            findings = detect_rules(line, RAG_INSTRUCTION_RULES, max_evidence_chars=self.config.max_evidence_chars)
            if findings:
                continue
            # Also strip common markdown callout lines that look like assistant instructions.
            if re.match(r"\s*[-*>#]*\s*(instruction|system|assistant)\s*[:：]", line, flags=re.IGNORECASE):
                continue
            kept.append(line)
        sanitized = "\n".join(kept).strip()
        return sanitized or "[该文档片段包含疑似指令注入内容，已被 Sentinel 净化。]"
