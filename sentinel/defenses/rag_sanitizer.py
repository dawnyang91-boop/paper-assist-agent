from __future__ import annotations

import re
import time
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
            payload = dict(getattr(getattr(ranked, "chunk", None), "payload", None) or getattr(ranked, "payload", {}) or {})
            source_trust_score = self.source_trust_score(payload)
            retrieval_penalty = self.retrieval_penalty(decision.risk_score, source_trust_score)
            if decision.findings:
                decisions.append({
                    "rank": getattr(ranked, "rank", None),
                    "action": decision.action,
                    "risk_score": decision.risk_score,
                    "source_trust_score": source_trust_score,
                    "retrieval_penalty": retrieval_penalty,
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
            try:
                ranked.chunk.payload = dict(ranked.chunk.payload or {})
                ranked.chunk.payload["source_trust_score"] = source_trust_score
                ranked.chunk.payload["sentinel_risk_score"] = decision.risk_score
                ranked.chunk.payload["retrieval_penalty"] = retrieval_penalty
                ranked.rerank_score = max(0.0, float(getattr(ranked, "rerank_score", 0.0)) * (1.0 - retrieval_penalty))
            except Exception:
                pass
            sanitized_chunks.append(ranked)
        sanitized_chunks.sort(key=lambda item: getattr(item, "rerank_score", 0.0), reverse=True)
        for index, ranked in enumerate(sanitized_chunks, start=1):
            try:
                ranked.rank = index
            except Exception:
                pass
        contradiction_warnings = self.contradiction_check(sanitized_chunks)
        if contradiction_warnings:
            decisions.append({
                "action": "warn",
                "risk_score": 0.55,
                "risk_type": "contradiction_check",
                "warnings": contradiction_warnings,
            })
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

    def source_trust_score(self, payload: Dict[str, Any]) -> float:
        explicit = payload.get("source_trust_score") or payload.get("trust_score")
        if explicit is not None:
            return self._clamp_float(explicit, default=0.5)

        score = 0.45
        source_file = str(payload.get("source_file") or payload.get("file_path") or payload.get("source") or "")
        if source_file:
            score += 0.14
        if payload.get("content_hash"):
            score += 0.08
        if payload.get("upload_user") or payload.get("user_id") or payload.get("owner_id"):
            score += 0.1
        if payload.get("author"):
            score += 0.06
        document_type = str(payload.get("document_type") or payload.get("mime_type") or payload.get("file_type") or "").lower()
        if any(marker in document_type for marker in ("pdf", "markdown", "md", "docx", "text", "local")):
            score += 0.08
        if source_file.lower().startswith(("http://", "https://")):
            score -= 0.05

        timestamp = payload.get("timestamp") or payload.get("created_at") or payload.get("updated_at") or payload.get("upload_time")
        recency_bonus = self._recency_bonus(timestamp)
        score += recency_bonus
        if payload.get("sentinel_sanitized"):
            score -= 0.12
        return max(0.1, min(1.0, score))

    def retrieval_penalty(self, risk_score: float, source_trust_score: float) -> float:
        risk = self._clamp_float(risk_score, default=0.0)
        trust = self._clamp_float(source_trust_score, default=0.5)
        penalty = risk * 0.45 + max(0.0, 0.5 - trust) * 0.25
        return max(0.0, min(0.75, penalty))

    def contradiction_check(self, ranked_chunks: Iterable[Any]) -> List[str]:
        chunks = list(ranked_chunks)
        warnings: List[str] = []
        opposing_pairs = [
            ("支持", "不支持"),
            ("需要", "不需要"),
            ("可以", "不可以"),
            ("应该", "不应该"),
            ("supports", "does not support"),
            ("should", "should not"),
        ]
        texts = [(getattr(chunk, "rank", index + 1), (getattr(chunk, "content", "") or "").lower()) for index, chunk in enumerate(chunks)]
        for left_index, left_text in texts:
            if not left_text:
                continue
            for right_index, right_text in texts:
                if right_index <= left_index or not right_text:
                    continue
                for positive, negative in opposing_pairs:
                    if self._has_positive_marker(left_text, positive, negative) and self._has_negative_marker(right_text, negative):
                        warnings.append(f"chunk {left_index} 与 chunk {right_index} 可能存在结论冲突：{positive}/{negative}")
                    elif self._has_negative_marker(left_text, negative) and self._has_positive_marker(right_text, positive, negative):
                        warnings.append(f"chunk {left_index} 与 chunk {right_index} 可能存在结论冲突：{negative}/{positive}")
                if len(warnings) >= 5:
                    return warnings
        return warnings

    def _has_positive_marker(self, text: str, positive: str, negative: str) -> bool:
        return positive in text and negative not in text

    def _has_negative_marker(self, text: str, negative: str) -> bool:
        return negative in text

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

    def _recency_bonus(self, timestamp: Any) -> float:
        try:
            value = float(timestamp)
        except (TypeError, ValueError):
            return 0.0
        # Accept both seconds and milliseconds.
        if value > 10_000_000_000:
            value = value / 1000.0
        age_days = max(0.0, (time.time() - value) / 86400.0)
        if age_days <= 30:
            return 0.08
        if age_days <= 180:
            return 0.04
        if age_days > 3650:
            return -0.04
        return 0.0

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, number))
