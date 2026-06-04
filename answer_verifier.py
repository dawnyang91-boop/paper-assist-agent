import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


DOCUMENT_CITATION_RE = re.compile(r"\[D(\d+)\]")


@dataclass
class VerificationResult:
    passed: bool = True
    status: str = "ok"
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)
    citation_coverage: float = 0.0
    likely_answered: bool = True
    need_repair: bool = False
    need_more_info: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "missing_citations": list(self.missing_citations),
            "citation_coverage": self.citation_coverage,
            "likely_answered": self.likely_answered,
            "need_repair": self.need_repair,
            "need_more_info": self.need_more_info,
            "reason": self.reason,
        }


class AnswerVerifier:
    """Minimal answer checker for citation integrity and obvious empty answers."""

    def verify(self, answer: str, built_context: Any, query_plan: Any = None) -> VerificationResult:
        result = VerificationResult()
        answer = (answer or "").strip()
        documents = list(getattr(built_context, "documents", []) or [])
        question = getattr(built_context, "question", "")
        valid_doc_ids = {getattr(document, "doc_id", "") for document in documents}
        cited_doc_ids = [f"D{match}" for match in DOCUMENT_CITATION_RE.findall(answer)]
        unique_cited_doc_ids = sorted(set(cited_doc_ids))
        if documents:
            valid_cited_count = len([doc_id for doc_id in unique_cited_doc_ids if doc_id in valid_doc_ids])
            result.citation_coverage = valid_cited_count / len(documents)

        if not answer:
            result.passed = False
            result.status = "answer_empty"
            result.issues.append("答案为空。")
            result.need_repair = True
            result.reason = "LLM 没有生成可用答案。"
            return result

        for doc_id in cited_doc_ids:
            if doc_id not in valid_doc_ids:
                result.missing_citations.append(doc_id)

        if result.missing_citations:
            result.passed = False
            result.status = "citation_error"
            result.issues.append(f"答案引用了不存在的文档编号：{', '.join(result.missing_citations)}。")
            result.need_repair = True
            result.reason = "引用编号不在本轮上下文文档中。"
            return result

        if documents and not cited_doc_ids:
            result.status = "warning"
            result.warnings.append("已有检索文档，但答案没有显式引用 [D?]。")

        if documents and cited_doc_ids and result.citation_coverage < 0.2:
            result.warnings.append("答案引用覆盖率较低，可能没有充分利用检索文档。")

        consistency_warnings = self._citation_consistency_warnings(answer, documents)
        if consistency_warnings:
            result.warnings.extend(consistency_warnings)

        if self._signals_insufficient_context(answer):
            result.need_more_info = bool(getattr(query_plan, "need_rag", False))
            result.warnings.append("答案表达了上下文不足，可能需要补充检索或工具调用。")

        if question and not self._looks_relevant(question, answer):
            result.likely_answered = False
            result.warnings.append("答案与问题的词面重合较低，可能答非所问。")
            if not documents and not getattr(built_context, "memories", []):
                result.need_more_info = True

        if not documents and self._claims_document_grounding(answer):
            result.passed = False
            result.status = "hallucination_risk"
            result.issues.append("没有检索文档，但答案声称依据文档。")
            result.need_repair = True
            result.reason = "缺少可支撑的文档来源。"

        if not documents and not getattr(built_context, "memories", []):
            result.need_more_info = bool(getattr(query_plan, "need_rag", False))

        return result

    def _claims_document_grounding(self, answer: str) -> bool:
        markers = ["根据文档", "根据资料", "从文档", "检索文档显示", "资料显示"]
        return any(marker in answer for marker in markers)

    def _citation_consistency_warnings(self, answer: str, documents: List[Any]) -> List[str]:
        by_id = {getattr(document, "doc_id", ""): document for document in documents}
        warnings: List[str] = []
        cited_ids = sorted(set(f"D{match}" for match in DOCUMENT_CITATION_RE.findall(answer or "")))
        if not cited_ids:
            return warnings

        for doc_id in cited_ids:
            document = by_id.get(doc_id)
            if document is None:
                continue
            snippets = self._sentences_with_citation(answer, doc_id)
            if not snippets:
                continue
            doc_terms = self._terms(getattr(document, "content", ""))
            if not doc_terms:
                continue
            best_overlap = max(
                (len(self._terms(snippet) & doc_terms) / max(1, len(self._terms(snippet))) for snippet in snippets),
                default=0.0,
            )
            if best_overlap < 0.08:
                warnings.append(f"引用 {doc_id} 附近表述与对应文档片段词面支撑较弱，请检查引用是否准确。")
        return warnings

    def _sentences_with_citation(self, answer: str, doc_id: str) -> List[str]:
        normalized = (answer or "").replace("\n", " ")
        parts = re.split(r"(?<=[。！？!?；;])\s+|\n+", normalized)
        if len(parts) <= 1:
            parts = re.split(r"(?<=[。！？!?；;])", normalized)
        return [part.strip() for part in parts if f"[{doc_id}]" in part or f"[{doc_id[0]}{doc_id[1:]}]" in part]

    def _signals_insufficient_context(self, answer: str) -> bool:
        markers = ["上下文不足", "资料不足", "无法确定", "没有检索到", "无法基于", "未提供足够"]
        return any(marker in answer for marker in markers)

    def _looks_relevant(self, question: str, answer: str) -> bool:
        question_terms = self._terms(question)
        if not question_terms:
            return True
        answer_terms = self._terms(answer)
        if not answer_terms:
            return False
        overlap = len(question_terms & answer_terms) / max(1, len(question_terms))
        return overlap >= 0.15

    def _terms(self, text: str) -> set[str]:
        ascii_terms = set(re.findall(r"[A-Za-z0-9_]{2,}", text.lower()))
        chinese = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_bigrams = {"".join(chinese[index:index + 2]) for index in range(max(0, len(chinese) - 1))}
        return ascii_terms | chinese_bigrams
