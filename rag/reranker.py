import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

from config import AppConfig, get_config
from rag.query_entity import QueryEntityExtractor
from rag.rag_query import RetrievedChunk


MODE_WEIGHTS = {
    "basic": 1.0,
    "mqe": 0.95,
    "hyde": 0.9,
}


GENERIC_TOPIC_TERMS = {
    "什么", "是什", "是什么", "怎么", "怎样", "如何", "为何", "为什么", "请问",
    "详细", "流程", "内容", "系统", "文档", "记忆", "检索", "回答", "问题",
    "相关", "根据", "提供", "当前", "用户", "上下", "上下文", "知识", "私域",
    "可以", "需要", "进行", "一个", "这个", "那个", "以及", "或者", "如果",
    "说明", "解释", "介绍", "总结", "分析", "比较", "区别", "作用", "方式",
    "方法", "方案", "建议", "推荐", "使用", "实现", "构建", "设计", "核心",
    "贡献", "主要", "包括", "例如", "具有", "中的", "关于", "无法", "基于",
}


@dataclass
class RankedChunk:
    chunk: RetrievedChunk
    rerank_score: float
    retrieval_score: float
    lexical_score: float
    diversity_score: float
    mode_score: float
    bm25_score: float = 0.0
    entity_score: float = 0.0
    hybrid_bonus: float = 0.0
    entity_missing_penalty: float = 0.0
    rank: int = 0

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def payload(self):
        return self.chunk.payload


def _normalize_score(score: float) -> float:
    if score is None:
        return 0.0
    if 0.0 <= score <= 1.0:
        return float(score)
    return 1.0 / (1.0 + math.exp(-float(score)))


def _extract_terms(text: str) -> Set[str]:
    text = (text or "").lower()
    ascii_terms = set(re.findall(r"[a-zA-Z0-9_]{2,}", text))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = {cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)}
    return ascii_terms | cjk_bigrams


def _extract_topic_terms(text: str) -> Set[str]:
    text = (text or "").lower()
    ascii_terms = {
        term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_.+-]{1,}", text)
        if term not in GENERIC_TOPIC_TERMS
    }
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_ngrams = set()
    for size in (2, 3, 4):
        cjk_ngrams.update(
            "".join(cjk_chars[index:index + size])
            for index in range(max(0, len(cjk_chars) - size + 1))
        )
    return {
        term for term in ascii_terms | cjk_ngrams
        if term and term not in GENERIC_TOPIC_TERMS and not _is_generic_topic_term(term)
    }


def _is_generic_topic_term(term: str) -> bool:
    if term in GENERIC_TOPIC_TERMS:
        return True
    return any(generic in term for generic in GENERIC_TOPIC_TERMS if len(generic) >= 2)


def lexical_overlap(question: str, content: str) -> float:
    query_terms = _extract_terms(question)
    if not query_terms:
        return 0.0
    content_terms = _extract_terms(content)
    if not content_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)


def topic_overlap(question: str, content: str) -> float:
    query_terms = _extract_topic_terms(question)
    if not query_terms:
        return 0.0
    content_terms = _extract_topic_terms(content)
    if not content_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)


class CandidateReranker:
    """Rule-based reranker for deduped RAG candidates."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.entity_extractor = QueryEntityExtractor()

    def rerank(
        self,
        question: str,
        candidates: Iterable[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RankedChunk]:
        top_k = top_k or self.config.rag_rerank_top_k
        entity_profile = self.entity_extractor.extract(question)
        ranked = []
        for chunk in candidates:
            retrieval_score = _normalize_score(chunk.score)
            lexical_score = lexical_overlap(question, chunk.content)
            diversity_score = min(len(chunk.source_hits), 4) / 4.0
            mode_score = self._mode_score(chunk)
            bm25_score = self._bm25_score(chunk)
            entity_score = self._entity_score(chunk, entity_profile.entities)
            hybrid_bonus = self.config.rag_rerank_hybrid_bonus if self._is_hybrid(chunk) else 0.0
            entity_missing_penalty = 0.0
            if (
                self.config.rag_entity_aware_enabled
                and self.config.rag_entity_required_for_strong_query
                and entity_profile.has_strong_entity
                and entity_score <= 0.0
            ):
                entity_missing_penalty = self.config.rag_rerank_entity_missing_penalty
            final_score = (
                retrieval_score * self.config.rag_rerank_vector_weight
                + lexical_score * self.config.rag_rerank_lexical_weight
                + diversity_score * self.config.rag_rerank_diversity_weight
                + mode_score * self.config.rag_rerank_mode_weight
                + bm25_score * self.config.rag_rerank_bm25_weight
                + entity_score * self.config.rag_rerank_entity_weight
                + hybrid_bonus
                - entity_missing_penalty
            )
            ranked.append(RankedChunk(
                chunk=chunk,
                rerank_score=final_score,
                retrieval_score=retrieval_score,
                lexical_score=lexical_score,
                diversity_score=diversity_score,
                mode_score=mode_score,
                bm25_score=bm25_score,
                entity_score=entity_score,
                hybrid_bonus=hybrid_bonus,
                entity_missing_penalty=entity_missing_penalty,
            ))

        ranked.sort(key=lambda item: item.rerank_score, reverse=True)
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked[:top_k]

    def _mode_score(self, chunk: RetrievedChunk) -> float:
        modes = [hit.get("query_mode") for hit in chunk.source_hits] or [chunk.query_mode]
        modes = [str(mode).replace("vector_", "") for mode in modes if mode]
        scores = [MODE_WEIGHTS.get(mode, 0.85) for mode in modes if mode]
        return max(scores) if scores else 0.85

    def _bm25_score(self, chunk: RetrievedChunk) -> float:
        payload_score = float((chunk.payload or {}).get("_bm25_score", 0.0) or 0.0)
        hit_scores = [
            float(hit.get("normalized_score", 0.0) or 0.0)
            for hit in chunk.source_hits
            if hit.get("query_mode") == "bm25"
        ]
        return max([payload_score, *hit_scores], default=0.0)

    def _entity_score(self, chunk: RetrievedChunk, entities: List[str]) -> float:
        if not entities or not self.config.rag_entity_aware_enabled:
            return 0.0
        payload = chunk.payload or {}
        haystack = "\n".join([
            chunk.content or "",
            *(str(value) for value in payload.values() if value is not None),
        ]).lower()
        covered = sum(1 for entity in entities if entity.lower() in haystack)
        return covered / len(entities) if entities else 0.0

    def _is_hybrid(self, chunk: RetrievedChunk) -> bool:
        if (chunk.payload or {}).get("_hybrid_hit"):
            return True
        modes = {hit.get("query_mode") for hit in chunk.source_hits}
        has_bm25 = "bm25" in modes
        has_vector = any(str(mode or "").startswith("vector_") for mode in modes)
        return has_bm25 and has_vector
