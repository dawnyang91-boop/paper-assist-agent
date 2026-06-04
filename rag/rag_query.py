import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config import AppConfig, get_config
from rag.embedding_service import EmbeddingService
from storage.redis_cache import RedisJsonCache
from storage.redis_runtime import RedisUnavailable


QUERY_MODE_BASIC = "basic"
QUERY_MODE_MQE = "mqe"
QUERY_MODE_HYDE = "hyde"


@dataclass
class QueryVariant:
    text: str
    mode: str
    origin: str
    vector: Optional[List[float]] = None


@dataclass
class RetrievedChunk:
    id: Any
    score: float
    payload: Dict[str, Any]
    query_mode: str
    query_text: str
    content: str = ""
    source_hits: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.content:
            self.content = self.payload.get("page_content") or self.payload.get("content", "")
        if not self.source_hits:
            self.source_hits.append({
                "id": self.id,
                "score": self.score,
                "query_mode": self.query_mode,
                "query_text": self.query_text,
            })


def _parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_content(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


class QueryGenerator:
    """Generate Basic, MQE, and HyDE query variants before vector search."""

    def __init__(self, config: Optional[AppConfig] = None, llm_client: Any = None):
        self.config = config or get_config()
        self.llm_client = llm_client
        try:
            self.cache = RedisJsonCache(config=self.config)
        except RedisUnavailable:
            self.cache = None

    def _get_llm_client(self):
        if self.llm_client is not None:
            return self.llm_client
        if not self.config.openai_api_key:
            return None

        from openai import OpenAI

        self.llm_client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )
        return self.llm_client

    def build_basic(self, question: str) -> List[QueryVariant]:
        return [QueryVariant(text=question.strip(), mode=QUERY_MODE_BASIC, origin=question.strip())]

    def build_mqe(self, question: str, count: int = 3) -> List[QueryVariant]:
        cache_key = self.cache.make_key("mqe", self.config.model_name, question, count) if self.cache and self.cache.enabled else None
        cached = self.cache.get(cache_key) if self.cache else None
        expanded = [str(item) for item in cached] if isinstance(cached, list) else []
        if not expanded:
            expanded = self._llm_mqe(question, count=count)
            if expanded:
                self.cache.set(cache_key, expanded) if self.cache else None
        if not expanded:
            expanded = self._fallback_mqe(question, count=count)

        variants = []
        seen = set()
        for item in [question, *expanded]:
            normalized = _normalize_content(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            variants.append(QueryVariant(text=item.strip(), mode=QUERY_MODE_MQE, origin=question.strip()))
        return variants

    def build_hyde(self, question: str) -> List[QueryVariant]:
        cache_key = self.cache.make_key("hyde", self.config.model_name, question) if self.cache and self.cache.enabled else None
        cached = self.cache.get(cache_key) if self.cache else None
        hypothetical = str(cached) if isinstance(cached, str) else ""
        if not hypothetical:
            hypothetical = self._llm_hyde(question)
            if hypothetical:
                self.cache.set(cache_key, hypothetical) if self.cache else None
        if not hypothetical:
            hypothetical = self._fallback_hyde(question)
        return [QueryVariant(text=hypothetical.strip(), mode=QUERY_MODE_HYDE, origin=question.strip())]

    def build_queries(
        self,
        question: str,
        modes: Sequence[str] = (QUERY_MODE_BASIC, QUERY_MODE_MQE, QUERY_MODE_HYDE),
        mqe_count: int = 3,
    ) -> List[QueryVariant]:
        variants = []
        for mode in modes:
            if mode == QUERY_MODE_BASIC:
                variants.extend(self.build_basic(question))
            elif mode == QUERY_MODE_MQE:
                variants.extend(self.build_mqe(question, count=mqe_count))
            elif mode == QUERY_MODE_HYDE:
                variants.extend(self.build_hyde(question))
            else:
                raise ValueError(f"未知查询模式: {mode}")
        return self._dedupe_query_variants(variants)

    def _llm_mqe(self, question: str, count: int) -> List[str]:
        client = self._get_llm_client()
        if client is None:
            return []

        prompt = f"""Rewrite the same retrieval question into {count} English query variants for RAG vector search.
Requirements:
- Preserve the original meaning and do not invent extra constraints.
- Cover keywords, synonyms, background concepts, and likely user phrasings.
- Return only a JSON array string.

Original question: {question}"""
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                **self.config.chat_completion_kwargs(),
            )
            data = _parse_json_object(response.choices[0].message.content)
            if isinstance(data, dict):
                data = data.get("queries", [])
            return [str(item).strip() for item in data if str(item).strip()][:count]
        except Exception:
            return []

    def _llm_hyde(self, question: str) -> str:
        client = self._get_llm_client()
        if client is None:
            return ""

        prompt = f"""Generate a hypothetical answer paragraph that could appear in real reference material, for HyDE retrieval.
Requirements:
- It does not need to be fully factually correct; it is a semantic carrier for query embedding.
- Write one dense English paragraph containing core terms, related concepts, and likely answer structure.
- Do not output a title, list, or Markdown.

User question: {question}"""
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Generate hypothetical answer paragraphs for retrieval."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                **self.config.chat_completion_kwargs(),
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _fallback_mqe(self, question: str, count: int) -> List[str]:
        templates = [
            "Explain: {question}",
            "What are the core concepts, methods, and conclusions of {question}?",
            "Key facts, background, and implementation details related to {question}",
        ]
        return [template.format(question=question.strip()) for template in templates[:count]]

    def _fallback_hyde(self, question: str) -> str:
        return (
            f"This question is about {question.strip()}. Relevant material usually explains its core concepts, background, "
            "key methods, implementation workflow, important conclusions, pros and cons, and project-practice details."
        )

    def _dedupe_query_variants(self, variants: Iterable[QueryVariant]) -> List[QueryVariant]:
        deduped = []
        seen = set()
        for variant in variants:
            normalized = _normalize_content(variant.text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(variant)
        return deduped


class RAGQueryEngine:
    """Generate query vectors, search Qdrant, and dedupe retrieved chunks."""

    def __init__(
        self,
        qdrant_client: Any,
        collection_name: Optional[str] = None,
        config: Optional[AppConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        query_generator: Optional[QueryGenerator] = None,
    ):
        self.config = config or get_config()
        self.qdrant = qdrant_client
        self.collection_name = collection_name or self.config.rag_collection_name
        self.embedding_service = embedding_service or EmbeddingService(config=self.config)
        self.query_generator = query_generator or QueryGenerator(config=self.config)

    def generate_query_vectors(
        self,
        question: str,
        modes: Sequence[str] = (QUERY_MODE_BASIC, QUERY_MODE_MQE, QUERY_MODE_HYDE),
        mqe_count: Optional[int] = None,
    ) -> List[QueryVariant]:
        mqe_count = mqe_count or self.config.rag_query_mqe_count
        variants = self.query_generator.build_queries(question, modes=modes, mqe_count=mqe_count)
        vectors = self.embedding_service.embed_texts([variant.text for variant in variants])
        for variant, vector in zip(variants, vectors):
            variant.vector = vector
        return variants

    def search(
        self,
        question: str,
        modes: Sequence[str] = (QUERY_MODE_BASIC, QUERY_MODE_MQE, QUERY_MODE_HYDE),
        top_k: Optional[int] = None,
        per_query_limit: Optional[int] = None,
        mqe_count: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        top_k = top_k or self.config.rag_query_top_k
        per_query_limit = per_query_limit or self.config.rag_query_per_query_limit or max(top_k, 5)
        variants = self.generate_query_vectors(question, modes=modes, mqe_count=mqe_count)

        candidates = []
        for variant in variants:
            if variant.vector is None:
                continue
            hits = self._vector_search(variant.vector, limit=per_query_limit)
            for hit in hits:
                candidates.append(self._to_retrieved_chunk(hit, variant))

        return self.dedupe_results(candidates)[:top_k]

    def _vector_search(self, query_vector: List[float], limit: int):
        if hasattr(self.qdrant, "search"):
            return self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )

        if hasattr(self.qdrant, "query_points"):
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
            )
            return getattr(response, "points", response)

        raise AttributeError(
            "当前 QdrantClient 既没有 search，也没有 query_points，"
            "请检查 qdrant-client 版本或传入兼容的客户端。"
        )

    def dedupe_results(self, chunks: Iterable[RetrievedChunk]) -> List[RetrievedChunk]:
        deduped: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = self._dedupe_key(chunk)
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = chunk
                continue

            existing.source_hits.extend(chunk.source_hits)
            if chunk.score > existing.score:
                chunk.source_hits = existing.source_hits
                deduped[key] = chunk

        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)

    def _to_retrieved_chunk(self, hit: Any, variant: QueryVariant) -> RetrievedChunk:
        payload = dict(getattr(hit, "payload", {}) or {})
        return RetrievedChunk(
            id=getattr(hit, "id", None),
            score=float(getattr(hit, "score", 0.0)),
            payload=payload,
            query_mode=variant.mode,
            query_text=variant.text,
        )

    def _dedupe_key(self, chunk: RetrievedChunk) -> str:
        payload = chunk.payload or {}
        if payload.get("content_hash"):
            return f"hash:{payload['content_hash']}"
        if payload.get("source_file") is not None and payload.get("chunk_index") is not None:
            return f"source:{payload['source_file']}:{payload['chunk_index']}"
        normalized = _normalize_content(chunk.content)
        if normalized:
            return f"content:{normalized}"
        return f"id:{chunk.id}"
