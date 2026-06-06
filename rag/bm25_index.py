import math
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    jieba = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]{1,}|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]")


@dataclass
class BM25Document:
    doc_id: Any
    content: str
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25SearchResult:
    document: BM25Document
    score: float


class SimpleBM25Okapi:
    """Small fallback BM25 implementation used when rank-bm25 is unavailable."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(doc) for doc in corpus]
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self._initialize()

    def _initialize(self) -> None:
        document_frequency: Dict[str, int] = {}
        for document in self.corpus:
            frequencies: Dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        total = max(1, len(self.corpus))
        for token, frequency in document_frequency.items():
            self.idf[token] = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        scores = []
        for index, frequencies in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[index] if index < len(self.doc_len) else 0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores


class BM25ChunkIndex:
    """Persistent BM25 index over RAG chunks stored in Qdrant."""

    def __init__(
        self,
        documents: Optional[List[BM25Document]] = None,
        tokenized_corpus: Optional[List[List[str]]] = None,
        collection_name: str = "",
        embed_model_name: str = "",
        built_at: Optional[float] = None,
    ):
        self.documents = documents or []
        self.tokenized_corpus = tokenized_corpus or []
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self.built_at = built_at or time.time()
        self.bm25 = self._build_bm25(self.tokenized_corpus)

    @classmethod
    def build(
        cls,
        documents: Iterable[BM25Document],
        collection_name: str = "",
        embed_model_name: str = "",
    ) -> "BM25ChunkIndex":
        docs = list(documents)
        corpus = [cls.tokenize(cls._index_text(document.payload, document.content)) for document in docs]
        return cls(
            documents=docs,
            tokenized_corpus=corpus,
            collection_name=collection_name,
            embed_model_name=embed_model_name,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Optional["BM25ChunkIndex"]:
        index_path = Path(path)
        if not index_path.exists():
            return None
        with index_path.open("rb") as handle:
            data = pickle.load(handle)
        documents = [
            item if isinstance(item, BM25Document) else BM25Document(**item)
            for item in data.get("documents", [])
        ]
        return cls(
            documents=documents,
            tokenized_corpus=data.get("tokenized_corpus", []),
            collection_name=data.get("collection_name", ""),
            embed_model_name=data.get("embed_model_name", ""),
            built_at=data.get("built_at"),
        )

    @classmethod
    def rebuild_from_qdrant(
        cls,
        qdrant_client: Any,
        collection_name: str,
        embed_model_name: str = "",
        batch_size: int = 256,
    ) -> "BM25ChunkIndex":
        documents: List[BM25Document] = []
        offset = None
        while True:
            response = qdrant_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(response, tuple):
                points, next_offset = response
            else:
                points = getattr(response, "points", [])
                next_offset = getattr(response, "next_page_offset", None)
            for point in points or []:
                payload = dict(getattr(point, "payload", {}) or {})
                content = payload.get("page_content") or payload.get("content") or ""
                if not content:
                    continue
                documents.append(BM25Document(
                    doc_id=getattr(point, "id", None),
                    content=str(content),
                    source_file=payload.get("source_file") or payload.get("file_name"),
                    chunk_index=payload.get("chunk_index"),
                    payload=payload,
                ))
            if next_offset is None:
                break
            offset = next_offset
        return cls.build(documents, collection_name=collection_name, embed_model_name=embed_model_name)

    def save(self, path: str | os.PathLike[str]) -> None:
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": [document.__dict__ for document in self.documents],
            "tokenized_corpus": self.tokenized_corpus,
            "collection_name": self.collection_name,
            "embed_model_name": self.embed_model_name,
            "built_at": self.built_at,
        }
        with index_path.open("wb") as handle:
            pickle.dump(data, handle)

    def search(self, query: str, top_k: int = 20) -> List[BM25SearchResult]:
        if not self.documents or not self.bm25:
            return []
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        scores = list(self.bm25.get_scores(query_tokens))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            BM25SearchResult(document=self.documents[index], score=float(score))
            for index, score in ranked[:top_k]
            if score > 0
        ]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        raw_tokens = []
        if jieba is not None:
            raw_tokens.extend(jieba.cut(text, cut_all=False))
        raw_tokens.extend(TOKEN_RE.findall(text))
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        raw_tokens.extend(
            "".join(cjk_chars[index:index + 2])
            for index in range(max(0, len(cjk_chars) - 1))
        )
        tokens = []
        for token in raw_tokens:
            normalized = token.strip().lower()
            if len(normalized) < 2:
                continue
            tokens.append(normalized)
        return tokens

    @staticmethod
    def _index_text(payload: Dict[str, Any], content: str) -> str:
        parts = [content or ""]
        for key in ("source_file", "file_name", "paper_title"):
            value = payload.get(key)
            if value:
                parts.append(str(value))
        for key in ("heading_paths", "acronyms", "model_names", "keywords"):
            value = payload.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value if item)
            elif value:
                parts.append(str(value))
        return "\n".join(parts)

    @staticmethod
    def _build_bm25(corpus: List[List[str]]):
        if not corpus:
            return None
        if BM25Okapi is not None:
            return BM25Okapi(corpus)
        return SimpleBM25Okapi(corpus)
