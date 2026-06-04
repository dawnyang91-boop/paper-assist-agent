from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from config import AppConfig, get_config
from reranker import RankedChunk


@dataclass
class ContextDocument:
    doc_id: str
    content: str
    score: float
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    heading_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltContext:
    question: str
    context: str
    documents: List[ContextDocument]
    memories: List[Dict[str, Any]]
    token_estimate: int
    allow_general_knowledge_fallback: bool = False
    evidence_status: Dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 2)


class ContextBuilder:
    """Build top-k document snippets and final prompt context for answer generation."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()

    def build_top_k_documents(
        self,
        ranked_chunks: Iterable[RankedChunk],
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[ContextDocument]:
        if top_k is None:
            top_k = self.config.rag_context_top_k
        if top_k <= 0:
            top_k = None
        max_tokens = max_tokens or self.config.rag_context_max_tokens
        documents: List[ContextDocument] = []
        used_tokens = 0

        ranked_list = list(ranked_chunks)
        if top_k is not None:
            ranked_list = ranked_list[:top_k]

        for ranked in ranked_list:
            payload = ranked.payload or {}
            content = self._trim_document_content(ranked.content.strip())
            if not content:
                continue

            source_file = payload.get("source_file")
            chunk_index = payload.get("chunk_index")
            doc_id = f"D{len(documents) + 1}"
            block = self._format_document_block(
                doc_id=doc_id,
                content=content,
                source_file=source_file,
                chunk_index=chunk_index,
                score=ranked.rerank_score,
                heading_paths=payload.get("heading_paths", []),
            )
            block_tokens = estimate_tokens(block, self.config.model_name)
            if documents and used_tokens + block_tokens > max_tokens:
                break

            used_tokens += block_tokens
            documents.append(ContextDocument(
                doc_id=doc_id,
                content=content,
                score=ranked.rerank_score,
                source_file=source_file,
                chunk_index=chunk_index,
                heading_paths=payload.get("heading_paths", []),
                metadata={
                    "retrieval_score": ranked.retrieval_score,
                    "lexical_score": ranked.lexical_score,
                    "diversity_score": ranked.diversity_score,
                    "mode_score": ranked.mode_score,
                    "source_hits": ranked.chunk.source_hits,
                    "truncated": content != ranked.content.strip(),
                    "payload": payload,
                },
            ))
        return documents

    def build_context(
        self,
        question: str,
        ranked_chunks: Iterable[RankedChunk],
        memories: Optional[Iterable[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        allow_general_knowledge_fallback: bool = True,
        evidence_status: Optional[Dict[str, Any]] = None,
    ) -> BuiltContext:
        documents = self.build_top_k_documents(ranked_chunks, top_k=top_k, max_tokens=max_tokens)
        memories_list = list(memories or [])
        general_knowledge_items = [
            memory for memory in memories_list
            if memory.get("type") == "general_knowledge"
        ]
        citation_memories = [
            memory for memory in memories_list
            if memory.get("type") != "general_knowledge"
        ]
        evidence_status = dict(evidence_status or {})
        original_question = evidence_status.get("original_question") or question
        response_language = evidence_status.get("response_language") or "Chinese"
        opening = (
            "You are a private-domain QA assistant. Prefer local documents, memories, skills, and tool context. "
            "If local documents and memories are insufficient and no useful online/tool context is available, first clearly say in Chinese that local evidence is insufficient, "
            "then answer from model general knowledge. Do not present general knowledge as [D] or [M] cited local evidence."
            if allow_general_knowledge_fallback
            else "You are a private-domain QA assistant. Answer from the provided context. If the context is insufficient, say so clearly in Chinese."
        )

        sections = [
            opening,
            "",
            f"Original user question: {original_question}",
            f"English retrieval question: {question}",
            f"Final answer language: {response_language}",
        ]

        if allow_general_knowledge_fallback:
            sections.extend([
                "",
                "Answer mode: local context is insufficient and online search is unavailable or returned no useful result. Clearly separate local evidence from model general knowledge.",
            ])

        if citation_memories:
            sections.extend(["", "Available memories:"])
            for index, memory in enumerate(citation_memories, start=1):
                content = memory.get("content", "")
                role = memory.get("role") or memory.get("type") or "memory"
                sections.append(f"[M{index}] {role}: {content}")

        if general_knowledge_items:
            sections.extend([
                "",
                "Model general-knowledge draft (not local document/memory evidence; do not cite it with [D] or [M]):",
            ])
            for index, memory in enumerate(general_knowledge_items, start=1):
                sections.append(f"[G{index}] {memory.get('content', '')}")

        sections.extend(["", "Retrieved documents:"])
        if documents:
            for document in documents:
                sections.append(self._format_document_block(
                    doc_id=document.doc_id,
                    content=document.content,
                    source_file=document.source_file,
                    chunk_index=document.chunk_index,
                    score=document.score,
                    heading_paths=document.heading_paths,
                ))
        else:
            sections.append("No retrieved documents are available.")

        sections.extend([
            "",
            "Answer requirements:",
            "- The final answer must be written in Chinese.",
            "- Prefer facts from retrieved documents and memories when they are relevant.",
            "- Cite local documents with IDs such as [D1] and memories with IDs such as [M1] when used.",
            "- Do not invent facts that are not supported by the provided local context.",
            "- For complex questions, synthesize multiple sources first, then provide a structured and sufficiently detailed answer.",
            "- Choose the number of citations according to answer needs; cite multiple sources when multiple sources are useful.",
        ])
        if allow_general_knowledge_fallback:
            sections.extend([
                "- If retrieved documents and memories are insufficient, first say: 本地文档/记忆中没有找到足够依据。",
                "- Then answer using model general knowledge; if a general-knowledge draft is provided, organize it into the final answer.",
                "- General knowledge and drafts are not local sources, so do not add [D] or [M] citations to that part.",
            ])

        context = "\n".join(sections)
        return BuiltContext(
            question=question,
            context=context,
            documents=documents,
            memories=memories_list,
            token_estimate=estimate_tokens(context, self.config.model_name),
            allow_general_knowledge_fallback=allow_general_knowledge_fallback,
            evidence_status=evidence_status,
        )

    def _format_document_block(
        self,
        doc_id: str,
        content: str,
        source_file: Optional[str],
        chunk_index: Optional[int],
        score: float,
        heading_paths: Optional[List[str]] = None,
    ) -> str:
        source = source_file or "unknown"
        chunk = "unknown" if chunk_index is None else str(chunk_index)
        heading_text = ""
        if heading_paths:
            heading_text = f" | headings: {' / '.join(heading_paths)}"
        return (
            f"[{doc_id}] source: {source} | chunk: {chunk} | score: {score:.4f}{heading_text}\n"
            f"{content}"
        )

    def _trim_document_content(self, content: str) -> str:
        max_tokens = self.config.rag_context_max_doc_tokens
        if max_tokens <= 0 or not content:
            return content
        if estimate_tokens(content, self.config.model_name) <= max_tokens:
            return content

        suffix = "\n...[片段已截断以容纳更多引用来源]"
        low = 0
        high = len(content)
        best = content[: max(1, min(len(content), max_tokens * 2))]
        while low <= high:
            mid = (low + high) // 2
            candidate = content[:mid].rstrip() + suffix
            if estimate_tokens(candidate, self.config.model_name) <= max_tokens:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best
