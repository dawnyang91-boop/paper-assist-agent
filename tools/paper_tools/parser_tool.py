"""Academic paper parser tool.

The default backend is `auto`: it first turns PDF input into text with the PDF
reader, then applies a lightweight academic-section parser. This keeps the tool
usable in MVP environments without a running GROBID server, while preserving the
schema fields needed to add GROBID/Unstructured later.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .common import clean_text, compact_metadata, fail, ok, sha1_text
from .pdf_reader_tool import paper_pdf_reader_tool

SECTION_PATTERNS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiment",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "limitation",
    "limitations",
    "conclusion",
    "references",
]


def paper_parser_tool(
    input_source: str,
    input_type: str = "auto",
    paper_metadata: Optional[Dict[str, Any]] = None,
    parser_backend: str = "auto",
    parse_level: str = "full",
    chunk_strategy: str = "by_section",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    include_references: bool = True,
    include_figures: bool = True,
    include_tables: bool = True,
    normalize_text: bool = True,
    output_format: str = "json",
    on_missing_input: str = "return_missing_status",
    **_: Any,
) -> Dict[str, Any]:
    """Parse a paper into structured metadata, sections, references and chunks."""
    metadata = compact_metadata(paper_metadata)
    input_source = input_source or ""
    if not clean_text(input_source):
        return _missing_input_response(metadata, on_missing_input, "No PDF, raw text, or document ID was provided.")

    input_type = _infer_input_type(input_source, input_type)
    parser_backend = "lightweight" if parser_backend == "auto" else parser_backend

    try:
        text, local_path, reader_metadata = _load_text(input_source, input_type, metadata)
    except Exception as exc:
        return _missing_input_response(metadata, on_missing_input, str(exc))

    if normalize_text:
        text = _normalize_text(text)
    if not text:
        return _missing_input_response(metadata, on_missing_input, "Input was available but no text could be extracted.")

    parsed = _parse_lightweight(text=text, paper_metadata=metadata, include_references=include_references)
    paper_id = _paper_id(metadata, input_source)
    rag_chunks = []
    if chunk_strategy != "none" and parse_level in {"full", "rag_chunks"}:
        rag_chunks = _build_rag_chunks(
            paper_id=paper_id,
            sections=parsed["sections"],
            metadata=parsed["metadata"],
            chunk_strategy=chunk_strategy,
            chunk_size=max(100, min(int(chunk_size or 800), 4000)),
            chunk_overlap=max(0, min(int(chunk_overlap or 100), 1000)),
            source_file=local_path or input_source,
        )

    data = {
        "status": "chunked" if rag_chunks else "parsed",
        "paper_id": paper_id,
        "input_source": input_source,
        "input_type": input_type,
        "parser_backend": parser_backend,
        "metadata": parsed["metadata"],
        "sections": parsed["sections"] if parse_level in {"sections", "full", "rag_chunks"} else [],
        "figures": _extract_captions(text, kind="figure") if include_figures and parse_level == "full" else [],
        "tables": _extract_captions(text, kind="table") if include_tables and parse_level == "full" else [],
        "references": parsed["references"] if include_references and parse_level in {"references", "full"} else [],
        "rag_chunks": rag_chunks,
        "reader_metadata": reader_metadata,
        "manual_upload_required": False,
    }
    if output_format == "markdown":
        data["markdown"] = _to_markdown(data)
    elif output_format == "rag_documents":
        data = {
            "status": data["status"],
            "paper_id": paper_id,
            "rag_documents": [
                {"page_content": chunk["text"], "metadata": chunk["metadata"]}
                for chunk in rag_chunks
            ],
            "manual_upload_required": False,
        }
    return ok(data, message="Paper parsing completed.")


def _infer_input_type(input_source: str, input_type: str) -> str:
    if input_type and input_type != "auto":
        return input_type
    lowered = input_source.lower().strip()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return "pdf_url" if ".pdf" in lowered else "document_id"
    if lowered.endswith(".pdf"):
        return "local_pdf"
    if "\n" in input_source or len(input_source) > 500:
        return "raw_text"
    return "document_id"


def _load_text(input_source: str, input_type: str, paper_metadata: Dict[str, Any]) -> tuple[str, Optional[str], Dict[str, Any]]:
    if input_type == "raw_text":
        return input_source, None, {}
    if input_type in {"pdf_url", "local_pdf"}:
        result = paper_pdf_reader_tool(
            pdf_source=input_source,
            source_type="url" if input_type == "pdf_url" else "local_path",
            paper_metadata=paper_metadata,
            read_mode="full_text",
            on_missing_source="return_missing_status",
        )
        if not result.get("success"):
            raise ValueError(result.get("error") or result.get("message") or "PDF reading failed.")
        data = result.get("data") or {}
        return data.get("text") or "", data.get("local_path"), data.get("metadata") or {}
    raise ValueError("document_id parsing is not implemented yet. Provide raw_text, local_pdf, or pdf_url.")


def _parse_lightweight(text: str, paper_metadata: Dict[str, Any], include_references: bool) -> Dict[str, Any]:
    sections = _split_sections(text)
    abstract = paper_metadata.get("abstract") or _find_abstract(sections, text)
    title = paper_metadata.get("title") or _guess_title(text)
    metadata = {
        **paper_metadata,
        "title": title,
        "abstract": abstract,
        "keywords": _find_keywords(text),
    }
    references = _extract_references(sections) if include_references else []
    return {"metadata": metadata, "sections": sections, "references": references}


def _split_sections(text: str) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    heading_indices = []
    pattern = re.compile(rf"^\s*((\d+(\.\d+)*)\s+)?({'|'.join(re.escape(s) for s in SECTION_PATTERNS)})\b[:\s-]*$", re.I)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if 2 <= len(stripped) <= 80 and pattern.match(stripped):
            heading_indices.append((idx, stripped))

    if not heading_indices:
        return [{"section_id": "1", "title": "Full Text", "level": 1, "text": text.strip()}]

    sections = []
    for pos, (line_idx, heading) in enumerate(heading_indices):
        next_idx = heading_indices[pos + 1][0] if pos + 1 < len(heading_indices) else len(lines)
        body = "\n".join(lines[line_idx + 1: next_idx]).strip()
        clean_heading = re.sub(r"^\d+(\.\d+)*\s+", "", heading).strip(" :-")
        sections.append({
            "section_id": str(pos + 1),
            "title": clean_heading,
            "level": 1,
            "text": body,
        })
    return sections


def _find_abstract(sections: List[Dict[str, Any]], text: str) -> Optional[str]:
    for section in sections:
        if section.get("title", "").lower().startswith("abstract"):
            return clean_text(section.get("text"))
    match = re.search(r"abstract\s*[:\n](.{200,2500}?)(\n\s*(introduction|1\s+introduction)\b)", text, flags=re.I | re.S)
    return clean_text(match.group(1)) if match else None


def _guess_title(text: str) -> Optional[str]:
    for line in text.splitlines()[:20]:
        stripped = clean_text(line)
        if stripped and 8 <= len(stripped) <= 220 and not stripped.lower().startswith(("abstract", "arxiv", "doi")):
            return stripped
    return None


def _find_keywords(text: str) -> List[str]:
    match = re.search(r"keywords?\s*[:—-]\s*(.+)", text, flags=re.I)
    if not match:
        return []
    raw = match.group(1).split("\n", 1)[0]
    return [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]


def _extract_references(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reference_text = ""
    for section in sections:
        if section.get("title", "").lower().startswith("references"):
            reference_text = section.get("text") or ""
            break
    if not reference_text:
        return []
    chunks = re.split(r"\n\s*(?:\[?\d+\]?\.?\s+)", "\n" + reference_text)
    refs = []
    for idx, raw in enumerate(chunks):
        raw = clean_text(raw)
        if not raw or len(raw) < 20:
            continue
        year_match = re.search(r"\b(19|20)\d{2}\b", raw)
        refs.append({
            "ref_id": f"b{len(refs) + 1}",
            "raw_text": raw,
            "title": None,
            "authors": [],
            "year": int(year_match.group(0)) if year_match else None,
            "doi": None,
        })
    return refs


def _extract_captions(text: str, kind: str) -> List[Dict[str, Any]]:
    prefix = "fig" if kind == "figure" else "table"
    pattern = re.compile(rf"\b({prefix}(?:ure)?\.?\s*\d+)\s*[:.-]\s*(.+)", flags=re.I)
    captions = []
    for match in pattern.finditer(text):
        captions.append({
            f"{kind}_id": match.group(1).replace(" ", "").lower(),
            "caption": clean_text(match.group(2))[:500] if clean_text(match.group(2)) else "",
            "page": None,
        })
        if len(captions) >= 50:
            break
    return captions


def _build_rag_chunks(
    paper_id: str,
    sections: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    source_file: str,
) -> List[Dict[str, Any]]:
    chunks = []
    for section in sections:
        title = section.get("title") or "Section"
        text = section.get("text") or ""
        if not text.strip():
            continue
        pieces = _split_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for piece in pieces:
            chunk_id = f"{paper_id}_{len(chunks):04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": piece,
                "metadata": {
                    "paper_id": paper_id,
                    "title": metadata.get("title"),
                    "doi": metadata.get("doi"),
                    "authors": metadata.get("authors") or [],
                    "venue": metadata.get("venue"),
                    "year": metadata.get("year"),
                    "section": title,
                    "source_file": source_file,
                    "chunk_index": len(chunks),
                    "chunk_strategy": chunk_strategy,
                },
            })
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []
    chunks = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current)
        current = (current[-overlap:] + "\n\n" + para).strip() if overlap else para
    if current:
        chunks.append(current)
    return chunks


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paper_id(metadata: Dict[str, Any], input_source: str) -> str:
    if metadata.get("doi"):
        return f"doi:{metadata['doi']}"
    if metadata.get("title"):
        return f"title:{sha1_text(metadata['title'])[:12]}"
    return f"local:{sha1_text(input_source)[:12]}"


def _to_markdown(data: Dict[str, Any]) -> str:
    meta = data.get("metadata") or {}
    lines = [f"# {meta.get('title') or 'Untitled Paper'}", ""]
    if meta.get("doi"):
        lines.append(f"- DOI: {meta['doi']}")
    if meta.get("authors"):
        lines.append(f"- Authors: {', '.join(meta['authors'])}")
    if meta.get("venue") or meta.get("year"):
        lines.append(f"- Venue/Year: {meta.get('venue') or ''} {meta.get('year') or ''}".strip())
    if meta.get("abstract"):
        lines.extend(["", "## Abstract", meta["abstract"]])
    for section in data.get("sections") or []:
        lines.extend(["", f"## {section.get('title')}", section.get("text") or ""])
    return "\n".join(lines).strip()


def _missing_input_response(
    paper_metadata: Dict[str, Any],
    on_missing_input: str,
    reason: str,
) -> Dict[str, Any]:
    data = {
        "status": "cannot_parse",
        "reason": "missing_pdf_or_text",
        "manual_upload_required": True,
        "paper_metadata": paper_metadata,
        "rag_chunks": [],
        "suggested_action": "Please manually upload the PDF before parsing or RAG ingestion.",
    }
    return fail("Paper content is missing. Parser skipped this paper.", error=reason, data=data)
