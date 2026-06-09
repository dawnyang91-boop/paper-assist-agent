from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SourceRef:
    source_file: str
    chunk_index: Optional[int] = None
    doc_id: Optional[str] = None
    heading_paths: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "heading_paths": list(self.heading_paths),
        }


FOLLOW_UP_MARKERS = {
    "这篇",
    "这篇文章",
    "这篇论文",
    "刚才",
    "上面",
    "前面",
    "它",
    "本文",
    "该文",
    "这项工作",
    "这个方法",
    "这个模型",
}


def looks_like_follow_up(question: str) -> bool:
    text = (question or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in FOLLOW_UP_MARKERS):
        return True
    return len(text) <= 80 and any(token in text for token in ("实验", "方法", "创新", "贡献", "缺点", "结论", "效果", "指标"))


def extract_source_refs_from_events(events: Iterable[Dict[str, Any]], limit: int = 12) -> List[SourceRef]:
    refs: List[SourceRef] = []
    seen = set()
    for event in reversed(list(events or [])[-limit:]):
        metadata = event.get("metadata") or {}
        references = metadata.get("references") if isinstance(metadata, dict) else None
        if not isinstance(references, dict):
            continue
        for source in references.get("sources") or []:
            if not isinstance(source, dict):
                continue
            source_file = source.get("source_file")
            if not source_file:
                continue
            chunk_index = source.get("chunk_index")
            try:
                chunk_index = int(chunk_index) if chunk_index is not None else None
            except (TypeError, ValueError):
                chunk_index = None
            key = (str(source_file), chunk_index)
            if key in seen:
                continue
            seen.add(key)
            refs.append(SourceRef(
                source_file=str(source_file),
                chunk_index=chunk_index,
                doc_id=str(source.get("doc_id")) if source.get("doc_id") else None,
                heading_paths=tuple(str(item) for item in source.get("heading_paths") or []),
            ))
    return refs


def source_files_from_refs(refs: Iterable[SourceRef]) -> List[str]:
    seen = set()
    files = []
    for ref in refs:
        if not ref.source_file or ref.source_file in seen:
            continue
        seen.add(ref.source_file)
        files.append(ref.source_file)
    return files
