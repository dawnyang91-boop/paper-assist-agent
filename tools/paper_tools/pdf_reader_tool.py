"""PDF reader tool for academic papers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    clean_text,
    compact_metadata,
    ensure_dir,
    fail,
    http_bytes,
    looks_like_url,
    ok,
    safe_filename,
)


def paper_pdf_reader_tool(
    pdf_source: str,
    source_type: str = "auto",
    paper_metadata: Optional[Dict[str, Any]] = None,
    read_mode: str = "full_text",
    page_range: Optional[Dict[str, int]] = None,
    download: bool = True,
    save_dir: str = "data/papers/raw",
    extract_images: bool = False,
    extract_tables: bool = False,
    max_pages: int = 100,
    on_missing_source: str = "return_missing_status",
    **_: Any,
) -> Dict[str, Any]:
    """Read a PDF from URL or local path and return raw/page-level text."""
    metadata = compact_metadata(paper_metadata)
    pdf_source = clean_text(pdf_source) or ""
    if not pdf_source:
        return _missing_pdf_response(metadata, on_missing_source, "No PDF URL or local PDF path was provided.")

    source_type = _infer_source_type(pdf_source, source_type)
    try:
        local_path = _resolve_pdf_source(
            pdf_source=pdf_source,
            source_type=source_type,
            download=download,
            save_dir=save_dir,
            paper_metadata=metadata,
        )
    except Exception as exc:
        return _missing_pdf_response(metadata, on_missing_source, str(exc), pdf_source=pdf_source)

    if not local_path.exists():
        return _missing_pdf_response(metadata, on_missing_source, f"PDF file does not exist: {local_path}", pdf_source=pdf_source)

    try:
        pdf_data = _read_pdf_with_pymupdf(
            local_path=local_path,
            read_mode=read_mode,
            page_range=page_range,
            max_pages=max_pages,
        )
    except Exception as exc:
        return fail(
            "PDF reading failed.",
            error=str(exc),
            data={
                "status": "failed",
                "pdf_source": pdf_source,
                "source_type": source_type,
                "local_path": str(local_path),
                "paper_metadata": metadata,
                "manual_upload_required": False,
            },
        )

    data = {
        "status": "pdf_read",
        "pdf_source": pdf_source,
        "source_type": source_type,
        "local_path": str(local_path),
        "paper_metadata": metadata,
        "metadata": pdf_data["metadata"],
        "text": pdf_data["text"] if read_mode != "metadata_only" else "",
        "pages": pdf_data["pages"] if read_mode in {"page_text", "full_text"} else [],
        "images": [],
        "tables": [],
        "extract_images_requested": extract_images,
        "extract_tables_requested": extract_tables,
        "manual_upload_required": False,
    }
    return ok(data, message="PDF reading completed.")


def _infer_source_type(pdf_source: str, source_type: str) -> str:
    if source_type and source_type != "auto":
        return source_type
    if looks_like_url(pdf_source):
        return "url"
    return "local_path"


def _resolve_pdf_source(
    pdf_source: str,
    source_type: str,
    download: bool,
    save_dir: str,
    paper_metadata: Dict[str, Any],
) -> Path:
    if source_type == "url":
        if not download:
            raise ValueError("Remote PDF URL requires download=True before local reading.")
        return _download_pdf(pdf_source, save_dir=save_dir, paper_metadata=paper_metadata)
    if source_type in {"local_path", "file_id"}:
        return Path(pdf_source).expanduser().resolve()
    raise ValueError(f"Unsupported PDF source_type: {source_type}")


def _download_pdf(url: str, save_dir: str, paper_metadata: Dict[str, Any]) -> Path:
    directory = ensure_dir(save_dir)
    title = paper_metadata.get("title") or paper_metadata.get("doi") or url.rsplit("/", 1)[-1] or "paper"
    filename = safe_filename(title)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    local_path = directory / filename
    content = http_bytes(url)
    if not content.startswith(b"%PDF") and len(content) < 1024:
        raise ValueError("Downloaded content does not look like a valid PDF.")
    local_path.write_bytes(content)
    return local_path


def _read_pdf_with_pymupdf(
    local_path: Path,
    read_mode: str,
    page_range: Optional[Dict[str, int]],
    max_pages: int,
) -> Dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required. Install with `pip install pymupdf`.") from exc

    pages: List[Dict[str, Any]] = []
    with fitz.open(str(local_path)) as document:
        page_count = len(document)
        meta = dict(document.metadata or {})
        start = max(1, int((page_range or {}).get("start") or 1))
        end = int((page_range or {}).get("end") or min(page_count, max_pages))
        end = min(page_count, end, start + max(1, int(max_pages or 1)) - 1)
        if read_mode == "metadata_only":
            return {
                "metadata": _metadata(local_path, meta, page_count),
                "text": "",
                "pages": [],
            }
        for page_number in range(start, end + 1):
            page = document[page_number - 1]
            text = page.get_text("text").strip()
            pages.append({"page_number": page_number, "text": text})
    full_text = "\n\n".join(page["text"] for page in pages if page.get("text"))
    return {
        "metadata": _metadata(local_path, meta, page_count),
        "text": full_text,
        "pages": pages,
    }


def _metadata(local_path: Path, fitz_metadata: Dict[str, Any], page_count: int) -> Dict[str, Any]:
    return {
        "title": clean_text(fitz_metadata.get("title")),
        "author": clean_text(fitz_metadata.get("author")),
        "subject": clean_text(fitz_metadata.get("subject")),
        "keywords": clean_text(fitz_metadata.get("keywords")),
        "page_count": page_count,
        "file_size": local_path.stat().st_size if local_path.exists() else None,
    }


def _missing_pdf_response(
    paper_metadata: Dict[str, Any],
    on_missing_source: str,
    reason: str,
    pdf_source: Optional[str] = None,
) -> Dict[str, Any]:
    data = {
        "status": "missing_pdf",
        "pdf_source": pdf_source,
        "paper_metadata": paper_metadata,
        "manual_upload_required": True,
        "suggested_action": "Please manually upload the PDF if you have legal access.",
    }
    if on_missing_source == "raise_error":
        return fail("PDF source is missing.", error=reason, data=data)
    return fail("PDF source is missing. Manual upload is required.", error=reason, data=data)
