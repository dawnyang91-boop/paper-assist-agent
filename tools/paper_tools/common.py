"""Shared helpers for academic paper tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = "paper-assist-agent/0.1 (+https://github.com/dawnyang91-boop/paper-assist-agent)"


def ok(data: Dict[str, Any], message: str = "Tool executed successfully.") -> Dict[str, Any]:
    return {"success": True, "data": data, "message": message, "error": None}


def fail(
    message: str,
    error: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"success": False, "data": data, "message": message, "error": error or message}


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def strip_html(value: Any) -> Optional[str]:
    text = clean_text(value)
    if not text:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    return clean_text(text)


def first(items: Any) -> Optional[Any]:
    if isinstance(items, list) and items:
        return items[0]
    return items


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def safe_filename(value: str, default: str = "paper") -> str:
    text = clean_text(value) or default
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")[:120]
    return text or default


def infer_query_type(query: str) -> str:
    q = (query or "").strip()
    lowered = q.lower()
    if re.fullmatch(r"10\.\d{4,9}/\S+", q):
        return "doi"
    if lowered.startswith("doi:") and re.search(r"10\.\d{4,9}/\S+", lowered):
        return "doi"
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", lowered):
        return "arxiv_id"
    if len(q.split()) >= 5:
        return "title"
    return "keyword"


def normalize_doi(value: Any) -> Optional[str]:
    text = clean_text(value)
    if not text:
        return None
    text = re.sub(r"^(doi:|https?://(dx\.)?doi\.org/)", "", text, flags=re.I).strip()
    match = re.search(r"10\.\d{4,9}/\S+", text)
    if not match:
        return None
    doi = match.group(0).rstrip(".,;)"]}")
    return doi


def http_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    req_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    request = Request(url, headers=req_headers)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def http_bytes(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    req_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/pdf,*/*"}
    if headers:
        req_headers.update(headers)
    request = Request(url, headers=req_headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def looks_like_url(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def build_query_url(base: str, params: Dict[str, Any]) -> str:
    clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
    return f"{base}?{urlencode(clean_params)}"


def make_pdf_access(
    status: str,
    pdf_url: Optional[str],
    source: Optional[str],
    checked_sources: Iterable[str],
    message: Optional[str] = None,
    license: Optional[str] = None,
) -> Dict[str, Any]:
    if status == "available":
        default_message = "Open access PDF found."
    elif status == "broken_url":
        default_message = "A PDF URL was found but it could not be accessed. Manual upload may be required."
    elif status == "restricted":
        default_message = "A landing page was found but no free downloadable PDF was available."
    else:
        default_message = "No open access PDF was found from the configured free APIs. Manual upload is required."
    return {
        "status": status,
        "pdf_url": pdf_url,
        "source": source,
        "license": license,
        "checked_sources": list(dict.fromkeys(checked_sources)),
        "message": message or default_message,
    }


def compact_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = metadata or {}
    return {
        "title": metadata.get("title"),
        "doi": metadata.get("doi"),
        "authors": metadata.get("authors") or [],
        "venue": metadata.get("venue"),
        "year": metadata.get("year"),
        "landing_page_url": metadata.get("landing_page_url"),
    }


def current_timestamp() -> float:
    return time.time()


def ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
