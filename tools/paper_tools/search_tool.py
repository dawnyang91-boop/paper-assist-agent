"""Academic paper search tool.

This tool uses free metadata/OA services when available:
- Crossref for DOI/title metadata
- OpenAlex for metadata and OA location
- Unpaywall for DOI-level OA PDF status
- arXiv for preprints

The implementation is intentionally dependency-light and uses the Python standard
library for HTTP calls. API failures are collected as warnings instead of causing
the whole tool to fail.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .common import (
    build_query_url,
    clean_text,
    fail,
    first,
    http_json,
    infer_query_type,
    make_pdf_access,
    normalize_doi,
    ok,
    sha1_text,
    strip_html,
)

DEFAULT_SOURCES = ["crossref", "openalex", "unpaywall", "arxiv"]


def paper_search_tool(
    query: str,
    query_type: str = "auto",
    sources: Optional[List[str]] = None,
    limit: int = 10,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    oa_only: bool = False,
    require_pdf: bool = True,
    missing_pdf_policy: str = "include_and_mark",
    with_pdf_url: bool = True,
    email: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Search paper metadata and report OA PDF availability explicitly."""
    query = clean_text(query) or ""
    if not query:
        return fail("Paper search query is empty.", error="query is required")

    limit = max(1, min(int(limit or 10), 50))
    query_type = infer_query_type(query) if query_type == "auto" else query_type
    selected_sources = sources or DEFAULT_SOURCES
    selected_sources = [source for source in selected_sources if source]
    email = email or os.getenv("UNPAYWALL_EMAIL") or os.getenv("CROSSREF_MAILTO")

    warnings: List[str] = []
    raw_papers: List[Dict[str, Any]] = []

    if "arxiv" in selected_sources and query_type in {"auto", "keyword", "title", "arxiv_id"}:
        try:
            raw_papers.extend(_search_arxiv(query=query, query_type=query_type, limit=limit))
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.append(f"arxiv search failed: {exc}")

    if "openalex" in selected_sources:
        try:
            raw_papers.extend(_search_openalex(query=query, query_type=query_type, limit=limit))
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.append(f"openalex search failed: {exc}")

    if "crossref" in selected_sources:
        try:
            raw_papers.extend(_search_crossref(query=query, query_type=query_type, limit=limit, email=email))
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.append(f"crossref search failed: {exc}")

    papers = _dedupe_papers(raw_papers)
    papers = _filter_by_year(papers, year_from=year_from, year_to=year_to)

    checked_sources = list(selected_sources)
    if "unpaywall" in selected_sources:
        for paper in papers:
            doi = paper.get("doi")
            if not doi:
                continue
            try:
                _enrich_with_unpaywall(paper, doi=doi, email=email)
            except Exception as exc:  # pragma: no cover - network dependent
                warnings.append(f"unpaywall lookup failed for {doi}: {exc}")

    for paper in papers:
        _finalize_pdf_access(paper, checked_sources=checked_sources)

    if oa_only:
        papers = [p for p in papers if p.get("open_access", {}).get("is_oa")]

    if missing_pdf_policy == "exclude":
        papers = [p for p in papers if p.get("pdf_access", {}).get("status") == "available"]
    elif missing_pdf_policy == "include_metadata_only":
        for paper in papers:
            if paper.get("pdf_access", {}).get("status") != "available":
                paper["processing_status"] = "search_only"

    if with_pdf_url:
        papers.sort(key=lambda p: 0 if p.get("pdf_access", {}).get("status") == "available" else 1)

    papers = papers[:limit]
    available_pdf_papers = [_paper_summary(p) for p in papers if p.get("pdf_access", {}).get("status") == "available"]
    missing_pdf_papers = [_missing_summary(p) for p in papers if p.get("pdf_access", {}).get("status") != "available"]

    data = {
        "query": query,
        "query_type": query_type,
        "sources": selected_sources,
        "total": len(papers),
        "available_pdf_count": len(available_pdf_papers),
        "missing_pdf_count": len(missing_pdf_papers),
        "manual_upload_required": bool(missing_pdf_papers) if require_pdf else False,
        "papers": papers,
        "available_pdf_papers": available_pdf_papers,
        "missing_pdf_papers": missing_pdf_papers,
        "warnings": warnings,
    }
    message = "Paper search completed."
    if missing_pdf_papers and require_pdf:
        message += f" {len(missing_pdf_papers)} paper(s) require manual PDF upload."
    return ok(data, message=message)


def _search_openalex(query: str, query_type: str, limit: int) -> List[Dict[str, Any]]:
    if query_type == "doi":
        doi = normalize_doi(query)
        if not doi:
            return []
        url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
        return [_paper_from_openalex(http_json(url))]

    params = {"search": query, "per-page": limit}
    url = build_query_url("https://api.openalex.org/works", params)
    payload = http_json(url)
    return [_paper_from_openalex(item) for item in payload.get("results", [])]


def _paper_from_openalex(item: Dict[str, Any]) -> Dict[str, Any]:
    locations = item.get("locations") or []
    best_oa = item.get("best_oa_location") or {}
    pdf_url = best_oa.get("pdf_url") or _first_pdf_from_locations(locations)
    landing_page_url = best_oa.get("landing_page_url") or item.get("primary_location", {}).get("landing_page_url")
    doi = normalize_doi(item.get("doi"))
    title = clean_text(item.get("title"))
    authorships = item.get("authorships") or []
    authors = [clean_text(a.get("author", {}).get("display_name")) for a in authorships]
    authors = [a for a in authors if a]
    host = item.get("primary_location", {}).get("source") or {}
    oa = item.get("open_access") or {}
    paper = {
        "paper_id": item.get("id") or f"openalex:{sha1_text(str(item))}",
        "title": title,
        "authors": authors,
        "abstract": _openalex_abstract(item.get("abstract_inverted_index")),
        "doi": doi,
        "venue": clean_text(host.get("display_name")),
        "year": item.get("publication_year"),
        "publisher": clean_text(host.get("host_organization_name")),
        "citation_count": item.get("cited_by_count"),
        "open_access": {
            "is_oa": bool(oa.get("is_oa")) or bool(pdf_url),
            "oa_status": oa.get("oa_status"),
            "source": "openalex",
        },
        "pdf_url": pdf_url,
        "landing_page_url": landing_page_url,
        "source": "openalex",
        "processing_status": "search_only",
    }
    if pdf_url:
        paper["pdf_access"] = make_pdf_access("available", pdf_url, "openalex", ["openalex"], license=best_oa.get("license"))
    return paper


def _search_crossref(query: str, query_type: str, limit: int, email: Optional[str]) -> List[Dict[str, Any]]:
    headers = {}
    if email:
        headers["mailto"] = email
    if query_type == "doi":
        doi = normalize_doi(query)
        if not doi:
            return []
        url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        payload = http_json(url, headers=headers)
        item = payload.get("message") or {}
        return [_paper_from_crossref(item)]
    params = {"rows": limit}
    if query_type == "title":
        params["query.title"] = query
    elif query_type == "author":
        params["query.author"] = query
    else:
        params["query"] = query
    url = build_query_url("https://api.crossref.org/works", params)
    payload = http_json(url, headers=headers)
    return [_paper_from_crossref(item) for item in payload.get("message", {}).get("items", [])]


def _paper_from_crossref(item: Dict[str, Any]) -> Dict[str, Any]:
    title = clean_text(first(item.get("title")))
    authors = []
    for author in item.get("author") or []:
        name = clean_text(" ".join(part for part in [author.get("given"), author.get("family")] if part))
        if name:
            authors.append(name)
    year = None
    for key in ["published-print", "published-online", "issued"]:
        parts = item.get(key, {}).get("date-parts") or []
        if parts and parts[0]:
            year = parts[0][0]
            break
    links = item.get("link") or []
    pdf_url = None
    for link in links:
        url = link.get("URL")
        content_type = (link.get("content-type") or "").lower()
        if url and ("pdf" in content_type or url.lower().endswith(".pdf")):
            pdf_url = url
            break
    paper = {
        "paper_id": f"crossref:{item.get('DOI') or sha1_text(str(item))}",
        "title": title,
        "authors": authors,
        "abstract": strip_html(item.get("abstract")),
        "doi": normalize_doi(item.get("DOI")),
        "venue": clean_text(first(item.get("container-title"))),
        "year": year,
        "publisher": clean_text(item.get("publisher")),
        "citation_count": item.get("is-referenced-by-count"),
        "open_access": {"is_oa": bool(pdf_url), "oa_status": None, "source": "crossref"},
        "pdf_url": pdf_url,
        "landing_page_url": item.get("URL"),
        "source": "crossref",
        "processing_status": "search_only",
    }
    if pdf_url:
        paper["pdf_access"] = make_pdf_access("available", pdf_url, "crossref", ["crossref"])
    return paper


def _search_arxiv(query: str, query_type: str, limit: int) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET

    if query_type == "arxiv_id":
        search_query = f"id:{query}"
    elif query_type == "title":
        search_query = f"ti:{query}"
    else:
        search_query = f"all:{query}"
    url = build_query_url(
        "https://export.arxiv.org/api/query",
        {"search_query": search_query, "start": 0, "max_results": limit},
    )
    # arXiv returns Atom XML, not JSON.
    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": "paper-assist-agent/0.1"})
    with urlopen(request, timeout=20) as response:
        xml = response.read()
    root = ET.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        arxiv_id_url = clean_text(entry.findtext("atom:id", default="", namespaces=ns))
        arxiv_id = arxiv_id_url.rsplit("/", 1)[-1] if arxiv_id_url else None
        authors = [clean_text(a.findtext("atom:name", default="", namespaces=ns)) for a in entry.findall("atom:author", ns)]
        authors = [a for a in authors if a]
        year = None
        published = clean_text(entry.findtext("atom:published", default="", namespaces=ns))
        if published and len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        paper = {
            "paper_id": f"arxiv:{arxiv_id or sha1_text(arxiv_id_url or title or '')}",
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "doi": None,
            "arxiv_id": arxiv_id,
            "venue": "arXiv",
            "year": year,
            "publisher": "arXiv",
            "citation_count": None,
            "open_access": {"is_oa": bool(pdf_url), "oa_status": "green", "source": "arxiv"},
            "pdf_url": pdf_url,
            "landing_page_url": arxiv_id_url,
            "source": "arxiv",
            "processing_status": "search_only",
        }
        if pdf_url:
            paper["pdf_access"] = make_pdf_access("available", pdf_url, "arxiv", ["arxiv"])
        papers.append(paper)
    return papers


def _enrich_with_unpaywall(paper: Dict[str, Any], doi: str, email: Optional[str]) -> None:
    if not email:
        # Unpaywall asks callers to identify with an email. Keep search usable while
        # making the missing enrichment explicit.
        return
    url = build_query_url(f"https://api.unpaywall.org/v2/{quote(doi, safe='')}", {"email": email})
    payload = http_json(url)
    best = payload.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf") or best.get("url")
    is_oa = bool(payload.get("is_oa"))
    paper["open_access"] = {
        "is_oa": is_oa,
        "oa_status": payload.get("oa_status"),
        "source": "unpaywall",
    }
    if pdf_url:
        paper["pdf_url"] = paper.get("pdf_url") or pdf_url
        paper["pdf_access"] = make_pdf_access(
            "available",
            pdf_url,
            "unpaywall",
            [paper.get("source"), "unpaywall"],
            license=best.get("license"),
        )


def _finalize_pdf_access(paper: Dict[str, Any], checked_sources: List[str]) -> None:
    pdf_access = paper.get("pdf_access") or {}
    pdf_url = pdf_access.get("pdf_url") or paper.get("pdf_url")
    if pdf_url:
        paper["pdf_url"] = pdf_url
        paper["pdf_access"] = make_pdf_access(
            "available",
            pdf_url,
            pdf_access.get("source") or paper.get("source"),
            checked_sources,
            license=pdf_access.get("license"),
        )
        paper["processing_status"] = "search_only"
        return
    status = "restricted" if paper.get("landing_page_url") else "missing"
    paper["pdf_url"] = None
    paper["pdf_access"] = make_pdf_access(status, None, None, checked_sources)
    paper["processing_status"] = "pdf_missing"


def _first_pdf_from_locations(locations: List[Dict[str, Any]]) -> Optional[str]:
    for location in locations:
        pdf_url = location.get("pdf_url")
        if pdf_url:
            return pdf_url
    return None


def _openalex_abstract(index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not isinstance(index, dict) or not index:
        return None
    words: List[Optional[str]] = [None] * (max(pos for positions in index.values() for pos in positions) + 1)
    for word, positions in index.items():
        for pos in positions:
            words[pos] = word
    return clean_text(" ".join(word or "" for word in words))


def _dedupe_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for paper in papers:
        doi = normalize_doi(paper.get("doi"))
        title_key = clean_text(paper.get("title") or "") or ""
        key = doi.lower() if doi else title_key.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        paper["doi"] = doi
        deduped.append(paper)
    return deduped


def _filter_by_year(
    papers: List[Dict[str, Any]],
    year_from: Optional[int],
    year_to: Optional[int],
) -> List[Dict[str, Any]]:
    filtered = []
    for paper in papers:
        year = paper.get("year")
        if year_from and year and int(year) < int(year_from):
            continue
        if year_to and year and int(year) > int(year_to):
            continue
        filtered.append(paper)
    return filtered


def _paper_summary(paper: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": paper.get("title"),
        "doi": paper.get("doi"),
        "pdf_url": paper.get("pdf_url"),
        "source": paper.get("pdf_access", {}).get("source"),
    }


def _missing_summary(paper: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": paper.get("title"),
        "doi": paper.get("doi"),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
        "landing_page_url": paper.get("landing_page_url"),
        "reason": paper.get("pdf_access", {}).get("message") or "No open access PDF found from free APIs.",
        "checked_sources": paper.get("pdf_access", {}).get("checked_sources", []),
        "suggested_action": "Please manually upload the PDF if you have legal access.",
    }
