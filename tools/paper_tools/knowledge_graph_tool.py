"""Mermaid knowledge-graph generator for paper notes.

The tool emits Markdown containing a Mermaid graph block, so GitHub, many
Markdown editors, and compatible note systems can render the graph directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common import clean_text, ensure_dir, fail, ok, safe_filename, sha1_text


def paper_knowledge_graph_tool(
    paper_metadata: Optional[Dict[str, Any]] = None,
    nodes: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
    sections: Optional[List[Dict[str, Any]]] = None,
    summary_markdown: Optional[str] = None,
    graph_direction: str = "TD",
    output_title: Optional[str] = None,
    save: bool = True,
    save_dir: str = "data/papers/graphs",
    filename: Optional[str] = None,
    include_metadata_table: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    """Generate a Markdown file containing a Mermaid knowledge graph."""
    metadata = paper_metadata or {}
    graph_direction = graph_direction if graph_direction in {"TD", "TB", "BT", "LR", "RL"} else "TD"
    title = clean_text(output_title) or clean_text(metadata.get("title")) or "Paper Knowledge Graph"

    normalized_nodes = _normalize_nodes(nodes or [])
    normalized_edges = _normalize_edges(edges or [])

    if not normalized_nodes:
        normalized_nodes, normalized_edges = _derive_graph_from_sections(
            metadata=metadata,
            sections=sections or [],
            summary_markdown=summary_markdown,
        )

    if not normalized_nodes:
        return fail(
            "Cannot build knowledge graph because no nodes or derivable paper structure were provided.",
            error="empty graph input",
            data={"manual_edit_required": True, "paper_metadata": metadata},
        )

    mermaid = _build_mermaid(normalized_nodes, normalized_edges, graph_direction=graph_direction)
    markdown = _build_markdown(
        title=title,
        metadata=metadata,
        mermaid=mermaid,
        include_metadata_table=include_metadata_table,
    )

    saved_path = None
    if save:
        directory = ensure_dir(save_dir)
        file_name = filename or f"{safe_filename(title)}_knowledge_graph.md"
        if not file_name.endswith(".md"):
            file_name += ".md"
        path = directory / file_name
        path.write_text(markdown, encoding="utf-8")
        saved_path = str(path)

    return ok(
        {
            "status": "saved" if saved_path else "generated",
            "paper_metadata": metadata,
            "mermaid": mermaid,
            "markdown": markdown,
            "saved_path": saved_path,
            "node_count": len(normalized_nodes),
            "edge_count": len(normalized_edges),
        },
        message="Paper knowledge graph generated.",
    )


def _normalize_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    seen = set()
    for index, node in enumerate(nodes):
        label = clean_text(node.get("label") or node.get("name") or node.get("id"))
        if not label:
            continue
        node_id = _safe_mermaid_id(node.get("id") or label or f"N{index}")
        if node_id in seen:
            node_id = f"{node_id}_{index}"
        seen.add(node_id)
        normalized.append({
            "id": node_id,
            "label": label,
            "type": clean_text(node.get("type")) or "concept",
        })
    return normalized


def _normalize_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for edge in edges:
        source = clean_text(edge.get("source") or edge.get("from"))
        target = clean_text(edge.get("target") or edge.get("to"))
        if not source or not target:
            continue
        normalized.append({
            "source": _safe_mermaid_id(source),
            "target": _safe_mermaid_id(target),
            "relation": clean_text(edge.get("relation") or edge.get("label")) or "relates to",
        })
    return normalized


def _derive_graph_from_sections(
    metadata: Dict[str, Any],
    sections: List[Dict[str, Any]],
    summary_markdown: Optional[str],
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    title = clean_text(metadata.get("title")) or "Paper"
    nodes = [
        {"id": "paper", "label": title, "type": "paper"},
        {"id": "task", "label": "研究问题 / Task", "type": "question"},
        {"id": "method", "label": "方法 / Method", "type": "method"},
        {"id": "experiment", "label": "实验 / Experiment", "type": "experiment"},
        {"id": "conclusion", "label": "结论 / Conclusion", "type": "finding"},
    ]
    edges = [
        {"source": "paper", "target": "task", "relation": "addresses"},
        {"source": "task", "target": "method", "relation": "motivates"},
        {"source": "method", "target": "experiment", "relation": "validated by"},
        {"source": "experiment", "target": "conclusion", "relation": "supports"},
    ]

    keywords = metadata.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [item.strip() for item in re.split(r"[,;]", keywords) if item.strip()]
    for idx, keyword in enumerate(keywords[:8]):
        node_id = f"kw_{idx}"
        nodes.append({"id": node_id, "label": keyword, "type": "keyword"})
        edges.append({"source": "paper", "target": node_id, "relation": "keyword"})

    section_titles = [clean_text(section.get("title")) for section in sections]
    for idx, section_title in enumerate([s for s in section_titles if s][:10]):
        node_id = f"section_{idx}"
        nodes.append({"id": node_id, "label": section_title, "type": "section"})
        edges.append({"source": "paper", "target": node_id, "relation": "contains"})

    if summary_markdown:
        for heading in _extract_summary_headings(summary_markdown)[:8]:
            node_id = _safe_mermaid_id(f"summary_{heading}")
            nodes.append({"id": node_id, "label": heading, "type": "summary"})
            edges.append({"source": "paper", "target": node_id, "relation": "summarized as"})

    return nodes, edges


def _build_mermaid(nodes: List[Dict[str, str]], edges: List[Dict[str, str]], graph_direction: str) -> str:
    lines = [f"flowchart {graph_direction}"]
    for node in nodes:
        node_id = node["id"]
        label = _escape_label(node["label"])
        shape = _shape_for_type(node.get("type"))
        lines.append(f"    {node_id}{shape[0]}\"{label}\"{shape[1]}")
    known_ids = {node["id"] for node in nodes}
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source not in known_ids or target not in known_ids:
            continue
        relation = _escape_label(edge.get("relation") or "relates to")
        lines.append(f"    {source} -- \"{relation}\" --> {target}")
    return "\n".join(lines)


def _build_markdown(title: str, metadata: Dict[str, Any], mermaid: str, include_metadata_table: bool) -> str:
    lines = [f"# 🧠 {title} 知识图谱", ""]
    if include_metadata_table:
        lines.extend([
            "## 论文信息",
            "",
            "| 字段 | 内容 |",
            "|---|---|",
            f"| Title | {metadata.get('title') or ''} |",
            f"| Authors | {'; '.join(metadata.get('authors') or []) if isinstance(metadata.get('authors'), list) else metadata.get('authors') or ''} |",
            f"| Venue | {metadata.get('venue') or ''} |",
            f"| Year | {metadata.get('year') or ''} |",
            f"| DOI | {metadata.get('doi') or ''} |",
            "",
        ])
    lines.extend([
        "## Mermaid Graph",
        "",
        "```mermaid",
        mermaid,
        "```",
        "",
        "## 使用说明",
        "",
        "该文件可在支持 Mermaid 的 Markdown 环境中直接渲染，例如 GitHub、部分 Markdown 编辑器或文档站点。",
    ])
    return "\n".join(lines)


def _shape_for_type(node_type: Optional[str]) -> tuple[str, str]:
    node_type = (node_type or "").lower()
    if node_type in {"paper", "root"}:
        return "([", "])"
    if node_type in {"question", "task", "problem"}:
        return "{", "}"
    if node_type in {"method", "model", "algorithm"}:
        return "[", "]"
    if node_type in {"finding", "conclusion", "result"}:
        return "([", "])"
    return "[", "]"


def _safe_mermaid_id(value: Any) -> str:
    text = clean_text(value) or "node"
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "node"
    if text[0].isdigit():
        text = f"N_{text}"
    return text[:64]


def _escape_label(value: str) -> str:
    return str(value).replace('"', "'").replace("\n", " ")


def _extract_summary_headings(markdown: str) -> List[str]:
    headings = []
    for line in markdown.splitlines():
        match = re.match(r"^#{2,4}\s+(.+)$", line.strip())
        if match:
            heading = clean_text(re.sub(r"^[#\s]+", "", match.group(1)))
            if heading:
                headings.append(heading)
    return headings
