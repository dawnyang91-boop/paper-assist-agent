"""Registry for local academic paper tools.

This registry mirrors the project's existing tool-registration mindset while
remaining independent from MCP servers. A caller can list tool schemas, fetch a
callable by name, or adapt these entries to OpenAI function tools/FastAPI routes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .pdf_reader_tool import paper_pdf_reader_tool
from .parser_tool import paper_parser_tool
from .schemas import (
    PAPER_PARSER_TOOL_SCHEMA,
    PAPER_PDF_READER_TOOL_SCHEMA,
    PAPER_SEARCH_TOOL_SCHEMA,
)
from .search_tool import paper_search_tool

PAPER_TOOLS: List[Dict[str, Any]] = [
    {"schema": PAPER_SEARCH_TOOL_SCHEMA, "function": paper_search_tool},
    {"schema": PAPER_PDF_READER_TOOL_SCHEMA, "function": paper_pdf_reader_tool},
    {"schema": PAPER_PARSER_TOOL_SCHEMA, "function": paper_parser_tool},
]


def list_paper_tools() -> List[Dict[str, Any]]:
    """Return OpenAI/MCP-compatible schemas for all paper tools."""
    return [item["schema"] for item in PAPER_TOOLS]


def get_paper_tool(name: str) -> Callable[..., Dict[str, Any]]:
    """Return a paper tool callable by its registered name."""
    for item in PAPER_TOOLS:
        schema = item["schema"]
        if schema.get("name") == name:
            return item["function"]
    available = ", ".join(schema["name"] for schema in list_paper_tools())
    raise KeyError(f"Unknown paper tool: {name}. Available tools: {available}")


def as_openai_tools() -> List[Dict[str, Any]]:
    """Convert schemas into OpenAI Chat Completions tool objects."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
        for schema in list_paper_tools()
    ]
