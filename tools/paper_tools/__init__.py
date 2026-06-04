"""Academic paper acquisition and parsing tools.

This package contains local Python tools for:
- searching academic metadata and OA PDF availability,
- reading PDF files from URL/local path,
- parsing papers into structured sections and RAG chunks.
"""

from .pdf_reader_tool import paper_pdf_reader_tool
from .parser_tool import paper_parser_tool
from .registry import PAPER_TOOLS, get_paper_tool, list_paper_tools
from .search_tool import paper_search_tool

__all__ = [
    "PAPER_TOOLS",
    "get_paper_tool",
    "list_paper_tools",
    "paper_search_tool",
    "paper_pdf_reader_tool",
    "paper_parser_tool",
]
