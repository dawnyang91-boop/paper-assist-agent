"""JSON schemas for academic paper tools.

The schemas are intentionally OpenAI-function/MCP compatible: each tool has a
unique name, a clear description, and an object-shaped parameters schema.
"""

PAPER_SEARCH_TOOL_SCHEMA = {
    "name": "paper_search_tool",
    "description": (
        "Search academic papers by keyword, title, DOI, author, venue, or arXiv ID. "
        "The tool returns metadata and explicitly reports whether an open-access "
        "PDF can be found through free APIs. Papers without PDFs are kept and marked "
        "as requiring manual upload."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Paper title, DOI, keyword, author name, venue name, or arXiv ID.",
            },
            "query_type": {
                "type": "string",
                "enum": ["auto", "title", "doi", "keyword", "author", "venue", "arxiv_id"],
                "default": "auto",
                "description": "How to interpret query. Use auto unless the caller knows the exact type.",
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "crossref",
                        "openalex",
                        "unpaywall",
                        "arxiv",
                        "pubmed_central",
                        "europe_pmc",
                        "core",
                    ],
                },
                "default": ["crossref", "openalex", "unpaywall", "arxiv"],
                "description": "Free metadata/OA APIs to query. Empty means auto source selection.",
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum number of paper records to return.",
            },
            "year_from": {
                "type": "integer",
                "description": "Optional lower publication-year bound.",
            },
            "year_to": {
                "type": "integer",
                "description": "Optional upper publication-year bound.",
            },
            "oa_only": {
                "type": "boolean",
                "default": False,
                "description": "Return only open-access papers.",
            },
            "require_pdf": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Whether the workflow expects downloadable PDFs. When true, papers "
                    "without PDFs are still returned but marked as manual-upload-required."
                ),
            },
            "missing_pdf_policy": {
                "type": "string",
                "enum": ["include_and_mark", "exclude", "include_metadata_only"],
                "default": "include_and_mark",
                "description": "How to handle papers whose free OA PDF cannot be found.",
            },
            "with_pdf_url": {
                "type": "boolean",
                "default": True,
                "description": "Prioritize papers with accessible PDF URLs.",
            },
            "email": {
                "type": "string",
                "description": (
                    "Optional email for polite API usage, especially Unpaywall and Crossref. "
                    "Prefer environment variables in production."
                ),
            },
        },
        "required": ["query"],
        "additionalProperties": True,
    },
}

PAPER_PDF_READER_TOOL_SCHEMA = {
    "name": "paper_pdf_reader_tool",
    "description": (
        "Read an academic paper PDF from URL or local path. If the PDF source is missing "
        "or inaccessible, it returns a clear missing status instead of failing silently."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pdf_source": {
                "type": "string",
                "description": "Remote PDF URL, local PDF path, or internal file ID.",
            },
            "source_type": {
                "type": "string",
                "enum": ["auto", "url", "local_path", "file_id"],
                "default": "auto",
                "description": "Type of pdf_source.",
            },
            "paper_metadata": {
                "type": "object",
                "description": "Optional metadata from paper_search_tool for clearer missing-PDF reports.",
                "additionalProperties": True,
            },
            "read_mode": {
                "type": "string",
                "enum": ["full_text", "page_text", "metadata_only"],
                "default": "full_text",
                "description": "Whether to return merged text, page-level text, or only metadata.",
            },
            "page_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                },
                "description": "Optional 1-based inclusive page range.",
            },
            "download": {
                "type": "boolean",
                "default": True,
                "description": "Download and save a remote PDF before reading it.",
            },
            "save_dir": {
                "type": "string",
                "default": "data/papers/raw",
                "description": "Directory for downloaded PDFs.",
            },
            "extract_images": {
                "type": "boolean",
                "default": False,
                "description": "Reserved flag for future image extraction.",
            },
            "extract_tables": {
                "type": "boolean",
                "default": False,
                "description": "Reserved flag for future table extraction.",
            },
            "max_pages": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "maximum": 500,
                "description": "Maximum pages to read.",
            },
            "on_missing_source": {
                "type": "string",
                "enum": ["return_missing_status", "raise_error"],
                "default": "return_missing_status",
                "description": "How to handle missing PDF source.",
            },
        },
        "required": ["pdf_source"],
        "additionalProperties": True,
    },
}

PAPER_PARSER_TOOL_SCHEMA = {
    "name": "paper_parser_tool",
    "description": (
        "Parse an academic paper into title/authors/abstract/sections/references and "
        "RAG-ready chunks. The default parser_backend is auto. If content is missing, "
        "the tool returns manual-upload-required status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input_source": {
                "type": "string",
                "description": "Local PDF path, PDF URL, raw text, or internal document ID.",
            },
            "input_type": {
                "type": "string",
                "enum": ["auto", "pdf_url", "local_pdf", "raw_text", "document_id"],
                "default": "auto",
                "description": "Type of input_source.",
            },
            "paper_metadata": {
                "type": "object",
                "description": "Optional paper metadata used in output and missing-content reports.",
                "additionalProperties": True,
            },
            "parser_backend": {
                "type": "string",
                "enum": ["auto", "grobid", "unstructured", "pymupdf", "lightweight"],
                "default": "auto",
                "description": "Parser backend. auto chooses the best available local parser.",
            },
            "parse_level": {
                "type": "string",
                "enum": ["metadata", "sections", "references", "full", "rag_chunks"],
                "default": "full",
                "description": "Parsing depth.",
            },
            "chunk_strategy": {
                "type": "string",
                "enum": ["none", "by_section", "by_paragraph", "fixed_size", "semantic"],
                "default": "by_section",
                "description": "Chunking strategy for RAG preprocessing.",
            },
            "chunk_size": {
                "type": "integer",
                "default": 800,
                "minimum": 100,
                "maximum": 4000,
                "description": "Target chunk size.",
            },
            "chunk_overlap": {
                "type": "integer",
                "default": 100,
                "minimum": 0,
                "maximum": 1000,
                "description": "Overlap between adjacent chunks.",
            },
            "include_references": {
                "type": "boolean",
                "default": True,
                "description": "Include parsed references when possible.",
            },
            "include_figures": {
                "type": "boolean",
                "default": True,
                "description": "Include figure captions when possible.",
            },
            "include_tables": {
                "type": "boolean",
                "default": True,
                "description": "Include table captions when possible.",
            },
            "normalize_text": {
                "type": "boolean",
                "default": True,
                "description": "Clean and normalize extracted text.",
            },
            "output_format": {
                "type": "string",
                "enum": ["json", "markdown", "rag_documents"],
                "default": "json",
                "description": "Structured JSON, readable Markdown, or vector-DB-ready documents.",
            },
            "on_missing_input": {
                "type": "string",
                "enum": ["return_missing_status", "raise_error"],
                "default": "return_missing_status",
                "description": "How to handle missing PDF/text input.",
            },
        },
        "required": ["input_source"],
        "additionalProperties": True,
    },
}

PDF_ACCESS_STATUS = ["available", "missing", "restricted", "broken_url", "unknown"]
DOCUMENT_PROCESSING_STATUS = [
    "search_only",
    "pdf_downloaded",
    "pdf_missing",
    "pdf_read",
    "parsed",
    "chunked",
    "ingested",
    "failed",
]
