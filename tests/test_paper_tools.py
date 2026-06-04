import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.paper_tools import get_paper_tool, list_paper_tools
from tools.paper_tools import search_tool as search_module


def test_paper_tool_registry_exposes_all_tools():
    schemas = list_paper_tools()
    names = {schema["name"] for schema in schemas}

    assert names == {
        "paper_search_tool",
        "paper_pdf_reader_tool",
        "paper_parser_tool",
        "paper_knowledge_graph_tool",
        "paper_summary_tool",
    }
    assert all(schema["parameters"]["type"] == "object" for schema in schemas)
    assert callable(get_paper_tool("paper_search_tool"))
    assert callable(get_paper_tool("paper_knowledge_graph_tool"))
    assert callable(get_paper_tool("paper_summary_tool"))


def test_paper_search_marks_missing_pdf_without_network(monkeypatch):
    def fake_crossref_search(query, query_type, limit, email):
        return [
            {
                "paper_id": "crossref:10.1234/example",
                "title": "Closed Access Example Paper",
                "authors": ["Author A"],
                "abstract": "Example abstract.",
                "doi": "10.1234/example",
                "venue": "Example Journal",
                "year": 2024,
                "publisher": "Example Publisher",
                "citation_count": 0,
                "open_access": {"is_oa": False, "oa_status": "closed", "source": "crossref"},
                "pdf_url": None,
                "landing_page_url": "https://publisher.example/paper",
                "source": "crossref",
                "processing_status": "search_only",
            }
        ]

    monkeypatch.setattr(search_module, "_search_crossref", fake_crossref_search)

    result = search_module.paper_search_tool(
        query="Closed Access Example Paper",
        sources=["crossref"],
        require_pdf=True,
        missing_pdf_policy="include_and_mark",
    )

    assert result["success"] is True
    assert result["data"]["manual_upload_required"] is True
    assert result["data"]["missing_pdf_count"] == 1
    paper = result["data"]["papers"][0]
    assert paper["pdf_access"]["status"] == "restricted"
    assert paper["pdf_url"] is None
    assert "Manual upload" in paper["pdf_access"]["message"] or "downloadable PDF" in paper["pdf_access"]["message"]


def test_pdf_reader_returns_missing_status_for_empty_source():
    reader = get_paper_tool("paper_pdf_reader_tool")

    result = reader(
        pdf_source="",
        paper_metadata={"title": "Missing PDF Paper", "doi": "10.1234/missing"},
    )

    assert result["success"] is False
    assert result["data"]["status"] == "missing_pdf"
    assert result["data"]["manual_upload_required"] is True
    assert result["data"]["paper_metadata"]["title"] == "Missing PDF Paper"


def test_parser_parses_raw_text_into_sections_and_chunks():
    parser = get_paper_tool("paper_parser_tool")
    raw_text = """
Example Paper Title

Abstract
This paper studies a retrieval augmented generation problem.

Introduction
Existing methods have limited grounding.

Method
We propose a retrieval and generation pipeline.

Experiments
We compare against baseline models.

Conclusion
The method improves grounded QA.

References
[1] Example reference. 2024.
""".strip()

    result = parser(
        input_source=raw_text,
        input_type="raw_text",
        paper_metadata={"title": "Example Paper Title", "doi": "10.1234/parser"},
        parse_level="full",
        chunk_size=200,
        chunk_overlap=20,
    )

    assert result["success"] is True
    data = result["data"]
    assert data["status"] in {"parsed", "chunked"}
    assert data["metadata"]["title"] == "Example Paper Title"
    assert any(section["title"].lower().startswith("abstract") for section in data["sections"])
    assert data["rag_chunks"]
    first_chunk = data["rag_chunks"][0]
    assert first_chunk["metadata"]["paper_id"] == "doi:10.1234/parser"
    assert "section" in first_chunk["metadata"]


def test_summary_tool_saves_markdown_template(tmp_path):
    summary = get_paper_tool("paper_summary_tool")

    result = summary(
        paper_metadata={
            "title": "A Test Paper",
            "authors": ["Alice", "Bob"],
            "venue": "TestConf",
            "date": "2026-01-02T00:00:00",
            "doi": "10.1234/summary",
            "abstract": "This paper proposes a test method.",
        },
        content="This paper proposes a test method. It evaluates the method on benchmark data.",
        use_llm=False,
        save=True,
        save_dir=str(tmp_path),
    )

    assert result["success"] is True
    saved_path = Path(result["data"]["saved_path"])
    assert saved_path.exists()
    markdown = saved_path.read_text(encoding="utf-8")
    assert "# 📝 论文总结-详细版" in markdown
    assert "## 📜 研究核心" in markdown
    assert "### 💡 Insight & Novelty" in markdown
    assert "## 🚀 Motivation" in markdown
    assert "## 🤔 个人总结" in markdown
    assert "https://doi.org/10.1234/summary" in markdown


def test_knowledge_graph_tool_saves_mermaid_markdown(tmp_path):
    graph = get_paper_tool("paper_knowledge_graph_tool")

    result = graph(
        paper_metadata={
            "title": "Graph Paper",
            "authors": ["Alice"],
            "venue": "GraphConf",
            "year": 2026,
            "doi": "10.1234/graph",
            "keywords": ["RAG", "Knowledge Graph"],
        },
        sections=[
            {"title": "Introduction", "text": "Problem statement."},
            {"title": "Method", "text": "Method details."},
        ],
        graph_direction="LR",
        save=True,
        save_dir=str(tmp_path),
    )

    assert result["success"] is True
    saved_path = Path(result["data"]["saved_path"])
    assert saved_path.exists()
    markdown = saved_path.read_text(encoding="utf-8")
    assert "```mermaid" in markdown
    assert "flowchart LR" in markdown
    assert "Graph Paper" in markdown
    assert result["data"]["node_count"] >= 5
    assert result["data"]["edge_count"] >= 4
