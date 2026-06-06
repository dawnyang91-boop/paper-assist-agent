from types import SimpleNamespace

from app.main import (
    format_agent_trace,
    format_latency_benchmark,
    format_reference_sources,
    parse_ingest_command,
    resolve_ingest_file_paths,
    summarize_latency,
)
from rag.context_builder import ContextDocument


def test_format_reference_sources_lists_document_metadata():
    result = SimpleNamespace(
        built_context=SimpleNamespace(
            documents=[
                ContextDocument(
                    doc_id="D1",
                    content="SENet 使用 SE block 建模通道依赖。",
                    score=0.8123,
                    source_file="test_files/SENet.md",
                    chunk_index=3,
                    heading_paths=["SENet", "核心贡献"],
                ),
                ContextDocument(
                    doc_id="D2",
                    content="SE block 包含 squeeze 和 excitation。",
                    score=0.7012,
                    source_file="test_files/SENet.md",
                    chunk_index=4,
                ),
            ]
        )
    )

    text = format_reference_sources(result)

    assert "引用来源：" in text
    assert "[D1] test_files/SENet.md | chunk 3 | score 0.8123" in text
    assert "headings: SENet / 核心贡献" in text
    assert "[D2] test_files/SENet.md | chunk 4 | score 0.7012" in text


def test_format_reference_sources_handles_empty_documents():
    result = SimpleNamespace(built_context=SimpleNamespace(documents=[]))

    text = format_reference_sources(result)

    assert "引用来源：" in text
    assert "无可用检索文档" in text


def test_format_agent_trace_lists_decisions_and_verification():
    result = SimpleNamespace(
        metadata={
            "stop_reason": "final",
            "loop_steps": 2,
            "context_tokens": 123,
            "resumed_transcript_count": 1,
            "decisions": [{"action": "build_context", "reason": "ready"}],
            "tool_observations": [{
                "role": "mcp:fetch.fetch",
                "metadata": {"server": "fetch", "tool": "fetch", "success": True},
            }],
            "verification": {"status": "ok", "passed": True, "citation_coverage": 1.0},
        }
    )

    text = format_agent_trace(result)

    assert "Agent Trace：" in text
    assert "build_context" in text
    assert "fetch.fetch success=True" in text
    assert "coverage=1.0" in text


def test_summarize_latency_returns_stable_statistics():
    summary = summarize_latency([10.0, 20.0, 30.0])

    assert summary["count"] == 3
    assert summary["avg_ms"] == 20.0
    assert summary["p50_ms"] == 20.0
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 30.0


def test_format_latency_benchmark_renders_runtime_comparison():
    text = format_latency_benchmark([
        {
            "runtime": "legacy",
            "summary": {"count": 2, "avg_ms": 100.0, "p50_ms": 100.0, "min_ms": 90.0, "max_ms": 110.0},
            "document_count": 2,
            "memory_count": 1,
            "dag_nodes": 0,
            "dag_edges": 0,
        },
        {
            "runtime": "dag",
            "summary": {"count": 2, "avg_ms": 80.0, "p50_ms": 80.0, "min_ms": 70.0, "max_ms": 90.0},
            "document_count": 2,
            "memory_count": 1,
            "dag_nodes": 5,
            "dag_edges": 4,
        },
    ])

    assert "Legacy vs DAG latency benchmark" in text
    assert "| legacy | 2 | 100.0" in text
    assert "| dag | 2 | 80.0" in text
    assert "legacy_avg / dag_avg = 1.25" in text


def test_parse_ingest_command_supports_default_and_custom_directory():
    assert parse_ingest_command("/ingest") == "./test_files"
    assert parse_ingest_command("/ingest ./papers") == "./papers"
    assert parse_ingest_command("/ingest    ") == "./test_files"
    assert parse_ingest_command("请总结 SENet") is None


def test_resolve_ingest_file_paths_supports_single_file_and_directory(tmp_path):
    single_file = tmp_path / "paper.pdf"
    single_file.write_text("paper", encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    nested_file = docs_dir / "note.md"
    nested_file.write_text("note", encoding="utf-8")

    assert resolve_ingest_file_paths(str(single_file), lambda path: []) == [str(single_file)]
    assert resolve_ingest_file_paths(str(docs_dir), lambda path: [str(nested_file)]) == [str(nested_file)]
    assert resolve_ingest_file_paths(str(tmp_path / "missing"), lambda path: []) == []
