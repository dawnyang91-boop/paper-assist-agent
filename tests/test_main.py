from types import SimpleNamespace

from context_builder import ContextDocument
from main import format_agent_trace, format_reference_sources


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
