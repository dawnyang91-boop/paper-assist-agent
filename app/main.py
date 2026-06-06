import argparse
import os
import statistics
import sys
import time
from dataclasses import replace
from typing import Optional

from config import get_config, reload_config_from_env
from agent.qa_agent import QAAgent
from memory.Episodic_Memory import EpisodicMemory
from memory.memory_manager import MemoryManager
from memory.Semantic_Memory import SemanticMemory
from memory.Sensory_Memory import SensoryMemory
from memory.Working_Memory import WorkingMemory
from rag.embedding_service import EmbeddingService
from tools.mcp_manager import MCPManager
from tools.skill_manager import SkillManager


def build_qdrant_client(config):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("请先安装依赖：pip install -r requirements.txt") from exc

    return QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout=config.qdrant_timeout,
    )


def build_neo4j_client(config):
    if not config.neo4j_uri or not config.neo4j_username or not config.neo4j_password:
        return None

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[警告] 已配置 Neo4j，但未安装 neo4j driver；图谱检索将被跳过。")
        return None

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
        connection_timeout=5,
    )
    try:
        driver.verify_connectivity()
    except Exception as exc:
        try:
            driver.close()
        except Exception:
            pass
        print(f"[警告] Neo4j 连接不可用，已跳过图谱记忆：{exc}")
        return None
    return driver


def build_assistant(load_sensory_model: bool = False, config_override=None):
    config = config_override or get_config()
    qdrant = build_qdrant_client(config)
    neo4j = build_neo4j_client(config)
    embedding_service = EmbeddingService(config=config)

    working_memory = WorkingMemory(config=config)
    episodic_memory = EpisodicMemory(qdrant_client=qdrant, config=config)
    semantic_memory = SemanticMemory(qdrant_client=qdrant, neo4j_client=neo4j, config=config)
    sensory_memory = SensoryMemory(
        qdrant_client=qdrant,
        config=config,
        load_model=load_sensory_model,
    )
    memory_manager = MemoryManager(
        working_memory=working_memory,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        sensory_memory=sensory_memory,
        embedding_service=embedding_service,
        config=config,
    )
    mcp_manager = MCPManager(config=config) if config.mcp_enabled else None
    skill_manager = SkillManager(config=config) if config.skills_enabled else None
    agent = QAAgent(
        qdrant_client=qdrant,
        config=config,
        memory_manager=memory_manager,
        mcp_manager=mcp_manager,
        skill_manager=skill_manager,
    )
    return agent, memory_manager


def summarize_latency(samples_ms: list[float]) -> dict:
    if not samples_ms:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    sorted_samples = sorted(samples_ms)
    return {
        "count": len(samples_ms),
        "avg_ms": round(sum(samples_ms) / len(samples_ms), 2),
        "p50_ms": round(statistics.median(sorted_samples), 2),
        "min_ms": round(min(sorted_samples), 2),
        "max_ms": round(max(sorted_samples), 2),
    }


def format_latency_benchmark(rows: list[dict]) -> str:
    lines = [
        "",
        "Legacy vs DAG latency benchmark:",
        "",
        "| runtime | runs | avg_ms | p50_ms | min_ms | max_ms | docs | memories | dag_nodes | dag_edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary = row.get("summary", {})
        lines.append(
            "| {runtime} | {count} | {avg_ms} | {p50_ms} | {min_ms} | {max_ms} | {docs} | {memories} | {dag_nodes} | {dag_edges} |".format(
                runtime=row.get("runtime", "unknown"),
                count=summary.get("count", 0),
                avg_ms=summary.get("avg_ms", 0.0),
                p50_ms=summary.get("p50_ms", 0.0),
                min_ms=summary.get("min_ms", 0.0),
                max_ms=summary.get("max_ms", 0.0),
                docs=row.get("document_count", 0),
                memories=row.get("memory_count", 0),
                dag_nodes=row.get("dag_nodes", 0),
                dag_edges=row.get("dag_edges", 0),
            )
        )

    by_runtime = {row.get("runtime"): row for row in rows}
    legacy = by_runtime.get("legacy")
    dag = by_runtime.get("dag")
    if legacy and dag:
        legacy_avg = float(legacy.get("summary", {}).get("avg_ms") or 0.0)
        dag_avg = float(dag.get("summary", {}).get("avg_ms") or 0.0)
        if legacy_avg > 0 and dag_avg > 0:
            speedup = round(legacy_avg / dag_avg, 3)
            delta = round(dag_avg - legacy_avg, 2)
            lines.extend([
                "",
                f"对比：dag / legacy avg delta = {delta} ms，legacy_avg / dag_avg = {speedup}。",
            ])
    return "\n".join(lines)


def benchmark_runtime(
    question: str,
    session_id: str = "benchmark",
    repeat: int = 3,
    warmup: int = 0,
    load_sensory_model: bool = False,
    include_memory_manager: bool = True,
    include_mcp: bool = True,
    include_skills: bool = True,
):
    base_config = reload_config_from_env(override=True)
    repeat = max(1, repeat)
    warmup = max(0, warmup)
    rows = []

    for runtime in ("legacy", "dag"):
        config = replace(base_config, agent_runtime=runtime)
        agent, _memory_manager = build_assistant(
            load_sensory_model=load_sensory_model,
            config_override=config,
        )
        samples = []
        last_result = None
        total_runs = warmup + repeat
        for index in range(total_runs):
            started = time.perf_counter()
            result = agent.answer(
                question,
                session_id=session_id,
                include_memory_manager=include_memory_manager,
                include_mcp=include_mcp,
                include_skills=include_skills,
                write_memory=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            last_result = result
            if index >= warmup:
                samples.append(elapsed_ms)

        metadata = getattr(last_result, "metadata", {}) if last_result is not None else {}
        dag_trace = metadata.get("dag") or {}
        rows.append({
            "runtime": runtime,
            "summary": summarize_latency(samples),
            "document_count": metadata.get("document_count", 0),
            "memory_count": metadata.get("memory_count", 0),
            "dag_nodes": len(dag_trace.get("nodes") or []),
            "dag_edges": len(dag_trace.get("edges") or []),
            "answer_chars": len(getattr(last_result, "answer", "") if last_result is not None else ""),
        })

    print(format_latency_benchmark(rows))
    return rows


def ingest_files(data_dir: str):
    config = reload_config_from_env(override=True)
    print(
        "[*] 当前 embedding 配置: "
        f"type={config.embed_model_type}, "
        f"model={config.embed_model_name}, "
        f"dim={config.embed_vector_size}"
    )
    try:
        from rag.DocumentConverter import DocumentConverter, chunk_paragraphs_by_tokens, files_in_directory, is_conversion_failure, split_paragraphs_with_headings, store_chunks_to_qdrant
    except ImportError as exc:
        raise RuntimeError("文档入库需要安装完整依赖：pip install -r requirements.txt") from exc

    converter = DocumentConverter()
    file_paths = list(files_in_directory(data_dir))
    if not file_paths:
        print(f"未找到可入库文件：{data_dir}")
        return

    for file_path in file_paths:
        if os.path.basename(file_path).startswith("."):
            continue
        print(f"\n--- 处理文件: {file_path} ---")
        markdown_content = converter.convert_to_markdown(file_path)
        if is_conversion_failure(markdown_content):
            print(f"[跳过] 文件转换失败或无有效文本，不写入向量库：{file_path}")
            print(markdown_content[:500])
            continue
        paragraphs = split_paragraphs_with_headings(markdown_content)
        chunks = chunk_paragraphs_by_tokens(paragraphs, max_tokens=2000, overlap_tokens=400)
        print(f"共划分为 {len(chunks)} 个 chunk")
        store_chunks_to_qdrant(chunks, source_file=file_path)


def parse_ingest_command(user_input: str) -> Optional[str]:
    command = (user_input or "").strip()
    if command == "/ingest":
        return "./test_files"
    if command.startswith("/ingest "):
        data_dir = command[len("/ingest "):].strip()
        return data_dir or "./test_files"
    return None


def format_reference_sources(result) -> str:
    documents = getattr(getattr(result, "built_context", None), "documents", []) or []
    lines = ["", "引用来源："]
    if not documents:
        lines.append("- 无可用检索文档")
        return "\n".join(lines)

    for document in documents:
        source = document.source_file or "unknown"
        chunk = "unknown" if document.chunk_index is None else str(document.chunk_index)
        score = f"{document.score:.4f}" if document.score is not None else "unknown"
        heading_text = ""
        if document.heading_paths:
            heading_text = f" | headings: {' / '.join(document.heading_paths)}"
        lines.append(f"[{document.doc_id}] {source} | chunk {chunk} | score {score}{heading_text}")
    return "\n".join(lines)


def print_answer_with_sources(prefix: str, result) -> None:
    print(f"{prefix}{result.answer}{format_reference_sources(result)}")


def format_agent_trace(result) -> str:
    metadata = getattr(result, "metadata", {}) or {}
    lines = ["", "Agent Trace："]
    lines.append(f"- stop_reason: {metadata.get('stop_reason', 'unknown')}")
    lines.append(f"- loop_steps: {metadata.get('loop_steps', 0)}")
    lines.append(f"- context_tokens: {metadata.get('context_tokens', 0)}")
    lines.append(f"- resumed_transcript_count: {metadata.get('resumed_transcript_count', 0)}")

    decisions = metadata.get("decisions") or []
    if decisions:
        lines.append("- decisions:")
        for index, decision in enumerate(decisions, start=1):
            lines.append(f"  {index}. {decision.get('action')} | {decision.get('reason', '')}")

    tool_observations = metadata.get("tool_observations") or []
    if tool_observations:
        lines.append("- tool_observations:")
        for index, observation in enumerate(tool_observations, start=1):
            meta = observation.get("metadata", {}) or {}
            server = meta.get("server") or observation.get("role", "mcp")
            tool = meta.get("tool") or ""
            success = meta.get("success", True)
            lines.append(f"  {index}. {server}.{tool} success={success}")

    verification = metadata.get("verification") or {}
    if verification:
        lines.append(
            "- verification: "
            f"status={verification.get('status')} "
            f"passed={verification.get('passed')} "
            f"coverage={verification.get('citation_coverage')}"
        )

    warnings = metadata.get("warnings") or []
    if warnings:
        lines.append("- warnings:")
        for warning in warnings:
            lines.append(f"  - {warning.get('node')}: {warning.get('warning')}")
    return "\n".join(lines)


def print_agent_trace(result) -> None:
    print(format_agent_trace(result))


def ask_once(question: str, session_id: str, load_sensory_model: bool = False, show_trace: bool = False):
    agent, memory_manager = build_assistant(load_sensory_model=load_sensory_model)
    result = agent.answer(question, session_id=session_id, write_memory=True)
    print_answer_with_sources("", result)
    if show_trace:
        print_agent_trace(result)


def chat(session_id: str, load_sensory_model: bool = False, show_trace: bool = False):
    agent, memory_manager = build_assistant(load_sensory_model=load_sensory_model)
    print("私域问答助手已启动。输入问题开始对话，输入 /exit 退出。")
    print("可用命令：/media <文件路径>  将图片或音频写入感知记忆")
    print("可用命令：/ingest [目录路径]  将目录内文档写入 RAG 知识库，默认 ./test_files")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("再见。")
            break
        ingest_dir = parse_ingest_command(user_input)
        if ingest_dir is not None:
            try:
                print(f"开始入库：{ingest_dir}")
                ingest_files(ingest_dir)
                print("入库任务执行完成。")
            except Exception as exc:
                print(f"入库失败：{exc}")
            continue
        if user_input.startswith("/media "):
            file_path = user_input[len("/media "):].strip()
            try:
                result = memory_manager.process_media_input(session_id=session_id, file_path=file_path)
                print(f"已写入感知记忆：{result.modality} | {result.caption}")
            except Exception as exc:
                print(f"写入感知记忆失败：{exc}")
            continue

        result = agent.answer(user_input, session_id=session_id, write_memory=True)
        print_answer_with_sources("\n助手：", result)
        if show_trace:
            print_agent_trace(result)


def parse_args(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="私域信息问答助手")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="将私域文件转换并写入 RAG 知识库")
    ingest_parser.add_argument("--data-dir", default="./test_files", help="私域文件目录，默认 ./test_files")

    ask_parser = subparsers.add_parser("ask", help="单轮提问")
    ask_parser.add_argument("question", help="用户问题")
    ask_parser.add_argument("--session-id", default="default", help="会话 ID")
    ask_parser.add_argument("--load-sensory-model", action="store_true", help="加载中文 CLIP 模型用于图片 embedding")
    ask_parser.add_argument("--show-trace", action="store_true", help="显示 Agent Loop 决策、工具调用和校验信息")

    chat_parser = subparsers.add_parser("chat", help="启动交互式对话")
    chat_parser.add_argument("--session-id", default="default", help="会话 ID")
    chat_parser.add_argument("--load-sensory-model", action="store_true", help="加载中文 CLIP 模型用于图片 embedding")
    chat_parser.add_argument("--show-trace", action="store_true", help="显示 Agent Loop 决策、工具调用和校验信息")

    benchmark_parser = subparsers.add_parser("benchmark-runtime", help="对比 legacy 与 DAG runtime 的问答延迟")
    benchmark_parser.add_argument("question", help="用于 benchmark 的问题")
    benchmark_parser.add_argument("--session-id", default="benchmark", help="会话 ID，默认 benchmark")
    benchmark_parser.add_argument("--repeat", type=int, default=3, help="每个 runtime 统计的正式运行次数，默认 3")
    benchmark_parser.add_argument("--warmup", type=int, default=0, help="每个 runtime 的预热次数，不计入统计，默认 0")
    benchmark_parser.add_argument("--load-sensory-model", action="store_true", help="加载中文 CLIP 模型用于图片 embedding")
    benchmark_parser.add_argument("--no-memory", action="store_true", help="benchmark 时跳过 MemoryManager 召回")
    benchmark_parser.add_argument("--no-mcp", action="store_true", help="benchmark 时跳过 MCP 上下文/工具")
    benchmark_parser.add_argument("--no-skills", action="store_true", help="benchmark 时跳过 Skills")

    return parser.parse_args(argv)


def main(argv: Optional[list] = None):
    args = parse_args(argv)
    if args.command == "ingest":
        ingest_files(args.data_dir)
        return
    if args.command == "ask":
        ask_once(
            args.question,
            session_id=args.session_id,
            load_sensory_model=args.load_sensory_model,
            show_trace=args.show_trace,
        )
        return
    if args.command == "chat":
        chat(session_id=args.session_id, load_sensory_model=args.load_sensory_model, show_trace=args.show_trace)
        return
    if args.command == "benchmark-runtime":
        benchmark_runtime(
            args.question,
            session_id=args.session_id,
            repeat=args.repeat,
            warmup=args.warmup,
            load_sensory_model=args.load_sensory_model,
            include_memory_manager=not args.no_memory,
            include_mcp=not args.no_mcp,
            include_skills=not args.no_skills,
        )
        return

    print("请指定命令：ingest、ask、chat 或 benchmark-runtime。示例：python -m app.main chat")


if __name__ == "__main__":
    main(sys.argv[1:])
