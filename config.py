import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import dotenv_values, load_dotenv


_PROJECT_DIR = Path(__file__).resolve().parent
_ENV_FILES = (
    _PROJECT_DIR.parent / ".env",
    _PROJECT_DIR / ".env",
)


def load_environment(override: bool = False) -> None:
    """Load .env files from the chapter directory and its parent project root."""
    for path in _ENV_FILES:
        if path.exists():
            load_dotenv(dotenv_path=path, override=override)


def env_signature() -> tuple[tuple[str, float], ...]:
    """Return a lightweight signature for detecting .env edits in long-running API servers."""
    signature = []
    for path in _ENV_FILES:
        if path.exists():
            signature.append((str(path), path.stat().st_mtime))
    return tuple(signature)


load_environment()


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _dotenv_value(name: str) -> Optional[str]:
    merged: Dict[str, Optional[str]] = {}
    for path in _ENV_FILES:
        if path.exists():
            merged.update(dotenv_values(path))
    return _clean_env_value(merged.get(name))


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = _clean_env_value(os.getenv(name))
    if value is not None:
        return value
    value = _dotenv_value(name)
    return value if value is not None else default


def _get_int(name: str, default: int) -> int:
    value = _get_env(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_first_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = _get_env(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def _get_float(name: str, default: float) -> float:
    value = _get_env(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool = False) -> bool:
    value = _get_env(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_json_dict(name: str) -> Dict[str, Any]:
    value = _get_env(name)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class AppConfig:
    python_version: str = "3.11"

    # LLM / vision model used by MarkItDown when parsing images.
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    llm_reasoning_effort: Optional[str] = None
    llm_thinking_enabled: bool = False
    llm_extra_body: Optional[Dict[str, Any]] = None

    # Embedding model used for RAG chunks and future query vectors.
    embed_model_type: str = "local"
    embed_api_key: Optional[str] = None
    embed_base_url: Optional[str] = None
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_vector_size: int = 384
    embed_max_try: int = 3
    embed_retry_wait_seconds: float = 2.0
    embed_retry_backoff: float = 2.0

    # Qdrant connection.
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_timeout: int = 30

    # Neo4j connection for graph-enhanced semantic memory.
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = None

    # Collection names.
    rag_collection_name: str = "semantic_knowledge_base"
    semantic_collection_name: str = "semantic_memory_collection"
    episodic_collection_name: str = "episodic_collection"
    sensory_collection_name: str = "sensory_collection"
    sensory_model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    sensory_model_enabled: bool = False
    sensory_model_local_path: Optional[str] = None
    sensory_model_local_files_only: bool = True
    hf_token: Optional[str] = None

    # Local stores and memory defaults.
    episodic_db_path: str = "episodic_memory.db"
    working_memory_max_capacity: int = 50
    working_memory_ttl_seconds: int = 3600
    memory_decay_rate: float = 0.01

    # RAG query defaults.
    rag_query_mqe_count: int = 3
    rag_query_top_k: int = 5
    rag_query_per_query_limit: int = 5
    rag_rerank_top_k: int = 5
    rag_complex_query_top_k: int = 12
    rag_complex_per_query_limit: int = 8
    rag_complex_rerank_top_k: int = 12
    rag_complex_context_top_k: int = 0
    rag_rerank_vector_weight: float = 0.6
    rag_rerank_lexical_weight: float = 0.25
    rag_rerank_diversity_weight: float = 0.1
    rag_rerank_mode_weight: float = 0.05
    rag_bm25_enabled: bool = True
    rag_bm25_index_path: str = "data/indexes/bm25_chunks.pkl"
    rag_bm25_top_k: int = 20
    rag_bm25_strong_entity_top_k: int = 30
    rag_hybrid_candidate_k: int = 40
    rag_entity_aware_enabled: bool = True
    rag_entity_required_for_strong_query: bool = True
    rag_rerank_bm25_weight: float = 0.25
    rag_rerank_entity_weight: float = 0.25
    rag_rerank_hybrid_bonus: float = 0.1
    rag_rerank_entity_missing_penalty: float = 0.3
    rag_context_top_k: int = 5
    rag_context_max_tokens: int = 6000
    rag_context_max_doc_tokens: int = 900
    qa_temperature: float = 0.2
    qa_max_answer_tokens: int = 2000
    qa_max_continuations: int = 2
    memory_summary_max_chars: int = 1200
    memory_summary_max_tokens: int = 400
    memory_semantic_fact_min_importance: int = 7

    # MCP integration. These servers are optional and run through stdio.
    mcp_enabled: bool = False
    mcp_filesystem_enabled: bool = True
    mcp_filesystem_root: str = "."
    mcp_fetch_enabled: bool = True
    mcp_fetch_ignore_robots_txt: bool = True
    mcp_sqlite_enabled: bool = True
    mcp_sqlite_db_path: str = "mcp_demo.db"
    mcp_brave_enabled: bool = False
    mcp_brave_api_key: Optional[str] = None
    mcp_timeout_seconds: float = 20.0
    mcp_max_result_chars: int = 4000
    mcp_tool_calling_enabled: bool = False
    mcp_tool_calling_max_iterations: int = 3

    # Local skills. Skills are Python modules that add domain-specific context.
    skills_enabled: bool = True
    skills_package: str = "skills"
    skills_enabled_names: str = "senet_explainer,paper_deep_summary"

    # Bounded agent loop and transcript persistence.
    agent_loop_max_steps: int = 3
    agent_loop_max_tool_calls: int = 3
    agent_loop_max_repair_attempts: int = 1
    agent_loop_context_max_tokens: int = 6000
    agent_loop_planner_enabled: bool = False
    agent_loop_online_fallback_enabled: bool = True
    agent_loop_local_relevance_threshold: float = 0.18
    agent_loop_max_memory_items: int = 12
    agent_loop_max_tool_observation_chars: int = 2000
    transcript_enabled: bool = False
    transcript_dir: str = "data/transcripts"
    transcript_resume_enabled: bool = False
    transcript_resume_max_events: int = 12
    checkpoint_enabled: bool = False
    checkpoint_db_path: str = "data/checkpoints/agent_checkpoints.sqlite3"
    agent_runtime: str = "legacy"
    session_summary_enabled: bool = True
    session_summary_max_chars: int = 1200
    session_summary_fuse_memory: bool = True
    api_auth_token: Optional[str] = None
    api_rate_limit_per_minute: int = 0
    pader_auth_mode: str = "local"
    pader_api_base_url: str = "https://www.pader.top"
    pader_api_timeout_seconds: float = 15.0
    pader_app_database_path: str = "data/app.db"

    # Local RAG uploads and background jobs.
    rag_upload_dir: str = "data/uploads"
    rag_upload_max_file_size_mb: int = 50
    rag_upload_allowed_extensions: str = ".md,.txt,.py,.pdf,.docx,.xlsx,.pptx,.html,.csv,.png,.jpg,.jpeg"

    # Optional Redis runtime. The assistant keeps local fallbacks when disabled.
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_prefix: str = "chatbot"
    redis_socket_timeout_seconds: float = 3.0
    redis_cache_ttl_seconds: int = 600
    redis_working_memory_ttl_seconds: int = 3600
    redis_task_ttl_seconds: int = 86400

    # LLM-Sentinel security middleware.
    sentinel_enabled: bool = True
    sentinel_input_guard_enabled: bool = True
    sentinel_rag_guard_enabled: bool = True
    sentinel_tool_guard_enabled: bool = True
    sentinel_output_guard_enabled: bool = True
    sentinel_block_threshold: float = 0.85
    sentinel_warn_threshold: float = 0.55
    sentinel_strip_rag_instructions: bool = True
    sentinel_llm_judge_enabled: bool = False
    sentinel_llm_judge_model: Optional[str] = None
    sentinel_llm_judge_api_key: Optional[str] = None
    sentinel_llm_judge_base_url: Optional[str] = None
    sentinel_llm_judge_max_tokens: int = 700
    sentinel_llm_judge_temperature: float = 0.0

    @property
    def embedding_api_key(self) -> Optional[str]:
        """Embedding key with OpenAI key fallback for OpenAI-compatible providers."""
        return self.embed_api_key or self.openai_api_key

    @property
    def embedding_base_url(self) -> Optional[str]:
        """Embedding base URL with OpenAI base URL fallback."""
        return self.embed_base_url or self.openai_base_url

    def chat_completion_kwargs(self) -> Dict[str, Any]:
        """Extra OpenAI-compatible chat parameters, e.g. DeepSeek thinking mode."""
        kwargs: Dict[str, Any] = {}
        if self.llm_reasoning_effort:
            kwargs["reasoning_effort"] = self.llm_reasoning_effort

        extra_body = dict(self.llm_extra_body or {})
        if self.llm_thinking_enabled:
            extra_body.setdefault("thinking", {"type": "enabled"})
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    load_environment()
    openai_base_url = _get_env("OPENAI_BASE_URL")
    is_deepseek = "deepseek" in (openai_base_url or "").lower()
    return AppConfig(
        openai_api_key=_get_env("OPENAI_API_KEY"),
        openai_base_url=openai_base_url,
        model_name=_get_env("MODEL_NAME", "gpt-4o-mini") or "gpt-4o-mini",
        llm_reasoning_effort=_get_env("LLM_REASONING_EFFORT") or _get_env("REASONING_EFFORT") or ("high" if is_deepseek else None),
        llm_thinking_enabled=_get_bool("LLM_THINKING_ENABLED", _get_bool("DEEPSEEK_THINKING_ENABLED", is_deepseek)),
        llm_extra_body=_get_json_dict("LLM_EXTRA_BODY"),
        embed_model_type=(_get_env("EMBED_MODEL_TYPE", "local") or "local").lower(),
        embed_api_key=_get_env("EMBED_API_KEY"),
        embed_base_url=_get_env("EMBED_BASE_URL"),
        embed_model_name=_get_env("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2",
        embed_vector_size=_get_first_int(("EMBED_VECTOR_SIZE", "QDRANT_VECTOR_SIZE"), 384),
        embed_max_try=_get_int("EMBED_MAX_TRY", 3),
        embed_retry_wait_seconds=_get_float("EMBED_RETRY_WAIT_SECONDS", 2.0),
        embed_retry_backoff=_get_float("EMBED_RETRY_BACKOFF", 2.0),
        qdrant_url=_get_env("QDRANT_URL"),
        qdrant_api_key=_get_env("QDRANT_API_KEY"),
        qdrant_timeout=_get_int("QDRANT_TIMEOUT", 30),
        neo4j_uri=_get_env("NEO4J_URI"),
        neo4j_username=_get_env("NEO4J_USERNAME") or _get_env("NEO4J_USER"),
        neo4j_password=_get_env("NEO4J_PASSWORD"),
        neo4j_database=_get_env("NEO4J_DATABASE"),
        rag_collection_name=_get_env("RAG_COLLECTION_NAME", _get_env("QDRANT_COLLECTION", "semantic_knowledge_base")) or "semantic_knowledge_base",
        semantic_collection_name=_get_env("SEMANTIC_COLLECTION_NAME", "semantic_memory_collection") or "semantic_memory_collection",
        episodic_collection_name=_get_env("EPISODIC_COLLECTION_NAME", "episodic_collection") or "episodic_collection",
        sensory_collection_name=_get_env("SENSORY_COLLECTION_NAME", "sensory_collection") or "sensory_collection",
        sensory_model_name=_get_env("SENSORY_MODEL_NAME", "OFA-Sys/chinese-clip-vit-base-patch16") or "OFA-Sys/chinese-clip-vit-base-patch16",
        sensory_model_enabled=_get_bool("SENSORY_MODEL_ENABLED", False),
        sensory_model_local_path=_get_env("SENSORY_MODEL_LOCAL_PATH"),
        sensory_model_local_files_only=_get_bool("SENSORY_MODEL_LOCAL_FILES_ONLY", True),
        hf_token=_get_env("HF_TOKEN") or _get_env("HUGGINGFACE_HUB_TOKEN"),
        episodic_db_path=_get_env("EPISODIC_DB_PATH", "episodic_memory.db") or "episodic_memory.db",
        working_memory_max_capacity=_get_int("WORKING_MEMORY_MAX_CAPACITY", 50),
        working_memory_ttl_seconds=_get_int("WORKING_MEMORY_TTL_SECONDS", 3600),
        memory_decay_rate=_get_float("MEMORY_DECAY_RATE", 0.01),
        rag_query_mqe_count=_get_int("RAG_QUERY_MQE_COUNT", 3),
        rag_query_top_k=_get_int("RAG_QUERY_TOP_K", 5),
        rag_query_per_query_limit=_get_int("RAG_QUERY_PER_QUERY_LIMIT", 5),
        rag_rerank_top_k=_get_int("RAG_RERANK_TOP_K", 5),
        rag_complex_query_top_k=_get_int("RAG_COMPLEX_QUERY_TOP_K", 12),
        rag_complex_per_query_limit=_get_int("RAG_COMPLEX_PER_QUERY_LIMIT", 8),
        rag_complex_rerank_top_k=_get_int("RAG_COMPLEX_RERANK_TOP_K", 12),
        rag_complex_context_top_k=_get_int("RAG_COMPLEX_CONTEXT_TOP_K", 0),
        rag_rerank_vector_weight=_get_float("RAG_RERANK_VECTOR_WEIGHT", 0.6),
        rag_rerank_lexical_weight=_get_float("RAG_RERANK_LEXICAL_WEIGHT", 0.25),
        rag_rerank_diversity_weight=_get_float("RAG_RERANK_DIVERSITY_WEIGHT", 0.1),
        rag_rerank_mode_weight=_get_float("RAG_RERANK_MODE_WEIGHT", 0.05),
        rag_bm25_enabled=_get_bool("RAG_BM25_ENABLED", True),
        rag_bm25_index_path=_get_env("RAG_BM25_INDEX_PATH", "data/indexes/bm25_chunks.pkl") or "data/indexes/bm25_chunks.pkl",
        rag_bm25_top_k=_get_int("RAG_BM25_TOP_K", 20),
        rag_bm25_strong_entity_top_k=_get_int("RAG_BM25_STRONG_ENTITY_TOP_K", 30),
        rag_hybrid_candidate_k=_get_int("RAG_HYBRID_CANDIDATE_K", 40),
        rag_entity_aware_enabled=_get_bool("RAG_ENTITY_AWARE_ENABLED", True),
        rag_entity_required_for_strong_query=_get_bool("RAG_ENTITY_REQUIRED_FOR_STRONG_QUERY", True),
        rag_rerank_bm25_weight=_get_float("RAG_RERANK_BM25_WEIGHT", 0.25),
        rag_rerank_entity_weight=_get_float("RAG_RERANK_ENTITY_WEIGHT", 0.25),
        rag_rerank_hybrid_bonus=_get_float("RAG_RERANK_HYBRID_BONUS", 0.1),
        rag_rerank_entity_missing_penalty=_get_float("RAG_RERANK_ENTITY_MISSING_PENALTY", 0.3),
        rag_context_top_k=_get_int("RAG_CONTEXT_TOP_K", 5),
        rag_context_max_tokens=_get_int("RAG_CONTEXT_MAX_TOKENS", 6000),
        rag_context_max_doc_tokens=_get_int("RAG_CONTEXT_MAX_DOC_TOKENS", 900),
        qa_temperature=_get_float("QA_TEMPERATURE", 0.2),
        qa_max_answer_tokens=_get_int("QA_MAX_ANSWER_TOKENS", 2000),
        qa_max_continuations=_get_int("QA_MAX_CONTINUATIONS", 2),
        memory_summary_max_chars=_get_int("MEMORY_SUMMARY_MAX_CHARS", 1200),
        memory_summary_max_tokens=_get_int("MEMORY_SUMMARY_MAX_TOKENS", 400),
        memory_semantic_fact_min_importance=_get_int("MEMORY_SEMANTIC_FACT_MIN_IMPORTANCE", 7),
        mcp_enabled=_get_bool("MCP_ENABLED", False),
        mcp_filesystem_enabled=_get_bool("MCP_FILESYSTEM_ENABLED", True),
        mcp_filesystem_root=_get_env("MCP_FILESYSTEM_ROOT", ".") or ".",
        mcp_fetch_enabled=_get_bool("MCP_FETCH_ENABLED", True),
        mcp_fetch_ignore_robots_txt=_get_bool("MCP_FETCH_IGNORE_ROBOTS_TXT", True),
        mcp_sqlite_enabled=_get_bool("MCP_SQLITE_ENABLED", True),
        mcp_sqlite_db_path=_get_env("MCP_SQLITE_DB_PATH", "mcp_demo.db") or "mcp_demo.db",
        mcp_brave_enabled=_get_bool("MCP_BRAVE_ENABLED", False),
        mcp_brave_api_key=_get_env("BRAVE_API_KEY") or _get_env("MCP_BRAVE_API_KEY"),
        mcp_timeout_seconds=_get_float("MCP_TIMEOUT_SECONDS", 20.0),
        mcp_max_result_chars=_get_int("MCP_MAX_RESULT_CHARS", 4000),
        mcp_tool_calling_enabled=_get_bool("MCP_TOOL_CALLING_ENABLED", False),
        mcp_tool_calling_max_iterations=_get_int("MCP_TOOL_CALLING_MAX_ITERATIONS", 3),
        skills_enabled=_get_bool("SKILLS_ENABLED", True),
        skills_package=_get_env("SKILLS_PACKAGE", "skills") or "skills",
        skills_enabled_names=_get_env("SKILLS_ENABLED_NAMES", "senet_explainer,paper_deep_summary") or "senet_explainer,paper_deep_summary",
        agent_loop_max_steps=_get_int("AGENT_LOOP_MAX_STEPS", 3),
        agent_loop_max_tool_calls=_get_int("AGENT_LOOP_MAX_TOOL_CALLS", 3),
        agent_loop_max_repair_attempts=_get_int("AGENT_LOOP_MAX_REPAIR_ATTEMPTS", 1),
        agent_loop_context_max_tokens=_get_int("AGENT_LOOP_CONTEXT_MAX_TOKENS", 6000),
        agent_loop_planner_enabled=_get_bool("AGENT_LOOP_PLANNER_ENABLED", False),
        agent_loop_online_fallback_enabled=_get_bool("AGENT_LOOP_ONLINE_FALLBACK_ENABLED", True),
        agent_loop_local_relevance_threshold=_get_float("AGENT_LOOP_LOCAL_RELEVANCE_THRESHOLD", 0.18),
        agent_loop_max_memory_items=_get_int("AGENT_LOOP_MAX_MEMORY_ITEMS", 12),
        agent_loop_max_tool_observation_chars=_get_int("AGENT_LOOP_MAX_TOOL_OBSERVATION_CHARS", 2000),
        transcript_enabled=_get_bool("TRANSCRIPT_ENABLED", False),
        transcript_dir=_get_env("TRANSCRIPT_DIR", "data/transcripts") or "data/transcripts",
        transcript_resume_enabled=_get_bool("TRANSCRIPT_RESUME_ENABLED", False),
        transcript_resume_max_events=_get_int("TRANSCRIPT_RESUME_MAX_EVENTS", 12),
        checkpoint_enabled=_get_bool("CHECKPOINT_ENABLED", False),
        checkpoint_db_path=_get_env("CHECKPOINT_DB_PATH", "data/checkpoints/agent_checkpoints.sqlite3") or "data/checkpoints/agent_checkpoints.sqlite3",
        agent_runtime=(_get_env("AGENT_RUNTIME", "legacy") or "legacy").lower(),
        session_summary_enabled=_get_bool("SESSION_SUMMARY_ENABLED", True),
        session_summary_max_chars=_get_int("SESSION_SUMMARY_MAX_CHARS", 1200),
        session_summary_fuse_memory=_get_bool("SESSION_SUMMARY_FUSE_MEMORY", True),
        api_auth_token=_get_env("API_AUTH_TOKEN"),
        api_rate_limit_per_minute=_get_int("API_RATE_LIMIT_PER_MINUTE", 0),
        pader_auth_mode=(_get_env("PADER_AUTH_MODE", "local") or "local").lower(),
        pader_api_base_url=_get_env("PADER_API_BASE_URL", "https://www.pader.top") or "https://www.pader.top",
        pader_api_timeout_seconds=_get_float("PADER_API_TIMEOUT_SECONDS", 15.0),
        pader_app_database_path=_get_env(
            "PADER_APP_DATABASE_PATH",
            "data/app.db",
        ) or "data/app.db",
        rag_upload_dir=_get_env("RAG_UPLOAD_DIR", "data/uploads") or "data/uploads",
        rag_upload_max_file_size_mb=_get_int("RAG_UPLOAD_MAX_FILE_SIZE_MB", 50),
        rag_upload_allowed_extensions=_get_env(
            "RAG_UPLOAD_ALLOWED_EXTENSIONS",
            ".md,.txt,.py,.pdf,.docx,.xlsx,.pptx,.html,.csv,.png,.jpg,.jpeg",
        ) or ".md,.txt,.py,.pdf,.docx,.xlsx,.pptx,.html,.csv,.png,.jpg,.jpeg",
        redis_enabled=_get_bool("REDIS_ENABLED", False),
        redis_url=_get_env("REDIS_URL", "redis://localhost:6379/0") or "redis://localhost:6379/0",
        redis_prefix=_get_env("REDIS_PREFIX", "chatbot") or "chatbot",
        redis_socket_timeout_seconds=_get_float("REDIS_SOCKET_TIMEOUT_SECONDS", 3.0),
        redis_cache_ttl_seconds=_get_int("REDIS_CACHE_TTL_SECONDS", 600),
        redis_working_memory_ttl_seconds=_get_int("REDIS_WORKING_MEMORY_TTL_SECONDS", 3600),
        redis_task_ttl_seconds=_get_int("REDIS_TASK_TTL_SECONDS", 86400),
        sentinel_enabled=_get_bool("SENTINEL_ENABLED", True),
        sentinel_input_guard_enabled=_get_bool("SENTINEL_INPUT_GUARD_ENABLED", True),
        sentinel_rag_guard_enabled=_get_bool("SENTINEL_RAG_GUARD_ENABLED", True),
        sentinel_tool_guard_enabled=_get_bool("SENTINEL_TOOL_GUARD_ENABLED", True),
        sentinel_output_guard_enabled=_get_bool("SENTINEL_OUTPUT_GUARD_ENABLED", True),
        sentinel_block_threshold=_get_float("SENTINEL_BLOCK_THRESHOLD", 0.85),
        sentinel_warn_threshold=_get_float("SENTINEL_WARN_THRESHOLD", 0.55),
        sentinel_strip_rag_instructions=_get_bool("SENTINEL_STRIP_RAG_INSTRUCTIONS", True),
        sentinel_llm_judge_enabled=_get_bool("SENTINEL_LLM_JUDGE_ENABLED", False),
        sentinel_llm_judge_model=_get_env("SENTINEL_LLM_JUDGE_MODEL"),
        sentinel_llm_judge_api_key=_get_env("SENTINEL_LLM_JUDGE_API_KEY"),
        sentinel_llm_judge_base_url=_get_env("SENTINEL_LLM_JUDGE_BASE_URL"),
        sentinel_llm_judge_max_tokens=_get_int("SENTINEL_LLM_JUDGE_MAX_TOKENS", 700),
        sentinel_llm_judge_temperature=_get_float("SENTINEL_LLM_JUDGE_TEMPERATURE", 0.0),
    )


def reload_config_from_env(override: bool = True) -> AppConfig:
    """Reload .env files and clear the cached AppConfig."""
    load_environment(override=override)
    get_config.cache_clear()
    return get_config()
