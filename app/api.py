import json
import hashlib
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from auth_store import AuthStore
from storage.background_tasks import get_task_store, run_background_task
from config import env_signature, get_config, reload_config_from_env
from app.main import build_assistant, ingest_files
from pader_auth_client import PaderAuthClient, PaderAuthError
from storage.upload_store import UploadStore


try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi import File, UploadFile
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - only triggered when API deps are missing.
    raise RuntimeError("请先安装 API 依赖：pip install fastapi uvicorn") from exc


app = FastAPI(title="私域问答助手 API", version="0.1.0")
LOGGER = logging.getLogger("chatbot.api")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
WEB_BASE_PATH = "/chatbot"
RATE_LIMIT_BUCKETS: Dict[str, list[float]] = {}
ASSISTANT_CACHE: Dict[bool, Tuple[Any, Any]] = {}
ASSISTANT_CACHE_LOCK = threading.Lock()
CONFIG_ENV_SIGNATURE = env_signature()
CONFIG_RUNTIME_SIGNATURE: Optional[Tuple[Any, ...]] = None
AUTH_PROVIDER: Optional[Any] = None
STATIC_DIR = WEB_DIST_DIR if WEB_DIST_DIR.exists() else WEB_DIR
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount(f"{WEB_BASE_PATH}/static", StaticFiles(directory=str(STATIC_DIR)), name="chatbot-static")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        LOGGER.info(
            "request_id=%s method=%s path=%s status=%s session_id=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            request.query_params.get("session_id", ""),
            latency_ms,
        )


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    write_memory: bool = True
    show_trace: bool = False


class IngestRequest(BaseModel):
    data_dir: str = "./test_files"


class UploadedIngestRequest(BaseModel):
    user_id: Optional[str] = None


class RenameSessionRequest(BaseModel):
    new_session_id: str


class RefreshSummaryRequest(BaseModel):
    max_events: int = 20


class RegisterRequest(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    password: str
    avatarUrl: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/")
@app.get(WEB_BASE_PATH)
@app.get(f"{WEB_BASE_PATH}/")
@app.get(f"{WEB_BASE_PATH}/login")
@app.get(f"{WEB_BASE_PATH}/register")
def web_index():
    index_path = _web_index_path()
    if not index_path.exists():
        raise HTTPException(status_code=404, detail={"code": "web_ui_missing", "message": "Web UI 文件不存在。"})
    return _web_index_response(index_path)


@app.get("/health")
@app.get(f"{WEB_BASE_PATH}/health")
def health():
    config = reload_config_from_env(override=True)
    return {
        "status": "ok",
        "config": {
            "openai_api_key_loaded": bool(config.openai_api_key),
            "openai_api_key_length": len(config.openai_api_key or ""),
            "openai_base_url": config.openai_base_url,
            "model_name": config.model_name,
            "embed_model_type": config.embed_model_type,
            "embed_model_name": config.embed_model_name,
            "embed_vector_size": config.embed_vector_size,
            "env_files": [path for path, _ in env_signature()],
            "assistant_cache_size": len(ASSISTANT_CACHE),
            "pader_auth_mode": config.pader_auth_mode,
            "pader_api_base_url": config.pader_api_base_url,
            "web_base_path": WEB_BASE_PATH,
        },
    }


@app.post(f"{WEB_BASE_PATH}/auth/register")
@app.post("/auth/register")
def register(payload: RegisterRequest):
    if not payload.email.strip() or not payload.password or not payload.name.strip():
        raise HTTPException(status_code=400, detail="Name, email and password are required")
    try:
        user = get_auth_provider().register(
            name=payload.name,
            email=payload.email,
            phone=payload.phone,
            password=payload.password,
            avatar_url=payload.avatarUrl,
        )
    except ValueError as exc:
        if str(exc) == "email_exists":
            raise HTTPException(status_code=400, detail="Email already registered") from exc
        raise
    except PaderAuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return {"user": user}


@app.post(f"{WEB_BASE_PATH}/auth/login")
@app.post("/auth/login")
def login(payload: LoginRequest):
    try:
        user = get_auth_provider().login(payload.email, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except PaderAuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return {"user": user}


@app.get(f"{WEB_BASE_PATH}/users/{{user_id}}")
@app.get("/users/{user_id}")
def get_user(user_id: str):
    try:
        user = get_auth_provider().get_user_profile(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    except PaderAuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return user


@app.on_event("shutdown")
def shutdown_event():
    clear_assistant_cache()
    _close_auth_provider()


@app.post(f"{WEB_BASE_PATH}/ask")
@app.post("/ask")
def ask(payload: AskRequest, request: Request):
    _enforce_api_policy(request)
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail={"code": "empty_question", "message": "问题不能为空。"})
    user_id = _request_user_id(request)
    try:
        agent, _ = get_cached_assistant()
        result = agent.answer(
            payload.question,
            session_id=payload.session_id,
            write_memory=payload.write_memory,
            transcript_enabled=True,
            transcript_store_override=_transcript_store(user_id=user_id),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"code": "ask_failed", "message": str(exc)}) from exc
    return _result_payload(result, show_trace=payload.show_trace)


@app.post(f"{WEB_BASE_PATH}/ask/stream")
@app.post("/ask/stream")
def ask_stream(payload: AskRequest, request: Request):
    _enforce_api_policy(request)
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail={"code": "empty_question", "message": "问题不能为空。"})
    user_id = _request_user_id(request)

    def generate():
        event_queue: queue.Queue[dict] = queue.Queue()
        done_marker = object()
        streamed_token_count = 0

        def enqueue_token(token: str) -> None:
            nonlocal streamed_token_count
            if not token:
                return
            streamed_token_count += 1
            event_queue.put({"event": "token", "content": token})

        def run_answer() -> None:
            nonlocal streamed_token_count
            try:
                agent, _ = get_cached_assistant()
                result = agent.answer(
                    payload.question,
                    session_id=payload.session_id,
                    write_memory=payload.write_memory,
                    transcript_enabled=True,
                    transcript_store_override=_transcript_store(user_id=user_id),
                    stream_callback=enqueue_token,
                )
                if streamed_token_count == 0:
                    for token in _chunk_text(result.answer):
                        event_queue.put({"event": "token", "content": token})
                event_queue.put({"event": "answer", "data": _result_payload(result, show_trace=payload.show_trace)})
                event_queue.put({"event": "done"})
            except Exception as exc:
                event_queue.put({"event": "error", "code": "ask_failed", "message": str(exc)})
            finally:
                event_queue.put(done_marker)

        worker = threading.Thread(target=run_answer, daemon=True)
        worker.start()
        yield _json_line({"event": "start", "session_id": payload.session_id})
        while True:
            event = event_queue.get()
            if event is done_marker:
                break
            yield _json_line(event)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _result_payload(result, show_trace: bool = False):
    payload = {
        "answer": result.answer,
        "sources": [
            {
                "doc_id": document.doc_id,
                "source_file": document.source_file,
                "chunk_index": document.chunk_index,
                "score": document.score,
                "heading_paths": document.heading_paths,
            }
            for document in result.built_context.documents
        ],
    }
    if show_trace:
        payload["trace"] = result.metadata
    payload["memories"] = result.metadata.get("memory_contexts", [])
    return payload


def _web_sensory_model_enabled() -> bool:
    return bool(get_config().sensory_model_enabled)


def _refresh_config_if_env_changed() -> None:
    """Reload .env and drop cached assistants built with stale runtime config."""
    global CONFIG_ENV_SIGNATURE, CONFIG_RUNTIME_SIGNATURE
    current_signature = env_signature()
    config = reload_config_from_env(override=True)
    runtime_signature = _runtime_config_signature(config)
    if current_signature == CONFIG_ENV_SIGNATURE and runtime_signature == CONFIG_RUNTIME_SIGNATURE:
        return
    clear_assistant_cache()
    CONFIG_ENV_SIGNATURE = current_signature
    CONFIG_RUNTIME_SIGNATURE = runtime_signature


def _runtime_config_signature(config) -> Tuple[Any, ...]:
    return (
        bool(getattr(config, "openai_api_key", None)),
        _secret_digest(getattr(config, "openai_api_key", None)),
        getattr(config, "openai_base_url", None),
        getattr(config, "model_name", None),
        repr(config.chat_completion_kwargs()),
        getattr(config, "qdrant_url", None),
        getattr(config, "rag_collection_name", None),
        getattr(config, "embed_model_type", None),
        getattr(config, "embed_model_name", None),
        getattr(config, "embed_vector_size", None),
        getattr(config, "sensory_model_enabled", None),
        getattr(config, "mcp_enabled", None),
        getattr(config, "mcp_tool_calling_enabled", None),
        getattr(config, "redis_enabled", None),
        getattr(config, "redis_url", None),
        getattr(config, "redis_prefix", None),
        getattr(config, "rag_upload_dir", None),
        getattr(config, "sentinel_enabled", None),
        getattr(config, "sentinel_input_guard_enabled", None),
        getattr(config, "sentinel_rag_guard_enabled", None),
        getattr(config, "sentinel_tool_guard_enabled", None),
        getattr(config, "sentinel_output_guard_enabled", None),
        getattr(config, "sentinel_block_threshold", None),
        getattr(config, "sentinel_warn_threshold", None),
        getattr(config, "sentinel_llm_judge_enabled", None),
        getattr(config, "sentinel_llm_judge_model", None),
        _secret_digest(getattr(config, "sentinel_llm_judge_api_key", None)),
        getattr(config, "sentinel_llm_judge_base_url", None),
    )


def _secret_digest(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_cached_assistant() -> Tuple[Any, Any]:
    _refresh_config_if_env_changed()
    load_sensory_model = _web_sensory_model_enabled()
    with ASSISTANT_CACHE_LOCK:
        cached = ASSISTANT_CACHE.get(load_sensory_model)
        if cached is None:
            cached = build_assistant(load_sensory_model=load_sensory_model)
            ASSISTANT_CACHE[load_sensory_model] = cached
        return cached


def clear_assistant_cache() -> None:
    with ASSISTANT_CACHE_LOCK:
        for agent, _ in ASSISTANT_CACHE.values():
            mcp_manager = getattr(agent, "mcp_manager", None)
            if mcp_manager is not None and hasattr(mcp_manager, "close"):
                try:
                    mcp_manager.close()
                except Exception:
                    pass
        ASSISTANT_CACHE.clear()


def get_auth_provider():
    global AUTH_PROVIDER
    config = get_config()
    mode = (config.pader_auth_mode or "remote").lower()
    if mode == "local":
        if (
            not isinstance(AUTH_PROVIDER, AuthStore)
            or AUTH_PROVIDER.path.as_posix() != config.pader_app_database_path
        ):
            _close_auth_provider()
            AUTH_PROVIDER = AuthStore(config.pader_app_database_path)
        return AUTH_PROVIDER

    if (
        not isinstance(AUTH_PROVIDER, PaderAuthClient)
        or AUTH_PROVIDER.base_url != config.pader_api_base_url.rstrip("/")
        or AUTH_PROVIDER.timeout_seconds != config.pader_api_timeout_seconds
    ):
        _close_auth_provider()
        AUTH_PROVIDER = PaderAuthClient(
            base_url=config.pader_api_base_url,
            timeout_seconds=config.pader_api_timeout_seconds,
        )
    return AUTH_PROVIDER


def _close_auth_provider() -> None:
    global AUTH_PROVIDER
    close = getattr(AUTH_PROVIDER, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    AUTH_PROVIDER = None


@app.post(f"{WEB_BASE_PATH}/ingest")
@app.post("/ingest")
def ingest(payload: IngestRequest, request: Request):
    _enforce_api_policy(request)
    try:
        ingest_files(payload.data_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"code": "ingest_failed", "message": str(exc)}) from exc
    return {"status": "ok", "data_dir": payload.data_dir}


@app.post(f"{WEB_BASE_PATH}/rag/uploads")
@app.post("/rag/uploads")
def upload_rag_files(request: Request, files: list[UploadFile] = File(...)):
    _enforce_api_policy(request)
    user_id = _request_user_id(request)
    try:
        saved = UploadStore().replace_files(files, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "upload_invalid", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"code": "upload_failed", "message": str(exc)}) from exc
    return {"status": "ok", "files": saved}


@app.get(f"{WEB_BASE_PATH}/rag/uploads")
@app.get("/rag/uploads")
def list_rag_uploads(request: Request):
    _enforce_api_policy(request)
    user_id = _request_user_id(request)
    return {"files": UploadStore().list_files(user_id=user_id)}


@app.delete(f"{WEB_BASE_PATH}/rag/uploads/{{file_id}}")
@app.delete("/rag/uploads/{file_id}")
def delete_rag_upload(file_id: str, request: Request):
    _enforce_api_policy(request)
    user_id = _request_user_id(request)
    deleted = UploadStore().delete_file(file_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "upload_not_found", "message": "文件不存在。"})
    return {"status": "ok", "deleted": file_id}


@app.post(f"{WEB_BASE_PATH}/rag/uploads/ingest")
@app.post("/rag/uploads/ingest")
def ingest_uploaded_rag_files(payload: UploadedIngestRequest, request: Request):
    _enforce_api_policy(request)
    user_id = payload.user_id or _request_user_id(request)
    upload_store = UploadStore()
    upload_dir = upload_store.upload_dir(user_id=user_id)
    store = get_task_store()
    task_id = store.create_task(
        "rag_upload_ingest",
        user_id=user_id,
        input_data={"upload_dir": upload_dir},
    )

    def work():
        reload_config_from_env(override=True)
        ingest_files(upload_dir)
        upload_store.clear_files(user_id=user_id)
        clear_assistant_cache()
        return {"upload_dir": upload_dir}

    run_background_task(task_id, work, task_store=store)
    return {"status": "accepted", "task_id": task_id, "upload_dir": upload_dir}


@app.get(f"{WEB_BASE_PATH}/tasks/{{task_id}}")
@app.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    _enforce_api_policy(request)
    task = get_task_store().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": "task_not_found", "message": "任务不存在或已过期。"})
    return {"task": task}


@app.get(f"{WEB_BASE_PATH}/sessions/{{session_id}}")
@app.get("/sessions/{session_id}")
def session_info(session_id: str, request: Request, transcript_dir: Optional[str] = None):
    _enforce_api_policy(request)
    store = _transcript_store(transcript_dir=transcript_dir, user_id=_request_user_id(request))
    return {"session_id": session_id, "events": store.load(session_id)}


@app.get(f"{WEB_BASE_PATH}/sessions")
@app.get("/sessions")
def list_sessions(request: Request):
    _enforce_api_policy(request)
    store = _transcript_store(user_id=_request_user_id(request))
    return {"sessions": store.list_sessions()}


@app.delete(f"{WEB_BASE_PATH}/sessions/{{session_id}}")
@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    _enforce_api_policy(request)
    store = _transcript_store(user_id=_request_user_id(request))
    deleted = store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": "Session 不存在。"})
    return {"status": "ok", "deleted": session_id}


@app.post(f"{WEB_BASE_PATH}/sessions/{{session_id}}/rename")
@app.post("/sessions/{session_id}/rename")
def rename_session(session_id: str, payload: RenameSessionRequest, request: Request):
    _enforce_api_policy(request)
    if not payload.new_session_id.strip():
        raise HTTPException(status_code=400, detail={"code": "empty_session_id", "message": "新 session_id 不能为空。"})
    store = _transcript_store(user_id=_request_user_id(request))
    try:
        renamed = store.rename(session_id, payload.new_session_id.strip())
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail={"code": "session_exists", "message": str(exc)}) from exc
    if not renamed:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": "Session 不存在。"})
    return {"status": "ok", "old_session_id": session_id, "new_session_id": payload.new_session_id.strip()}


@app.post(f"{WEB_BASE_PATH}/sessions/{{session_id}}/summary")
@app.post("/sessions/{session_id}/summary")
def refresh_session_summary(session_id: str, payload: RefreshSummaryRequest, request: Request):
    _enforce_api_policy(request)
    from config import get_config
    from memory.session_summary import SessionSummaryBuilder

    config = get_config()
    store = _transcript_store(user_id=_request_user_id(request))
    events = store.load(session_id)
    if not events:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": "Session 不存在。"})
    source_events = [event for event in events if event.get("role") != "summary"]
    recent_events = source_events[-max(1, payload.max_events):]
    summary_event = SessionSummaryBuilder(max_chars=config.session_summary_max_chars).build_event(session_id, recent_events)
    summary_event["ts"] = time.time()
    kept_events = [event for event in events if event.get("role") != "summary"]
    kept_events.append(summary_event)
    store.replace(session_id, kept_events)
    return {"status": "ok", "session_id": session_id, "summary": summary_event}


def _json_line(payload: dict) -> str:
    return json.dumps(_safe_jsonable(payload), ensure_ascii=False) + "\n"


def _safe_jsonable(value):
    try:
        return jsonable_encoder(value)
    except Exception:
        return _fallback_jsonable(value)


def _fallback_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _fallback_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_fallback_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _fallback_jsonable({field: getattr(value, field) for field in value.__dataclass_fields__})
    if hasattr(value, "__dict__"):
        return _fallback_jsonable(vars(value))
    return str(value)


def _chunk_text(text: str, size: int = 24):
    for index in range(0, len(text), size):
        yield text[index:index + size]


def _enforce_api_policy(request: Request) -> None:
    from config import get_config

    config = get_config()
    if config.api_auth_token:
        supplied = request.headers.get("x-api-key") or request.headers.get("authorization", "").replace("Bearer ", "")
        if supplied != config.api_auth_token:
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "API key 无效或缺失。"})

    if config.api_rate_limit_per_minute <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60
    bucket = [ts for ts in RATE_LIMIT_BUCKETS.get(client, []) if ts >= window_start]
    if len(bucket) >= config.api_rate_limit_per_minute:
        RATE_LIMIT_BUCKETS[client] = bucket
        raise HTTPException(status_code=429, detail={"code": "rate_limited", "message": "请求过于频繁，请稍后再试。"})
    bucket.append(now)
    RATE_LIMIT_BUCKETS[client] = bucket


def _request_user_id(request: Request) -> str:
    return (
        request.headers.get("x-user-id")
        or request.headers.get("x-user-email")
        or request.query_params.get("user_id")
        or "local"
    )


def _transcript_store(transcript_dir: Optional[str] = None, user_id: Optional[str] = None):
    from config import get_config
    from storage.transcript_store import TranscriptStore

    config = get_config()
    root_dir = Path(transcript_dir or config.transcript_dir)
    if user_id:
        root_dir = root_dir / _safe_user_id(user_id)
    return TranscriptStore(root_dir=root_dir, enabled=True)


def _safe_user_id(user_id: str) -> str:
    value = user_id or "local"
    return "".join(char if char.isalnum() or char in "._@-" else "_" for char in value)


def _web_index_path() -> Path:
    return (WEB_DIST_DIR / "index.html") if (WEB_DIST_DIR / "index.html").exists() else WEB_DIR / "index.html"


def _web_index_response(index_path: Path) -> FileResponse:
    return FileResponse(index_path, headers={"Cache-Control": "no-store"})


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    api_prefixes = ("ask", "sessions", "auth", "users", "health", "ingest", "rag", "tasks", "static")
    if full_path.startswith(api_prefixes):
        raise HTTPException(status_code=404, detail="Not found")
    if full_path.startswith(f"{WEB_BASE_PATH.strip('/')}/"):
        nested_path = full_path[len(WEB_BASE_PATH.strip('/')) + 1:]
        if nested_path.startswith(api_prefixes):
            raise HTTPException(status_code=404, detail="Not found")
    index_path = _web_index_path()
    if not index_path.exists():
        raise HTTPException(status_code=404, detail={"code": "web_ui_missing", "message": "Web UI 文件不存在。"})
    return _web_index_response(index_path)
