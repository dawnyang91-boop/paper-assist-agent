import json
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

from config import AppConfig, get_config
from redis_runtime import RedisUnavailable, get_redis_runtime
from redis_task_queue import RedisTaskStore


class LocalTaskStore:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create_task(self, task_type: str, user_id: Optional[str] = None, session_id: Optional[str] = None, input_data: Optional[dict] = None) -> str:
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = {
                "type": task_type,
                "status": "pending",
                "progress": "0",
                "message": "",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "user_id": user_id or "",
                "session_id": session_id or "",
                "input_json": json.dumps(input_data or {}, ensure_ascii=False),
            }
        return task_id

    def update_task(self, task_id: str, fields: Dict[str, Any]) -> None:
        with self.lock:
            task = self.tasks.setdefault(task_id, {})
            task.update({key: self._stringify(value) for key, value in fields.items()})
            task["updated_at"] = str(time.time())

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            task = self.tasks.get(task_id)
            return dict(task) if task else None

    def _stringify(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return "" if value is None else str(value)
        return json.dumps(value, ensure_ascii=False, default=str)


_LOCAL_TASK_STORE = LocalTaskStore()


def get_task_store(config: Optional[AppConfig] = None):
    cfg = config or get_config()
    try:
        runtime = get_redis_runtime(cfg)
    except RedisUnavailable as exc:
        print(f"[警告] Redis 任务状态不可用，已回退到进程内任务状态：{exc}")
        runtime = None
    if runtime is None:
        return _LOCAL_TASK_STORE
    return RedisTaskStore(runtime=runtime, config=cfg)


def run_background_task(task_id: str, func: Callable[[], Any], task_store=None) -> None:
    store = task_store or get_task_store()

    def runner():
        store.update_task(task_id, {"status": "running", "progress": 5, "message": "任务已开始", "started_at": time.time()})
        try:
            result = func()
            store.update_task(
                task_id,
                {
                    "status": "succeeded",
                    "progress": 100,
                    "message": "任务完成",
                    "result_json": result or {},
                    "finished_at": time.time(),
                },
            )
        except Exception as exc:
            store.update_task(
                task_id,
                {
                    "status": "failed",
                    "progress": 100,
                    "message": "任务失败",
                    "error": str(exc),
                    "finished_at": time.time(),
                },
            )

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
