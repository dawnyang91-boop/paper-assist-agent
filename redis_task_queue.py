import json
import time
import uuid
from typing import Any, Dict, Optional

from config import AppConfig, get_config
from redis_runtime import RedisRuntime, get_redis_runtime


class RedisTaskStore:
    def __init__(self, runtime: Optional[RedisRuntime] = None, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.runtime = runtime if runtime is not None else get_redis_runtime(self.config)

    @property
    def enabled(self) -> bool:
        return self.runtime is not None

    def create_task(
        self,
        task_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        self.update_task(
            task_id,
            {
                "type": task_type,
                "status": "pending",
                "progress": "0",
                "message": "",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
                "user_id": user_id or "",
                "session_id": session_id or "",
                "input_json": json.dumps(input_data or {}, ensure_ascii=False),
            },
        )
        return task_id

    def update_task(self, task_id: str, fields: Dict[str, Any]) -> None:
        if not self.runtime:
            return
        key = self.runtime.key("task", task_id)
        payload = {name: self._stringify(value) for name, value in fields.items()}
        payload.setdefault("updated_at", str(time.time()))
        self.runtime.client.hset(key, mapping=payload)
        self.runtime.client.expire(key, self.config.redis_task_ttl_seconds)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not self.runtime:
            return None
        data = self.runtime.client.hgetall(self.runtime.key("task", task_id))
        if not data:
            return None
        return dict(data)

    def enqueue_overflow(self, task_id: str, session_id: str, overflow_items: list[dict]) -> Optional[str]:
        if not self.runtime:
            return None
        stream = self.runtime.key("memory", "overflow")
        return self.runtime.client.xadd(
            stream,
            {
                "task_id": task_id,
                "session_id": session_id,
                "overflow_items_json": json.dumps(overflow_items, ensure_ascii=False, default=str),
                "created_at": str(time.time()),
                "retry_count": "0",
            },
        )

    def _stringify(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return "" if value is None else str(value)
        return json.dumps(value, ensure_ascii=False, default=str)
