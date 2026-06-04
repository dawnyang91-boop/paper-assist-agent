import json
import time
from typing import Any, Dict, List, Optional

from config import AppConfig, get_config
from storage.redis_runtime import RedisRuntime, get_redis_runtime


class RedisWorkingMemoryStore:
    """Redis Stream based hot working memory store."""

    def __init__(self, runtime: Optional[RedisRuntime] = None, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.runtime = runtime if runtime is not None else get_redis_runtime(self.config)

    @property
    def enabled(self) -> bool:
        return self.runtime is not None

    def key(self, session_id: str, user_id: str = "local") -> Optional[str]:
        if not self.runtime:
            return None
        return self.runtime.key("wm", user_id or "local", session_id or "default")

    def append(self, session_id: str, memory: Dict[str, Any], user_id: str = "local") -> None:
        key = self.key(session_id, user_id=user_id)
        if not self.runtime or not key:
            return
        payload = {
            "session_id": session_id,
            "role": str(memory.get("role", "")),
            "content": str(memory.get("content", "")),
            "timestamp": str(memory.get("timestamp", time.time())),
            "importance": str(memory.get("importance", 0)),
            "metadata_json": json.dumps(memory.get("metadata", {}), ensure_ascii=False, default=str),
        }
        self.runtime.client.xadd(key, payload)
        self.runtime.client.xtrim(key, maxlen=self.config.working_memory_max_capacity, approximate=True)
        self.runtime.client.expire(key, self.config.redis_working_memory_ttl_seconds)

    def read_recent(self, session_id: str, count: int, user_id: str = "local") -> List[Dict[str, Any]]:
        key = self.key(session_id, user_id=user_id)
        if not self.runtime or not key:
            return []
        rows = self.runtime.client.xrevrange(key, count=max(1, int(count)))
        memories = [self._decode(row_id, fields) for row_id, fields in rows]
        return list(reversed(memories))

    def clear(self, session_id: str, user_id: str = "local") -> None:
        key = self.key(session_id, user_id=user_id)
        if self.runtime and key:
            self.runtime.client.delete(key)

    def _decode(self, row_id: str, fields: Dict[str, str]) -> Dict[str, Any]:
        try:
            metadata = json.loads(fields.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        try:
            timestamp = float(fields.get("timestamp") or 0)
        except ValueError:
            timestamp = time.time()
        try:
            importance = float(fields.get("importance") or 0)
        except ValueError:
            importance = 0.0
        return {
            "redis_id": row_id,
            "session_id": fields.get("session_id", "default"),
            "role": fields.get("role", "memory"),
            "content": fields.get("content", ""),
            "timestamp": timestamp,
            "importance": importance,
            "metadata": metadata,
        }
