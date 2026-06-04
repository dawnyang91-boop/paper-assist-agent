import hashlib
import json
from typing import Any, Optional

from config import AppConfig, get_config
from redis_runtime import RedisRuntime, get_redis_runtime


class RedisJsonCache:
    def __init__(self, runtime: Optional[RedisRuntime] = None, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.runtime = runtime if runtime is not None else get_redis_runtime(self.config)

    @property
    def enabled(self) -> bool:
        return self.runtime is not None

    def make_key(self, namespace: str, *parts: object) -> Optional[str]:
        if not self.runtime:
            return None
        digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
        return self.runtime.key("cache", namespace, digest)

    def get(self, key: Optional[str]) -> Any:
        if not self.runtime or not key:
            return None
        raw = self.runtime.client.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, key: Optional[str], value: Any, ttl_seconds: Optional[int] = None) -> None:
        if not self.runtime or not key:
            return
        ttl = ttl_seconds or self.config.redis_cache_ttl_seconds
        self.runtime.client.setex(key, ttl, json.dumps(value, ensure_ascii=False, default=str))
