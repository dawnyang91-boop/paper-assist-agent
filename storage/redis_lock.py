import secrets
from dataclasses import dataclass
from typing import Optional

from config import AppConfig, get_config
from storage.redis_runtime import RedisRuntime, get_redis_runtime


_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""


@dataclass
class RedisLockHandle:
    key: str
    token: str
    runtime: RedisRuntime

    def release(self) -> bool:
        result = self.runtime.client.eval(_RELEASE_SCRIPT, 1, self.key, self.token)
        return bool(result)


class RedisLockManager:
    def __init__(self, runtime: Optional[RedisRuntime] = None, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.runtime = runtime if runtime is not None else get_redis_runtime(self.config)

    def acquire(self, scope: str, resource_id: str, ttl_seconds: int = 60) -> Optional[RedisLockHandle]:
        if not self.runtime:
            return None
        key = self.runtime.key("lock", scope, resource_id)
        token = secrets.token_urlsafe(24)
        acquired = self.runtime.client.set(key, token, nx=True, ex=max(1, int(ttl_seconds)))
        if not acquired:
            return None
        return RedisLockHandle(key=key, token=token, runtime=self.runtime)
