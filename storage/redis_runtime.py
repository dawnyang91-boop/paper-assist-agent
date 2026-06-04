from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import AppConfig, get_config


class RedisUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RedisRuntime:
    client: object
    prefix: str

    def key(self, *parts: object) -> str:
        clean = [self.prefix.strip(":")]
        clean.extend(str(part).strip(":") for part in parts if part is not None and str(part) != "")
        return ":".join(clean)


_RUNTIME: Optional[RedisRuntime] = None


def get_redis_runtime(config: Optional[AppConfig] = None) -> Optional[RedisRuntime]:
    """Return a shared Redis runtime when enabled; otherwise None."""
    cfg = config or get_config()
    if not cfg.redis_enabled:
        return None

    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    try:
        import redis
    except ImportError as exc:
        raise RedisUnavailable("已启用 REDIS_ENABLED，但未安装 redis。请运行 `pip install -r requirements.txt`。") from exc

    client = redis.Redis.from_url(
        cfg.redis_url,
        decode_responses=True,
        socket_timeout=cfg.redis_socket_timeout_seconds,
        socket_connect_timeout=cfg.redis_socket_timeout_seconds,
    )
    try:
        client.ping()
    except Exception as exc:
        raise RedisUnavailable(f"Redis 连接不可用：{exc}") from exc

    _RUNTIME = RedisRuntime(client=client, prefix=cfg.redis_prefix)
    return _RUNTIME


def clear_redis_runtime() -> None:
    global _RUNTIME
    _RUNTIME = None
