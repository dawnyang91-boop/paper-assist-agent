package com.sihan.chatbot.lock;

import java.time.Duration;
import java.util.UUID;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

@Component
public class RedisLockService {
    private final StringRedisTemplate redisTemplate;

    public RedisLockService(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    public LockHandle tryLock(String key, Duration ttl) {
        String token = UUID.randomUUID().toString();
        Boolean locked = redisTemplate.opsForValue().setIfAbsent(key, token, ttl);
        if (!Boolean.TRUE.equals(locked)) {
            return null;
        }
        return new LockHandle(key, token);
    }

    public void unlock(LockHandle handle) {
        if (handle == null) {
            return;
        }
        String current = redisTemplate.opsForValue().get(handle.key());
        if (handle.token().equals(current)) {
            redisTemplate.delete(handle.key());
        }
    }

    public record LockHandle(String key, String token) {
    }
}
