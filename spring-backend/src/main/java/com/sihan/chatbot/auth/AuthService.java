package com.sihan.chatbot.auth;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.server.ResponseStatusException;

@Service
public class AuthService {
    private static final String EMAIL_KEY_PREFIX = "auth:user:email:";
    private static final String USER_KEY_PREFIX = "auth:user:";

    private final StringRedisTemplate redisTemplate;
    private final PasswordEncoder passwordEncoder;

    public AuthService(StringRedisTemplate redisTemplate, PasswordEncoder passwordEncoder) {
        this.redisTemplate = redisTemplate;
        this.passwordEncoder = passwordEncoder;
    }

    public UserProfile register(RegisterRequest request) {
        String email = request.email().trim().toLowerCase();
        String userId = UUID.randomUUID().toString();
        Boolean ok = redisTemplate.opsForValue().setIfAbsent(emailKey(email), userId);
        if (!Boolean.TRUE.equals(ok)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "邮箱已注册");
        }

        Map<String, String> fields = new HashMap<>();
        fields.put("id", userId);
        fields.put("email", email);
        fields.put("name", request.name());
        fields.put("phone", valueOrEmpty(request.phone()));
        fields.put("passwordHash", passwordEncoder.encode(request.password()));
        fields.put("role", "user");
        fields.put("accountTier", "free");
        fields.put("institution", "");
        fields.put("avatarUrl", valueOrEmpty(request.avatarUrl()));
        redisTemplate.opsForHash().putAll(userKey(userId), fields);
        return toProfile(fields);
    }

    public UserProfile login(LoginRequest request) {
        String email = request.email().trim().toLowerCase();
        String userId = redisTemplate.opsForValue().get(emailKey(email));
        if (!StringUtils.hasText(userId)) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "账号不存在，请先注册账户");
        }
        Map<Object, Object> fields = redisTemplate.opsForHash().entries(userKey(userId));
        String passwordHash = (String) fields.get("passwordHash");
        if (!StringUtils.hasText(passwordHash) || !passwordEncoder.matches(request.password(), passwordHash)) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "密码错误，请重新输入");
        }
        return toProfile(stringMap(fields));
    }

    public UserProfile getUser(String userId) {
        Map<Object, Object> fields = redisTemplate.opsForHash().entries(userKey(userId));
        if (fields.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "用户不存在");
        }
        return toProfile(stringMap(fields));
    }

    private UserProfile toProfile(Map<String, String> fields) {
        return new UserProfile(
                fields.get("id"),
                fields.get("email"),
                fields.getOrDefault("name", "User"),
                fields.getOrDefault("phone", ""),
                fields.getOrDefault("role", "user"),
                fields.getOrDefault("accountTier", "free"),
                fields.getOrDefault("institution", ""),
                fields.getOrDefault("avatarUrl", ""),
                Map.of("liked", 0, "streak", 0, "folders", 0),
                List.of(),
                List.of()
        );
    }

    private Map<String, String> stringMap(Map<Object, Object> fields) {
        Map<String, String> result = new HashMap<>();
        fields.forEach((key, value) -> result.put(String.valueOf(key), String.valueOf(value)));
        return result;
    }

    private String emailKey(String email) {
        return EMAIL_KEY_PREFIX + email;
    }

    private String userKey(String userId) {
        return USER_KEY_PREFIX + userId;
    }

    private String valueOrEmpty(String value) {
        return value == null ? "" : value;
    }
}
