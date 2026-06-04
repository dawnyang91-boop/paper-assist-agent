package com.sihan.chatbot.task;

import com.sihan.chatbot.common.SessionUser;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
public class TaskService {
    private static final String TASK_KEY_PREFIX = "task:";
    private static final String ACTIVE_INGEST_PREFIX = "task:ingest:active:";
    private static final Duration TASK_TTL = Duration.ofHours(6);

    private final StringRedisTemplate redisTemplate;

    public TaskService(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    public String activeIngestTaskId(SessionUser user) {
        String taskId = redisTemplate.opsForValue().get(activeIngestKey(user));
        if (!StringUtils.hasText(taskId)) {
            return null;
        }
        Map<String, Object> task = getTask(taskId);
        String status = stringValue(task.get("status"));
        if (isTerminal(status)) {
            redisTemplate.delete(activeIngestKey(user));
            return null;
        }
        return taskId;
    }

    public String createIngestTask(SessionUser user, String uploadDir) {
        String taskId = UUID.randomUUID().toString();
        String now = Instant.now().toString();
        Map<String, String> fields = new HashMap<>();
        fields.put("type", "rag_upload_ingest");
        fields.put("status", "pending");
        fields.put("progress", "0");
        fields.put("message", "等待向量化任务开始");
        fields.put("userId", user.id());
        fields.put("uploadDir", uploadDir);
        fields.put("created_at", now);
        fields.put("updated_at", now);
        redisTemplate.opsForHash().putAll(taskKey(taskId), fields);
        redisTemplate.expire(taskKey(taskId), TASK_TTL);
        redisTemplate.opsForValue().set(activeIngestKey(user), taskId, TASK_TTL);
        return taskId;
    }

    public void markRunning(String taskId, String message) {
        update(taskId, Map.of("status", "running", "progress", "20", "message", message));
    }

    public void markSucceeded(String taskId, String resultJson) {
        update(taskId, Map.of(
                "status", "succeeded",
                "progress", "100",
                "message", "任务完成",
                "result_json", resultJson == null ? "{}" : resultJson
        ));
        clearActiveForTask(taskId);
    }

    public void syncFromAgentTask(String taskId, Map<String, Object> agentTask) {
        Map<String, String> fields = new HashMap<>();
        copyIfPresent(agentTask, fields, "status");
        copyIfPresent(agentTask, fields, "progress");
        copyIfPresent(agentTask, fields, "message");
        copyIfPresent(agentTask, fields, "error");
        copyIfPresent(agentTask, fields, "result_json");
        if (!fields.isEmpty()) {
            update(taskId, fields);
        }
        String status = stringValue(agentTask.get("status"));
        if (isTerminal(status)) {
            clearActiveForTask(taskId);
        }
    }

    public void markFailed(String taskId, String error) {
        update(taskId, Map.of(
                "status", "failed",
                "progress", "100",
                "message", "任务失败",
                "error", error == null ? "unknown error" : error
        ));
        clearActiveForTask(taskId);
    }

    public Map<String, Object> getTask(String taskId) {
        Map<Object, Object> raw = redisTemplate.opsForHash().entries(taskKey(taskId));
        Map<String, Object> result = new HashMap<>();
        raw.forEach((key, value) -> result.put(String.valueOf(key), value));
        return result;
    }

    private void update(String taskId, Map<String, String> fields) {
        Map<String, String> values = new HashMap<>(fields);
        values.put("updated_at", Instant.now().toString());
        redisTemplate.opsForHash().putAll(taskKey(taskId), values);
        redisTemplate.expire(taskKey(taskId), TASK_TTL);
    }

    private void clearActiveForTask(String taskId) {
        Object userId = getTask(taskId).get("userId");
        if (userId != null) {
            redisTemplate.delete(ACTIVE_INGEST_PREFIX + safeUserId(String.valueOf(userId)));
        }
    }

    private boolean isTerminal(String status) {
        return "succeeded".equals(status) || "failed".equals(status) || "cancelled".equals(status);
    }

    private String taskKey(String taskId) {
        return TASK_KEY_PREFIX + taskId;
    }

    private String activeIngestKey(SessionUser user) {
        return ACTIVE_INGEST_PREFIX + safeUserId(user.id());
    }

    private String safeUserId(String userId) {
        return userId == null ? "local" : userId.replaceAll("[^a-zA-Z0-9_.@-]", "_");
    }

    private String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private void copyIfPresent(Map<String, Object> source, Map<String, String> target, String key) {
        Object value = source.get(key);
        if (value != null) {
            target.put(key, String.valueOf(value));
        }
    }
}
