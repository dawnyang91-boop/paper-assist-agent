package com.sihan.chatbot.upload;

import com.sihan.chatbot.agent.AgentClient;
import com.sihan.chatbot.common.CurrentUser;
import com.sihan.chatbot.common.SessionUser;
import com.sihan.chatbot.config.ChatbotProperties;
import com.sihan.chatbot.rate.RedisRateLimiter;
import com.sihan.chatbot.task.TaskService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/rag/uploads")
public class RagUploadController {
    private final UploadService uploadService;
    private final CurrentUser currentUser;
    private final RedisRateLimiter rateLimiter;
    private final ChatbotProperties properties;
    private final TaskService taskService;
    private final AgentClient agentClient;
    private final ObjectMapper objectMapper;

    public RagUploadController(
            UploadService uploadService,
            CurrentUser currentUser,
            RedisRateLimiter rateLimiter,
            ChatbotProperties properties,
            TaskService taskService,
            AgentClient agentClient,
            ObjectMapper objectMapper
    ) {
        this.uploadService = uploadService;
        this.currentUser = currentUser;
        this.rateLimiter = rateLimiter;
        this.properties = properties;
        this.taskService = taskService;
        this.agentClient = agentClient;
        this.objectMapper = objectMapper;
    }

    @GetMapping
    public Map<String, Object> listUploads(HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        return Map.of("files", uploadService.listFiles(user));
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> upload(
            @RequestParam("files") List<MultipartFile> files,
            HttpServletRequest request
    ) {
        SessionUser user = currentUser.require(request);
        rateLimiter.check("rate:upload:" + user.id(), properties.getRateLimits().getUploadPerMinute(), Duration.ofMinutes(1));
        return Map.of("status", "ok", "files", uploadService.replaceFiles(files, user));
    }

    @DeleteMapping("/{fileId}")
    public Map<String, Object> deleteUpload(@PathVariable String fileId, HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        boolean deleted = uploadService.deleteFile(fileId, user);
        return Map.of("status", "ok", "deleted", deleted ? fileId : "");
    }

    @PostMapping("/ingest")
    public Map<String, Object> ingest(HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        rateLimiter.check("rate:ingest:" + user.id(), properties.getRateLimits().getIngestPerMinute(), Duration.ofMinutes(1));
        String activeTaskId = taskService.activeIngestTaskId(user);
        if (activeTaskId != null) {
            return Map.of("status", "accepted", "task_id", activeTaskId, "upload_dir", uploadService.uploadDir(user));
        }

        String taskId = taskService.createIngestTask(user, uploadService.uploadDir(user));
        CompletableFuture.runAsync(() -> runIngest(taskId, user));
        return Map.of("status", "accepted", "task_id", taskId, "upload_dir", uploadService.uploadDir(user));
    }

    @SuppressWarnings("unchecked")
    private void runIngest(String taskId, SessionUser user) {
        try {
            taskService.markRunning(taskId, "正在请求 Python Agent 执行向量化");
            Map<String, Object> accepted = agentClient.postJson("/rag/uploads/ingest", Map.of("user_id", user.id()), user);
            Object pythonTaskId = accepted.get("task_id");
            if (pythonTaskId == null) {
                taskService.markSucceeded(taskId, toJson(accepted));
                return;
            }
            for (int i = 0; i < 900; i++) {
                Map<String, Object> response = agentClient.getJson("/tasks/" + pythonTaskId, user);
                Object taskObject = response.get("task");
                if (taskObject instanceof Map<?, ?> rawTask) {
                    Map<String, Object> agentTask = (Map<String, Object>) rawTask;
                    taskService.syncFromAgentTask(taskId, agentTask);
                    String status = String.valueOf(agentTask.getOrDefault("status", ""));
                    if ("succeeded".equals(status)) {
                        taskService.markSucceeded(taskId, toJson(agentTask));
                        uploadService.clearFiles(user);
                        return;
                    }
                    if ("failed".equals(status) || "cancelled".equals(status)) {
                        taskService.markFailed(taskId, String.valueOf(agentTask.getOrDefault("error", "向量化失败")));
                        return;
                    }
                }
                Thread.sleep(2000);
            }
            taskService.markFailed(taskId, "向量化任务超时");
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
            taskService.markFailed(taskId, exc.getMessage());
        } catch (Exception exc) {
            taskService.markFailed(taskId, exc.getMessage());
        }
    }

    private String toJson(Object value) throws JsonProcessingException {
        return objectMapper.writeValueAsString(value);
    }
}
