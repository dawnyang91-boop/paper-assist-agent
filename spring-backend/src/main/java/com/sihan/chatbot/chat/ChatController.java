package com.sihan.chatbot.chat;

import com.sihan.chatbot.agent.AgentClient;
import com.sihan.chatbot.common.CurrentUser;
import com.sihan.chatbot.common.SessionUser;
import com.sihan.chatbot.config.ChatbotProperties;
import com.sihan.chatbot.rate.RedisRateLimiter;
import jakarta.servlet.http.HttpServletRequest;
import java.time.Duration;
import java.util.Map;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/ask")
public class ChatController {
    private final AgentClient agentClient;
    private final CurrentUser currentUser;
    private final RedisRateLimiter rateLimiter;
    private final ChatbotProperties properties;

    public ChatController(
            AgentClient agentClient,
            CurrentUser currentUser,
            RedisRateLimiter rateLimiter,
            ChatbotProperties properties
    ) {
        this.agentClient = agentClient;
        this.currentUser = currentUser;
        this.rateLimiter = rateLimiter;
        this.properties = properties;
    }

    @PostMapping(value = "/stream", produces = "application/x-ndjson")
    public Flux<DataBuffer> askStream(@RequestBody Map<String, Object> payload, HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        rateLimiter.check("rate:ask:" + user.id(), properties.getRateLimits().getAskPerMinute(), Duration.ofMinutes(1));
        return agentClient.askStream(payload, user);
    }

    @PostMapping(produces = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> ask(@RequestBody Map<String, Object> payload, HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        rateLimiter.check("rate:ask:" + user.id(), properties.getRateLimits().getAskPerMinute(), Duration.ofMinutes(1));
        return agentClient.postJson("/ask", payload, user);
    }
}
