package com.sihan.chatbot.health;

import com.sihan.chatbot.agent.AgentClient;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class HealthController {
    private final AgentClient agentClient;

    public HealthController(AgentClient agentClient) {
        this.agentClient = agentClient;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("status", "ok", "service", "chatbot-business-backend");
    }

    @GetMapping("/agent/health")
    public Map<String, Object> agentHealth() {
        return agentClient.getJson("/health", null);
    }
}
