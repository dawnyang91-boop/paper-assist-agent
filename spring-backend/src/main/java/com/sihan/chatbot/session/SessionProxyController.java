package com.sihan.chatbot.session;

import com.sihan.chatbot.agent.AgentClient;
import com.sihan.chatbot.common.CurrentUser;
import com.sihan.chatbot.common.SessionUser;
import jakarta.servlet.http.HttpServletRequest;
import java.util.Map;
import java.nio.charset.StandardCharsets;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.util.UriUtils;

@RestController
@RequestMapping("/api/sessions")
public class SessionProxyController {
    private final AgentClient agentClient;
    private final CurrentUser currentUser;

    public SessionProxyController(AgentClient agentClient, CurrentUser currentUser) {
        this.agentClient = agentClient;
        this.currentUser = currentUser;
    }

    @GetMapping
    public Map<String, Object> listSessions(HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        return agentClient.getJson("/sessions", user);
    }

    @GetMapping("/{sessionId}")
    public Map<String, Object> loadSession(@PathVariable String sessionId, HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        return agentClient.getJson("/sessions/" + encodePathSegment(sessionId), user);
    }

    @PostMapping("/{sessionId}/rename")
    public Map<String, Object> renameSession(
            @PathVariable String sessionId,
            @RequestBody Map<String, Object> payload,
            HttpServletRequest request
    ) {
        SessionUser user = currentUser.require(request);
        return agentClient.postJson("/sessions/" + encodePathSegment(sessionId) + "/rename", payload, user);
    }

    @PostMapping("/{sessionId}/summary")
    public Map<String, Object> refreshSummary(
            @PathVariable String sessionId,
            @RequestBody Map<String, Object> payload,
            HttpServletRequest request
    ) {
        SessionUser user = currentUser.require(request);
        return agentClient.postJson("/sessions/" + encodePathSegment(sessionId) + "/summary", payload, user);
    }

    @DeleteMapping("/{sessionId}")
    public Map<String, Object> deleteSession(@PathVariable String sessionId, HttpServletRequest request) {
        SessionUser user = currentUser.require(request);
        return agentClient.deleteJson("/sessions/" + encodePathSegment(sessionId), user);
    }

    private String encodePathSegment(String value) {
        return UriUtils.encodePathSegment(value, StandardCharsets.UTF_8);
    }
}
