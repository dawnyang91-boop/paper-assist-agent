package com.sihan.chatbot.agent;

import com.sihan.chatbot.common.SessionUser;
import com.sihan.chatbot.config.ChatbotProperties;
import java.time.Duration;
import java.util.Map;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Component
public class AgentClient {
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE =
            new ParameterizedTypeReference<>() {
            };

    private final ChatbotProperties properties;
    private final WebClient webClient;

    public AgentClient(ChatbotProperties properties, WebClient.Builder builder) {
        this.properties = properties;
        this.webClient = builder
                .baseUrl(trimTrailingSlash(properties.getAgentBaseUrl()))
                .build();
    }

    public Flux<DataBuffer> askStream(Map<String, Object> payload, SessionUser user) {
        return webClient.post()
                .uri("/ask/stream")
                .headers(headers -> applyAgentHeaders(headers, user))
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(payload)
                .retrieve()
                .bodyToFlux(DataBuffer.class);
    }

    public Map<String, Object> postJson(String path, Map<String, Object> payload, SessionUser user) {
        return webClient.post()
                .uri(path)
                .headers(headers -> applyAgentHeaders(headers, user))
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(payload)
                .retrieve()
                .bodyToMono(MAP_TYPE)
                .block(Duration.ofMinutes(10));
    }

    public Map<String, Object> getJson(String path, SessionUser user) {
        return webClient.get()
                .uri(path)
                .headers(headers -> applyAgentHeaders(headers, user))
                .retrieve()
                .bodyToMono(MAP_TYPE)
                .block(Duration.ofSeconds(30));
    }

    public Map<String, Object> deleteJson(String path, SessionUser user) {
        return webClient.delete()
                .uri(path)
                .headers(headers -> applyAgentHeaders(headers, user))
                .retrieve()
                .bodyToMono(MAP_TYPE)
                .onErrorResume(error -> Mono.just(Map.of("ok", true)))
                .block(Duration.ofSeconds(30));
    }

    private void applyAgentHeaders(HttpHeaders headers, SessionUser user) {
        if (StringUtils.hasText(properties.getInternalApiToken())) {
            headers.setBearerAuth(properties.getInternalApiToken());
        }
        if (user != null) {
            putIfPresent(headers, "X-User-Id", user.id());
            putIfPresent(headers, "X-User-Email", user.email());
            putIfPresent(headers, "X-User-Name", user.name());
            putIfPresent(headers, "X-User-Role", user.role());
            putIfPresent(headers, "X-User-Account-Tier", user.accountTier());
            putIfPresent(headers, "X-User-Institution", user.institution());
            putIfPresent(headers, "X-User-Avatar-Url", user.avatarUrl());
        }
    }

    private void putIfPresent(HttpHeaders headers, String key, String value) {
        if (StringUtils.hasText(value)) {
            headers.set(key, value);
        }
    }

    private static String trimTrailingSlash(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.replaceAll("/+$", "");
    }
}
