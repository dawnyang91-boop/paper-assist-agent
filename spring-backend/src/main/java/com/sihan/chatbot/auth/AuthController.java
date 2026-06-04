package com.sihan.chatbot.auth;

import com.sihan.chatbot.common.CurrentUser;
import com.sihan.chatbot.common.SessionUser;
import com.sihan.chatbot.config.ChatbotProperties;
import com.sihan.chatbot.rate.RedisRateLimiter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import java.time.Duration;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class AuthController {
    private final AuthService authService;
    private final CurrentUser currentUser;
    private final RedisRateLimiter rateLimiter;
    private final ChatbotProperties properties;

    public AuthController(
            AuthService authService,
            CurrentUser currentUser,
            RedisRateLimiter rateLimiter,
            ChatbotProperties properties
    ) {
        this.authService = authService;
        this.currentUser = currentUser;
        this.rateLimiter = rateLimiter;
        this.properties = properties;
    }

    @PostMapping("/auth/register")
    public Map<String, Object> register(@Valid @RequestBody RegisterRequest request, HttpServletRequest servletRequest) {
        rateLimiter.check(
                "rate:register:" + servletRequest.getRemoteAddr(),
                properties.getRateLimits().getRegisterPerMinute(),
                Duration.ofMinutes(1)
        );
        UserProfile profile = authService.register(request);
        bindSession(servletRequest.getSession(true), profile);
        return Map.of("user", profile);
    }

    @PostMapping("/auth/login")
    public Map<String, Object> login(@Valid @RequestBody LoginRequest request, HttpServletRequest servletRequest) {
        rateLimiter.check(
                "rate:login:" + servletRequest.getRemoteAddr(),
                properties.getRateLimits().getLoginPerMinute(),
                Duration.ofMinutes(1)
        );
        UserProfile profile = authService.login(request);
        bindSession(servletRequest.getSession(true), profile);
        return Map.of("user", profile);
    }

    @PostMapping("/auth/logout")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void logout(HttpServletRequest request) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            session.invalidate();
        }
    }

    @GetMapping("/users/{userId}")
    public UserProfile getUser(@PathVariable String userId, HttpServletRequest request) {
        SessionUser requester = currentUser.require(request);
        if (!requester.id().equals(userId)) {
            return authService.getUser(requester.id());
        }
        return authService.getUser(userId);
    }

    private void bindSession(HttpSession session, UserProfile profile) {
        session.setAttribute("user", profile.toSessionUser());
        session.setAttribute("userId", profile.id());
        session.setAttribute("email", profile.email());
    }
}
