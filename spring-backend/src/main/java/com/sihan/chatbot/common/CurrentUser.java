package com.sihan.chatbot.common;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.server.ResponseStatusException;

@Component
public class CurrentUser {

    public SessionUser require(HttpServletRequest request) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            Object user = session.getAttribute("user");
            if (user instanceof SessionUser sessionUser) {
                return sessionUser;
            }
        }

        String headerUserId = request.getHeader("X-User-Id");
        if (StringUtils.hasText(headerUserId)) {
            return new SessionUser(
                    headerUserId,
                    request.getHeader("X-User-Email"),
                    request.getHeader("X-User-Name"),
                    request.getHeader("X-User-Role"),
                    request.getHeader("X-User-Account-Tier"),
                    request.getHeader("X-User-Institution"),
                    request.getHeader("X-User-Avatar-Url")
            );
        }

        throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "请先登录");
    }
}
