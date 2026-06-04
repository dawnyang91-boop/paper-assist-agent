package com.sihan.chatbot.auth;

import com.sihan.chatbot.common.SessionUser;
import java.util.List;
import java.util.Map;

public record UserProfile(
        String id,
        String email,
        String name,
        String phone,
        String role,
        String accountTier,
        String institution,
        String avatarUrl,
        Map<String, Integer> stats,
        List<String> interests,
        List<String> researchFields
) {
    public SessionUser toSessionUser() {
        return new SessionUser(id, email, name, role, accountTier, institution, avatarUrl);
    }
}
