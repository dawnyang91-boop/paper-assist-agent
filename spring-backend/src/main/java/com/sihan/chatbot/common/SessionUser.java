package com.sihan.chatbot.common;

import java.io.Serializable;

public record SessionUser(
        String id,
        String email,
        String name,
        String role,
        String accountTier,
        String institution,
        String avatarUrl
) implements Serializable {
}
