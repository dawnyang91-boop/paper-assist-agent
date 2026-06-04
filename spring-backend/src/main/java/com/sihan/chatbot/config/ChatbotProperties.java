package com.sihan.chatbot.config;

import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "chatbot")
public class ChatbotProperties {
    private String agentBaseUrl = "http://127.0.0.1:8000/chatbot";
    private String internalApiToken = "";
    private String uploadRoot = "../data/uploads";
    private String allowedExtensions = ".txt,.md,.pdf,.docx,.pptx,.xlsx,.csv,.png,.jpg,.jpeg,.webp";
    private long maxFileSizeMb = 80;
    private List<String> corsAllowedOrigins = new ArrayList<>();
    private RateLimits rateLimits = new RateLimits();

    public String getAgentBaseUrl() {
        return agentBaseUrl;
    }

    public void setAgentBaseUrl(String agentBaseUrl) {
        this.agentBaseUrl = agentBaseUrl;
    }

    public String getInternalApiToken() {
        return internalApiToken;
    }

    public void setInternalApiToken(String internalApiToken) {
        this.internalApiToken = internalApiToken;
    }

    public String getUploadRoot() {
        return uploadRoot;
    }

    public void setUploadRoot(String uploadRoot) {
        this.uploadRoot = uploadRoot;
    }

    public String getAllowedExtensions() {
        return allowedExtensions;
    }

    public void setAllowedExtensions(String allowedExtensions) {
        this.allowedExtensions = allowedExtensions;
    }

    public long getMaxFileSizeMb() {
        return maxFileSizeMb;
    }

    public void setMaxFileSizeMb(long maxFileSizeMb) {
        this.maxFileSizeMb = maxFileSizeMb;
    }

    public List<String> getCorsAllowedOrigins() {
        return corsAllowedOrigins;
    }

    public void setCorsAllowedOrigins(List<String> corsAllowedOrigins) {
        this.corsAllowedOrigins = corsAllowedOrigins;
    }

    public RateLimits getRateLimits() {
        return rateLimits;
    }

    public void setRateLimits(RateLimits rateLimits) {
        this.rateLimits = rateLimits;
    }

    public static class RateLimits {
        private int loginPerMinute = 10;
        private int registerPerMinute = 10;
        private int askPerMinute = 30;
        private int uploadPerMinute = 20;
        private int ingestPerMinute = 3;

        public int getLoginPerMinute() {
            return loginPerMinute;
        }

        public void setLoginPerMinute(int loginPerMinute) {
            this.loginPerMinute = loginPerMinute;
        }

        public int getRegisterPerMinute() {
            return registerPerMinute;
        }

        public void setRegisterPerMinute(int registerPerMinute) {
            this.registerPerMinute = registerPerMinute;
        }

        public int getAskPerMinute() {
            return askPerMinute;
        }

        public void setAskPerMinute(int askPerMinute) {
            this.askPerMinute = askPerMinute;
        }

        public int getUploadPerMinute() {
            return uploadPerMinute;
        }

        public void setUploadPerMinute(int uploadPerMinute) {
            this.uploadPerMinute = uploadPerMinute;
        }

        public int getIngestPerMinute() {
            return ingestPerMinute;
        }

        public void setIngestPerMinute(int ingestPerMinute) {
            this.ingestPerMinute = ingestPerMinute;
        }
    }
}
