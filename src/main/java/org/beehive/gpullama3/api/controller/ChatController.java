package org.beehive.gpullama3.api.controller;

import org.beehive.gpullama3.api.service.LLMService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;

@RestController
public class ChatController {

    @Autowired
    private LLMService llmService;

    @PostMapping("/chat")
    public Map<String, String> chat(@RequestBody ChatRequest request) {
        // Use request parameters with fallbacks to defaults
        int maxTokens = request.getMaxTokens() != null ? request.getMaxTokens() : 150;
        double temperature = request.getTemperature() != null ? request.getTemperature() : 0.7;
        double topP = request.getTopP() != null ? request.getTopP() : 0.9;

        logRequest("NON_STREAMING", request, maxTokens, temperature, topP);

        if (request.getMessage() == null || request.getMessage().trim().isEmpty()) {
            throw new IllegalArgumentException("Message cannot be empty");
        }

        String response = llmService.generateResponse(request.getMessage(), request.getSystemMessage(),
                maxTokens, temperature, topP);

        return Map.of("response", response);
    }

    @PostMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamChat(@RequestBody ChatRequest request) {
        // Use request parameters with fallbacks to defaults
        int maxTokens = request.getMaxTokens() != null ? request.getMaxTokens() : 150;
        double temperature = request.getTemperature() != null ? request.getTemperature() : 0.7;
        double topP = request.getTopP() != null ? request.getTopP() : 0.9;

        logRequest("STREAMING", request, maxTokens, temperature, topP);

        if (request.getMessage() == null || request.getMessage().trim().isEmpty()) {
            throw new IllegalArgumentException("Message cannot be empty");
        }

        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);
        llmService.generateStreamingResponse(
                request.getMessage(),
                request.getSystemMessage(),
                emitter,
                maxTokens,
                temperature,
                topP,
                request.getSeed());

        return emitter;
    }


    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "healthy", "timestamp", String.valueOf(System.currentTimeMillis()));
    }

    private void logRequest(String type, ChatRequest request, int maxTokens, double temperature, double topP) {
        System.out.printf("REQUEST [%s] user='%s' system='%s' maxTokens=%d temp=%.2f topP=%.2f%n",
                type,
                truncate(request.getMessage(), 100),
                request.getSystemMessage() != null ? truncate(request.getSystemMessage(), 40) : "none",
                maxTokens,
                temperature,
                topP
        );
    }

    private String truncate(String text, int maxLength) {
        if (text == null) return "null";
        return text.length() > maxLength ? text.substring(0, maxLength) + "..." : text;
    }

    // Simple request class for custom parameters
    public static class ChatRequest {
        private String message;
        private String systemMessage;
        private Integer maxTokens;
        private Double temperature;
        private Double topP;
        private Long seed;

        // Getters and Setters
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }

        public String getSystemMessage() { return systemMessage; }
        public void setSystemMessage(String systemMessage) { this.systemMessage = systemMessage; }

        public Integer getMaxTokens() { return maxTokens; }
        public void setMaxTokens(Integer maxTokens) { this.maxTokens = maxTokens; }

        public Double getTemperature() { return temperature; }
        public void setTemperature(Double temperature) { this.temperature = temperature; }

        public Double getTopP() { return topP; }
        public void setTopP(Double topP) { this.topP = topP; }

        public Long getSeed() { return seed; }
        public void setSeed(Long seed) { this.seed = seed; }
    }

}
