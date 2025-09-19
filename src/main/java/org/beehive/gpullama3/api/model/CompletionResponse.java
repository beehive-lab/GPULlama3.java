package org.beehive.gpullama3.api.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class CompletionResponse {
    private String id;
    private String object = "text_completion";
    private Long created;
    private String model;
    private List<Choice> choices;
    private Usage usage;

    public static class Choice {
        private String text;
        private Integer index;

        @JsonProperty("finish_reason")
        private String finishReason;

        public Choice() {}

        public Choice(String text, Integer index, String finishReason) {
            this.text = text;
            this.index = index;
            this.finishReason = finishReason;
        }

        // Getters and Setters
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }

        public Integer getIndex() { return index; }
        public void setIndex(Integer index) { this.index = index; }

        public String getFinishReason() { return finishReason; }
        public void setFinishReason(String finishReason) { this.finishReason = finishReason; }
    }

    public static class Usage {
        @JsonProperty("prompt_tokens")
        private Integer promptTokens;

        @JsonProperty("completion_tokens")
        private Integer completionTokens;

        @JsonProperty("total_tokens")
        private Integer totalTokens;

        public Usage() {}

        public Usage(Integer promptTokens, Integer completionTokens) {
            this.promptTokens = promptTokens;
            this.completionTokens = completionTokens;
            this.totalTokens = promptTokens + completionTokens;
        }

        // Getters and Setters
        public Integer getPromptTokens() { return promptTokens; }
        public void setPromptTokens(Integer promptTokens) { this.promptTokens = promptTokens; }

        public Integer getCompletionTokens() { return completionTokens; }
        public void setCompletionTokens(Integer completionTokens) { this.completionTokens = completionTokens; }

        public Integer getTotalTokens() { return totalTokens; }
        public void setTotalTokens(Integer totalTokens) { this.totalTokens = totalTokens; }
    }

    // Constructors
    public CompletionResponse() {
        this.id = "cmpl-" + System.currentTimeMillis();
        this.created = System.currentTimeMillis() / 1000;
    }

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getObject() { return object; }
    public void setObject(String object) { this.object = object; }

    public Long getCreated() { return created; }
    public void setCreated(Long created) { this.created = created; }

    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }

    public List<Choice> getChoices() { return choices; }
    public void setChoices(List<Choice> choices) { this.choices = choices; }

    public Usage getUsage() { return usage; }
    public void setUsage(Usage usage) { this.usage = usage; }
}