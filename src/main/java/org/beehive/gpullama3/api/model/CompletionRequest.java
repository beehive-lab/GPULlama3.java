package org.beehive.gpullama3.api.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class CompletionRequest {
    private String model = "gpullama3";
    private String prompt;

    @JsonProperty("max_tokens")
    private Integer maxTokens = 100;

    private Double temperature = 0.7;

    @JsonProperty("top_p")
    private Double topP = 0.9;

    @JsonProperty("stop")
    private List<String> stopSequences;

    private Boolean stream = false;

    // Constructors
    public CompletionRequest() {}

    // Getters and Setters
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }

    public String getPrompt() { return prompt; }
    public void setPrompt(String prompt) { this.prompt = prompt; }

    public Integer getMaxTokens() { return maxTokens; }
    public void setMaxTokens(Integer maxTokens) { this.maxTokens = maxTokens; }

    public Double getTemperature() { return temperature; }
    public void setTemperature(Double temperature) { this.temperature = temperature; }

    public Double getTopP() { return topP; }
    public void setTopP(Double topP) { this.topP = topP; }

    public List<String> getStopSequences() { return stopSequences; }
    public void setStopSequences(List<String> stopSequences) { this.stopSequences = stopSequences; }

    public Boolean getStream() { return stream; }
    public void setStream(Boolean stream) { this.stream = stream; }

    @Override
    public String toString() {
        return "CompletionRequest{" +
                "model='" + model + '\'' +
                ", prompt='" + (prompt != null ? prompt.substring(0, Math.min(50, prompt.length())) + "..." : null) + '\'' +
                ", maxTokens=" + maxTokens +
                ", temperature=" + temperature +
                ", topP=" + topP +
                ", stream=" + stream +
                '}';
    }
}