package org.beehive.gpullama3.model;

import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.Tokenizer;

public abstract class AbstractModel implements Model {

    protected final Tokenizer tokenizer;
    protected final Weights weights;
    protected final ChatFormat chatFormat;

    protected AbstractModel(Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.chatFormat = chatFormat;
    }

    // Common methods across models

    public Weights weights() {
        return weights;
    }

    public ChatFormat chatFormat() {
        return chatFormat;
    }
}
