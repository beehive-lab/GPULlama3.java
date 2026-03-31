package org.beehive.gpullama3.model.qwen2;

import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.Tokenizer;

public class DeepSeekR1Qwen extends Qwen2 {

    public DeepSeekR1Qwen(Qwen2Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(configuration, tokenizer, weights, chatFormat);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DEEPSEEK_R1_DISTILL_QWEN;
    }

    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }
}
