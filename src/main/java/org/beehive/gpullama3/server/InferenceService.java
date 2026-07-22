package org.beehive.gpullama3.server;

import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Reusable, thread-safe inference wrapper over a single loaded {@link Model}.
 *
 * <p>Holds one {@link State} and one {@link TornadoVMMasterPlan} (the expensive, compile-once GPU
 * task graph) and serializes generation on them — the GPU is a single context, so requests run
 * one at a time. This is the reuse seam shared by the OpenAI server and any other front-end:
 * build the prompt, stream decoded tokens through a callback, return the full text. The single
 * proven single-token decode path is used verbatim (KV cache is overwritten from position 0 each
 * request, so state is safely reused without reallocation).</p>
 */
public final class InferenceService {

    private final Model model;
    private final State state;
    private final TornadoVMMasterPlan plan;
    private final boolean gpu;
    private final int initialToken;   // model start/BOS token that createNewState() seeds
    private final Object lock = new Object();

    public InferenceService(Model model, boolean gpu) {
        this.model = model;
        this.gpu = gpu;
        this.state = model.createNewState();
        // createNewState() seeds latestToken with the model's start token (BOS / header) — the
        // first forward pass consumes it. Capture it so each request can reset to a clean start.
        this.initialToken = state.latestToken;
        this.plan = gpu ? TornadoVMMasterPlan.initializeTornadoVMPlan(state, model) : null;
    }

    public Model model() {
        return model;
    }

    /** A single generation request. {@code messages} are role/content turns (chat); a lone
     *  user turn covers the completions endpoint. */
    public record Request(List<ChatFormat.Message> messages, int maxTokens, float temperature, float topP, long seed) {}

    /** Result of a generation: the text and the token counts (for {@code usage}). */
    public record Result(String text, int promptTokens, int completionTokens, boolean stopped) {}

    /**
     * Generate a completion. If {@code onToken} is non-null each decoded token's text is streamed
     * to it as it is produced; the full text is always returned. Thread-safe (serialized).
     */
    public Result generate(Request req, java.util.function.Consumer<String> onToken) {
        synchronized (lock) {
            ChatFormat cf = model.chatFormat();
            List<Integer> promptTokens = new ArrayList<>();
            if (model.shouldAddBeginOfText()) {
                promptTokens.add(cf.getBeginOfText());
            }
            for (ChatFormat.Message m : req.messages()) {
                promptTokens.addAll(cf.encodeMessage(m));
            }
            promptTokens.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            if (model.shouldIncludeReasoning()) {
                promptTokens.addAll(model.tokenizer().encode("<think>\n", model.tokenizer().getSpecialTokens().keySet()));
            }

            int maxTokens = req.maxTokens() > 0 ? promptTokens.size() + req.maxTokens() : 0;
            Sampler sampler = Sampler.selectSampler(model.configuration().vocabularySize(),
                    req.temperature(), req.topP(), req.seed());
            Set<Integer> stopTokens = cf.getStopTokens();

            StringBuilder full = new StringBuilder();
            IntConsumer tokenConsumer = token -> {
                if (model.tokenizer().shouldDisplayToken(token)) {
                    String piece = model.tokenizer().decode(List.of(token));
                    full.append(piece);
                    if (onToken != null) {
                        onToken.accept(piece);
                    }
                }
            };

            // Reset to the fresh-state convention: model start token + KV overwritten from pos 0.
            state.latestToken = initialToken;

            List<Integer> responseTokens = gpu
                    ? model.generateTokensGPU(state, 0, promptTokens, stopTokens, maxTokens, sampler, false, tokenConsumer, plan)
                    : model.generateTokens(state, 0, promptTokens, stopTokens, maxTokens, sampler, false, tokenConsumer);

            boolean stopped = !responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast());
            int completion = responseTokens.size() - (stopped ? 1 : 0);
            return new Result(full.toString(), promptTokens.size(), Math.max(0, completion), stopped);
        }
    }

    /** Free the GPU plan (call on server shutdown). */
    public void close() {
        if (plan != null) {
            plan.freeTornadoExecutionPlan();
        }
    }
}
