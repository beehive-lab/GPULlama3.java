package org.beehive.gpullama3.api.service;

import jakarta.annotation.PostConstruct;
import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import org.springframework.boot.ApplicationArguments;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.IntConsumer;

import static org.beehive.gpullama3.inference.sampler.Sampler.selectSampler;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

@Service
public class LLMService {

    private final ApplicationArguments args;

    private Options options;
    private Model model;

    public LLMService(ApplicationArguments args) {
        this.args = args;
    }

    @PostConstruct
    public void init() {
        try {
            System.out.println("Initializing LLM service...");

            // Step 1: Parse service options
            System.out.println("Step 1: Parsing service options...");
            options = Options.parseServiceOptions(args.getSourceArgs());
            System.out.println("Model path: " + options.modelPath());
            System.out.println("Context length: " + options.maxTokens());

            // Step 2: Load model weights
            System.out.println("\nStep 2: Loading model...");
            System.out.println("Loading model from: " + options.modelPath());
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true, options.useTornadovm());
            System.out.println("✓ Model loaded successfully");
            System.out.println("  Model type: " + model.getClass().getSimpleName());
            System.out.println("  Vocabulary size: " + model.configuration().vocabularySize());
            System.out.println("  Context length: " + model.configuration().contextLength());

            System.out.println("\n✓ Model service initialization completed successfully!");
            System.out.println("=== Ready to serve requests ===\n");

        } catch (Exception e) {
            System.err.println("✗ Failed to initialize model service: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Model initialization failed", e);
        }
    }

    /**
     * Generate response with default parameters.
     */
    public String generateResponse(String message, String systemMessage) {
        return generateResponse(message, systemMessage, 150, 0.7, 0.9);
    }

    public String generateResponse(String message, String systemMessage, int maxTokens, double temperature, double topP) {
        return generateResponse(message, systemMessage, maxTokens, temperature, topP, null);
    }

    public String generateResponse(String message, String systemMessage, int maxTokens, double temperature, double topP, Long seed) {
        try {
            // Create sampler and state like runInstructOnce
            long actualSeed = seed != null ? seed : System.currentTimeMillis();
            Sampler sampler = selectSampler(model.configuration().vocabularySize(), (float) temperature, (float) topP, actualSeed);
            State state = model.createNewState();

            // Use model's ChatFormat
            ChatFormat chatFormat = model.chatFormat();
            List<Integer> promptTokens = new ArrayList<>();

            // Add begin of text if needed
            if (model.shouldAddBeginOfText()) {
                promptTokens.add(chatFormat.getBeginOfText());
            }

            // Add system message properly formatted
            if (model.shouldAddSystemPrompt() && systemMessage != null && !systemMessage.trim().isEmpty()) {
                promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, systemMessage)));
            }

            // Add user message properly formatted
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, message)));
            promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

            // Handle reasoning tokens if needed (for Deepseek-R1-Distill-Qwen)
            if (model.shouldIncludeReasoning()) {
                List<Integer> thinkStartTokens = model.tokenizer().encode("<think>\n", model.tokenizer().getSpecialTokens().keySet());
                promptTokens.addAll(thinkStartTokens);
            }

            // Use proper stop tokens from chat format
            Set<Integer> stopTokens = chatFormat.getStopTokens();

            long startTime = System.currentTimeMillis();

            // Use CPU path for now (GPU path disabled as noted)
            List<Integer> generatedTokens = model.generateTokens(
                    state, 0, promptTokens, stopTokens, maxTokens, sampler, false, token -> {}
            );

            // Remove stop tokens if present
            if (!generatedTokens.isEmpty() && stopTokens.contains(generatedTokens.getLast())) {
                generatedTokens.removeLast();
            }

            long duration = System.currentTimeMillis() - startTime;
            double tokensPerSecond = generatedTokens.size() * 1000.0 / duration;
            System.out.printf("COMPLETED tokens=%d duration=%dms rate=%.1f tok/s%n",
                    generatedTokens.size(), duration, tokensPerSecond);

            String responseText = model.tokenizer().decode(generatedTokens);

            // Add reasoning prefix for non-streaming if needed
            if (model.shouldIncludeReasoning()) {
                responseText = "<think>\n" + responseText;
            }

            return responseText;

        } catch (Exception e) {
            System.err.println("FAILED " + e.getMessage());
            throw new RuntimeException("Failed to generate response", e);
        }
    }

    public void generateStreamingResponse(String message, String systemMessage, SseEmitter emitter,
            int maxTokens, double temperature, double topP, Long seed) {
        CompletableFuture.runAsync(() -> {
            try {
                long actualSeed = seed != null ? seed : System.currentTimeMillis();
                Sampler sampler = selectSampler(model.configuration().vocabularySize(), (float) temperature, (float) topP, actualSeed);
                State state = model.createNewState();

                // Use proper chat format like in runInstructOnce
                ChatFormat chatFormat = model.chatFormat();
                List<Integer> promptTokens = new ArrayList<>();

                if (model.shouldAddBeginOfText()) {
                    promptTokens.add(chatFormat.getBeginOfText());
                }

                if (model.shouldAddSystemPrompt() && systemMessage != null && !systemMessage.trim().isEmpty()) {
                    promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, systemMessage)));
                }

                promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, message)));
                promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

                // Include reasoning for Deepseek-R1-Distill-Qwen
                if (model.shouldIncludeReasoning()) {
                    List<Integer> thinkStartTokens = model.tokenizer().encode("<think>\n", model.tokenizer().getSpecialTokens().keySet());
                    promptTokens.addAll(thinkStartTokens);
                    // We are in streaming, immediately output the think start
                    emitter.send(SseEmitter.event().data("<think>\n"));
                }

                Set<Integer> stopTokens = chatFormat.getStopTokens();

                final int[] tokenCount = {0};
                IntConsumer tokenConsumer = token -> {
                    try {
                        // Only display tokens that should be displayed (like in your original)
                        if (model.tokenizer().shouldDisplayToken(token)) {
                            String tokenText = model.tokenizer().decode(List.of(token));
                            emitter.send(SseEmitter.event().data(tokenText));
                            tokenCount[0]++;
                        }
                    } catch (Exception e) {
                        emitter.completeWithError(e);
                    }
                };

                // Initialize TornadoVM plan once per request if GPU path is enabled
                TornadoVMMasterPlan tornadoVMPlan = null;
                if (options.useTornadovm()) {
                    tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
                }

                // Select execution path
                if (options.useTornadovm()) {
                    // GPU path
                    model.generateTokensGPU(state, 0, promptTokens, stopTokens, maxTokens, sampler, false, tokenConsumer, tornadoVMPlan);
                } else {
                    // CPU path
                    model.generateTokens(state, 0, promptTokens, stopTokens, maxTokens, sampler, false, tokenConsumer);
                }

                LastRunMetrics metrics = LastRunMetrics.getMetrics();
                double seconds = metrics.totalSeconds();
                int  tokens = metrics.totalTokens();
                double tokensPerSecond = tokens / seconds;
                System.out.printf("COMPLETED - [ achieved tok/s: %.2f. Tokens: %d, seconds: %.2f ]\n",
                        tokensPerSecond, tokens, seconds);

                // Send metrics as named event before [DONE]
                String metricsString = LastRunMetrics.getMetricsString();
                if (!metricsString.isEmpty()) {
                    emitter.send(SseEmitter.event().name("metrics").data(metricsString));
                }

                emitter.send(SseEmitter.event().data("[DONE]"));
                emitter.complete();

            } catch (Exception e) {
                System.err.println("FAILED " + e.getMessage());
                emitter.completeWithError(e);
            }
        });
    }

    // Getters for other services to access the initialized components
    public Options getOptions() {
        if (options == null) {
            throw new IllegalStateException("Model service not initialized yet");
        }
        return options;
    }

    public Model getModel() {
        if (model == null) {
            throw new IllegalStateException("Model service not initialized yet");
        }
        return model;
    }
}