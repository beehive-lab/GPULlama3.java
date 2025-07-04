package com.example.model;

import com.example.LlamaApp;
import com.example.Options;
import com.example.auxiliary.LastRunMetrics;
import com.example.inference.InferenceEngine;
import com.example.inference.sampler.Sampler;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.model.format.ChatFormat;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.function.IntConsumer;

import static com.example.LlamaApp.SHOW_PERF_INTERACTIVE;

public interface Model {
    Configuration configuration();

    Tokenizer tokenizer();

    Weights weights();

    ModelType getModelType();

    State createNewState();

    State createNewState(int batchsize);

    /**
     * Model agnostic default implementation for interactive mode.
     * @param sampler
     * @param options
     */
    default void runInteractive(Sampler sampler, Options options) {
        State state = null;
        List<Integer> conversationTokens = new ArrayList<>();

        ChatFormat chatFormat = ChatFormat.create(tokenizer());
        conversationTokens.add(chatFormat.getBeginOfText());

        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }

        int startPosition = 0;
        Scanner in = new Scanner(System.in);

        // Initialize TornadoVM plan once at the beginning if GPU path is enabled
        TornadoVMMasterPlan tornadoVMPlan = null;

        // Get the LlamaApp singleton to read configuration values
        LlamaApp llamaApp = LlamaApp.getInstance();

        try {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = in.nextLine();
                if (List.of("quit", "exit").contains(userText)) {
                    break;
                }
                if (state == null) {
                    // State allocation can take some time for large context sizes,
                    // allocate the model state only after printing the user '>' prompt.
                    state = createNewState();
                }

                if (llamaApp.getUseTornadoVM() && tornadoVMPlan == null) {
                    tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, this);
                }

                conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
                conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
                Set<Integer> stopTokens = chatFormat.getStopTokens();

                List<Integer> responseTokens;
                IntConsumer tokenConsumer = token -> {
                    if (options.stream()) {
                        if (tokenizer().shouldDisplayToken(token)) {
                            System.out.print(tokenizer().decode(List.of(token)));
                        }
                    }
                };

                // Choose between GPU and CPU path based on configuration
                if (llamaApp.getUseTornadoVM()) {
                    // GPU path using TornadoVM
                    responseTokens = InferenceEngine.generateTokensGPU(this, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens,
                            options.maxTokens(), sampler, options.echo(), options.stream() ? tokenConsumer : null, tornadoVMPlan);
                } else {
                    // CPU path
                    responseTokens = InferenceEngine.generateTokens(this, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(),
                            sampler, options.echo(), tokenConsumer);
                }

                // Include stop token in the prompt history, but not in the response displayed to the user.
                conversationTokens.addAll(responseTokens);
                startPosition = conversationTokens.size();
                Integer stopToken = null;
                if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                    stopToken = responseTokens.getLast();
                    responseTokens.removeLast();
                }
                if (!options.stream()) {
                    String responseText = tokenizer().decode(responseTokens);
                    System.out.println(responseText);
                }
                if (stopToken == null) {
                    System.err.println("\n Ran out of context length...\n Increase context length with by passing to llama-tornado --max-tokens XXX");
                    break;
                }
                System.out.print("\n");

                // Optionally print performance metrics after each response
                if (SHOW_PERF_INTERACTIVE) {
                    LastRunMetrics.printMetrics();
                }
            }
        } finally {
            // Clean up TornadoVM resources when exiting the chat loop
            if (llamaApp.getUseTornadoVM() && tornadoVMPlan != null) {
                try {
                    tornadoVMPlan.freeTornadoExecutionPlan();
                } catch (Exception e) {
                    System.err.println("Error while cleaning up TornadoVM resources: " + e.getMessage());
                }
            }
        }
    }

    /**
     * Model agnostic default implementation for instruct mode.
     * @param sampler
     * @param options
     */
    default void runInstructOnce(Sampler sampler, Options options) {
        State state = createNewState();
        ChatFormat chatFormat = ChatFormat.create(tokenizer());
        TornadoVMMasterPlan tornadoVMPlan = null;
        LlamaApp llamaApp = LlamaApp.getInstance();

        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfText());

        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        List<Integer> responseTokens;

        IntConsumer tokenConsumer = token -> {
            if (options.stream()) {
                if (tokenizer().shouldDisplayToken(token)) {
                    System.out.print(tokenizer().decode(List.of(token)));
                }
            }
        };

        Set<Integer> stopTokens = chatFormat.getStopTokens();

        if (llamaApp.getUseTornadoVM()) {
            tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, this);
            // Call generateTokensGPU without the token consumer parameter
            responseTokens = InferenceEngine.generateTokensGPU(this, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), options.stream() ? tokenConsumer : null,
                    tornadoVMPlan);
        } else {
            responseTokens = InferenceEngine.generateTokens(this, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }

        LastRunMetrics.printMetrics();

        if (tornadoVMPlan != null) {
            tornadoVMPlan.freeTornadoExecutionPlan();
        }
    }
}
