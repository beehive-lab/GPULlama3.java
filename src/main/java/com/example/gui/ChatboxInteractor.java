package com.example.gui;

import com.example.LlamaApp;
import com.example.Options;
import com.example.inference.sampler.Sampler;
import com.example.model.Model;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ChatboxInteractor {

    private final ChatboxModel model;
    private Model llm;
    private Sampler sampler;
    private Model.Response currentResponse;

    public ChatboxInteractor(ChatboxModel viewModel) {
        this.model = viewModel;
    }

    private String[] buildCommands() {
        List<String> commands = new ArrayList<>();

        ChatboxModel.Engine engine = model.getSelectedEngine();
        LlamaApp llamaApp = LlamaApp.getInstance();
        if (engine == ChatboxModel.Engine.TORNADO_VM) {
            llamaApp.setUseTornadoVM(true);
        } else {
            llamaApp.setUseTornadoVM(false);
        }

        ChatboxModel.Mode mode = model.getSelectedMode();
        if (mode == ChatboxModel.Mode.INTERACTIVE) {
            commands.add("--interactive");
        }

        // Assume that models are found in a /models directory.
        String selectedModel = model.getSelectedModel();
        if (selectedModel == null || selectedModel.isEmpty()) {
            model.setOutputText("Please select a model.");
            return null;
        }
        String modelPath = String.format("./models/%s", selectedModel);
        String prompt = String.format("\"%s\"", model.getPromptText());

        commands.addAll(Arrays.asList("--model", modelPath));
        if (!model.getInteractiveSessionActive()) {
            commands.addAll(Arrays.asList("--prompt", prompt));
        }

        return commands.toArray(new String[commands.size()]);
    }

    private void cleanTornadoVMResources() {
        if (currentResponse != null && currentResponse.tornadoVMPlan() != null) {
            try {
                currentResponse.tornadoVMPlan().freeTornadoExecutionPlan();
            } catch (Exception e) {
                System.err.println("Error while cleaning up TornadoVM resources: " + e.getMessage());
            }
        }
    }

    private void endInteractiveSession() {
        cleanTornadoVMResources();
        llm = null;
        currentResponse = null;
        model.setInteractiveSessionActive(false);
        System.out.println("Interactive session ended");
    }

    // Load and run a model while capturing its output text to a custom stream.
    public void runLlamaTornado() {
        // Save the original System.out stream
        PrintStream originalOut = System.out;
        try {
            String[] commands = buildCommands();
            if (commands == null) {
                // commands is null if no model was found, so exit this process
                return;
            }

            StringBuilder builder = new StringBuilder();

            // Create a custom PrintStream to capture output from loading the model and running it.
            PrintStream customStream = new PrintStream(new ByteArrayOutputStream()) {
                @Override
                public void println(String str) {
                    process(str + "\n");
                }

                @Override
                public void print(String str) {
                    process(str);
                }

                private void process(String str) {
                    // Capture the output stream into the GUI output area.
                    builder.append(str);
                    final String currentOutput = builder.toString();
                    javafx.application.Platform.runLater(() -> model.setOutputText(currentOutput));
                }
            };

            // Redirect System.out to the custom print stream.
            System.setOut(customStream);
            System.setErr(customStream);

            Options options = Options.parseOptions(commands);

            if (model.getInteractiveSessionActive()) {
                builder.append(model.getOutputText()); // Include the current output to avoid clearing the entire text.
                String userText = model.getPromptText();
                // Display the user message with a '>' prefix
                builder.append("> ");
                builder.append(userText);
                builder.append(System.getProperty("line.separator"));
                if (List.of("quit", "exit").contains(userText)) {
                    endInteractiveSession();
                } else {
                    currentResponse = llm.runInteractiveStep(sampler, options, userText, currentResponse);
                }
            } else {
                builder.append("Processing... please wait");
                builder.append(System.getProperty("line.separator"));

                // Load the model and run.
                llm = LlamaApp.loadModel(options);
                sampler = LlamaApp.createSampler(llm, options);
                if (options.interactive()) {
                    // Start a new interactive session.
                    builder.append("Interactive session started (write 'exit' or 'quit' to stop)");
                    builder.append(System.getProperty("line.separator"));
                    // Display the user message with a '>' prefix
                    builder.append("> ");
                    builder.append(model.getPromptText());
                    builder.append(System.getProperty("line.separator"));
                    currentResponse = llm.runInteractiveStep(sampler, options, model.getPromptText(), new Model.Response());
                    model.setInteractiveSessionActive(true);
                } else {
                    llm.runInstructOnce(sampler, options);
                    llm = null;
                    sampler = null;
                }
            }

        } catch (Exception e) {
            // Catch all exceptions so that they're logged in the output area.
            e.printStackTrace();
            e.printStackTrace(originalOut);
            // Make sure to end the interactive session if an exception occurs.
            if (model.getInteractiveSessionActive()) {
                endInteractiveSession();
            }
        } finally {
            System.setOut(originalOut);
        }
    }

}