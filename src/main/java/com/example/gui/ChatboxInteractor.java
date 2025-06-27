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

    public ChatboxInteractor(ChatboxModel viewModel) {
        this.model = viewModel;
    }

    private String[] buildCommands() {
        List<String> commands = new ArrayList<>();

        ChatboxModel.Engine engine = model.getSelectedEngine();
        if (engine == ChatboxModel.Engine.TORNADO_VM) {
            // TODO: LlamaApp.USE_TORNADOVM is a final constant, but the GUI needs to be able to set this value
            //commands.add("--gpu");
        }

        // Assume that models are found in a /models directory.
        String selectedModel = model.getSelectedModel();
        if (selectedModel == null || selectedModel.isEmpty()) {
            model.setOutputText("Please select a model.");
            return null;
        }
        String modelPath = String.format("./models/%s", selectedModel);
        String prompt = String.format("\"%s\"", model.getPromptText());

        commands.addAll(Arrays.asList(
                "--model", modelPath,
                "--prompt", prompt
        ));

        return commands.toArray(new String[commands.size()]);
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

            builder.append("Processing... please wait");
            builder.append(System.getProperty("line.separator"));

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

            // Load the model and run.
            Model llm = LlamaApp.loadModel(options);
            Sampler sampler = LlamaApp.createSampler(llm, options);
            if (options.interactive()) {
                llm.runInteractive(sampler, options);
            } else {
                llm.runInstructOnce(sampler, options);
            }
        } catch (Exception e) {
            // Catch all exceptions so that they're logged in the output area.
            e.printStackTrace();
            e.printStackTrace(originalOut);
        } finally {
            System.setOut(originalOut);
        }
    }

}