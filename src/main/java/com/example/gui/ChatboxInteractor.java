package com.example.gui;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ChatboxInteractor {

    private final ChatboxModel model;

    public ChatboxInteractor(ChatboxModel viewModel) {
        this.model = viewModel;
    }

    // Run the 'llama-tornado' script in the given Llama3 path, and stream its output to the GUI's output area.
    public void runLlamaTornado() {
        List<String> commands = new ArrayList<>();

        String llama3Path = model.getLlama3Path();
        if (llama3Path == null || llama3Path.isEmpty()) {
            model.setOutputText("Please set the Llama3 path:");
            return;
        }

        // Format for running 'llama-tornado' depends on the operating system.
        if (System.getProperty("os.name").startsWith("Windows")) {
            commands.add(String.format("%s\\external\\tornadovm\\.venv\\Scripts\\python", llama3Path));
            commands.add("llama-tornado");
        } else {
            commands.add(String.format("%s/.llama-tornado", llama3Path));
        }

        ChatboxModel.Engine engine = model.getSelectedEngine();
        if (engine == ChatboxModel.Engine.TORNADO_VM) {
            commands.add("--gpu");
        }

        // Assume that models are found in a /models directory.
        String selectedModel = model.getSelectedModel();
        if (selectedModel == null || selectedModel.isEmpty()) {
            model.setOutputText("Please select a model.");
            return;
        }
        String modelPath = String.format("%s/models/%s", llama3Path, selectedModel);
        String prompt = String.format("\"%s\"", model.getPromptText());

        commands.addAll(Arrays.asList(
                "--model", modelPath,
                "--prompt", prompt
        ));

        ProcessBuilder processBuilder = new ProcessBuilder(commands);
        processBuilder.redirectErrorStream(true);
        BufferedReader bufferedReader = null;
        Process process;
        try {
            process = processBuilder.start();
            bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder builder = new StringBuilder();

            // Make sure to output the raw command.
            builder.append("Running command: ");
            builder.append(String.join(" ", processBuilder.command().toArray(new String[0])));
            builder.append(System.getProperty("line.separator"));

            String line;
            while ((line = bufferedReader.readLine()) != null) {
                builder.append(line);
                builder.append(System.getProperty("line.separator"));
                final String currentOutput = builder.toString();
                javafx.application.Platform.runLater(() -> model.setOutputText(currentOutput));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (bufferedReader != null) {
                try {
                    bufferedReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}