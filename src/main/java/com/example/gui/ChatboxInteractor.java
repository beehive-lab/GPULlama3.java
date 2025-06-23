package com.example.gui;

public class ChatboxInteractor {

    private final ChatboxModel model;

    public ChatboxInteractor(ChatboxModel viewModel) {
        this.model = viewModel;
    }

    // TODO: Business logic for running inference and updating the output display
    public void runLlamaTornado() {
        String output = String.format("Engine: %s\nLlama3 Path: %s\nModel: %s\nPrompt: %s\n",
                model.getSelectedEngine(),
                model.getLlama3Path(),
                model.getSelectedModel(),
                model.getPromptText()
        );
        System.out.printf(output);
        model.setOutputText(output);
    }

}