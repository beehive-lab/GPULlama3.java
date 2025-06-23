package com.example.gui;

import javafx.scene.layout.Region;

public class ChatboxController {

    private final ChatboxViewBuilder viewBuilder;
    private final ChatboxInteractor interactor;

    public ChatboxController() {
        ChatboxModel model = new ChatboxModel();
        interactor = new ChatboxInteractor(model);
        viewBuilder = new ChatboxViewBuilder(model, this::runInference);
    }

    private void runInference(Runnable postRunAction) {
        // TODO: Run llama tornado
        System.out.println("Starting LLM inference.");
        interactor.runLlamaTornado();
        postRunAction.run();
    }

    public Region getView() {
        return viewBuilder.build();
    }

}