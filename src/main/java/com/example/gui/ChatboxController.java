package com.example.gui;

import javafx.concurrent.Task;
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
        Task<Void> inferenceTask = new Task<>() {
            @Override
            protected Void call() {
                interactor.runLlamaTornado();
                return null;
            }
        };
        inferenceTask.setOnSucceeded(evt -> {
            postRunAction.run();
        });
        Thread inferenceThread = new Thread(inferenceTask);
        inferenceThread.start();
    }

    public Region getView() {
        return viewBuilder.build();
    }

}