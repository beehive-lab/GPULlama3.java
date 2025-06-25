package com.example.gui;

import atlantafx.base.theme.CupertinoDark;
import atlantafx.base.theme.PrimerDark;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class LlamaChatbox extends Application {

    @Override
    public void start(Stage stage) {
        Application.setUserAgentStylesheet(new CupertinoDark().getUserAgentStylesheet());
        ChatboxController controller = new ChatboxController();
        Scene scene = new Scene(controller.getView(), 800, 600);
        stage.setTitle("TornadoVM Chat");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}