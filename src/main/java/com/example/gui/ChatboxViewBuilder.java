package com.example.gui;

import javafx.beans.property.StringProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.SplitPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.util.Builder;

import java.util.function.Consumer;

public class ChatboxViewBuilder implements Builder<Region> {

    private final ChatboxModel model;
    private final Consumer<Runnable> actionHandler;

    public ChatboxViewBuilder(ChatboxModel model, Consumer<Runnable> actionHandler) {
        this.model = model;
        this.actionHandler = actionHandler;
    }

    @Override
    public Region build(){
        VBox results = new VBox();
        results.setPrefWidth(800);
        results.setPrefHeight(600);

        SplitPane panels = new SplitPane();
        VBox.setVgrow(panels, Priority.ALWAYS);
        results.getChildren().add(panels);

        panels.getItems().addAll(
                createLeftPanel(),
                createRightPanel()
        );

        return results;
    }

    private Node createLeftPanel() {
        VBox panel = new VBox(12);
        panel.setPadding(new Insets(24));
        panel.getChildren().addAll(
                createHeaderLabel("TornadoVM Chat"),
                createEngineBox(),
                createLlama3PathBox(),
                createModelSelectBox(),
                new Label("Prompt:"),
                createPromptBox(),
                createRunButton(),
                new Label("Output:"),
                createOutputArea()
        );
        return panel;
    }

    private Node createEngineBox() {
        ComboBox<ChatboxModel.Engine> engineDropdown = new ComboBox<>();
        engineDropdown.valueProperty().bindBidirectional(model.selectedEngineProperty());
        engineDropdown.getItems().addAll(ChatboxModel.Engine.values());
        engineDropdown.setMaxWidth(Double.MAX_VALUE);
        HBox box = new HBox(8, new Label("Engine:"), engineDropdown);
        box.setAlignment(Pos.CENTER_LEFT);
        return box;
    }

    private Node createLlama3PathBox() {
        Button browseButton = new Button("Browse");
        // TODO: Browse directory
        browseButton.setOnAction(e -> {
            System.out.println("Browse pressed!");
        });

        TextField pathField = boundTextField(model.llama3PathProperty());
        HBox box = new HBox(8, new Label("Llama3 Path:"), pathField, browseButton);
        box.setAlignment(Pos.CENTER_LEFT);
        HBox.setHgrow(pathField, Priority.ALWAYS);
        pathField.setMaxWidth(Double.MAX_VALUE);

        return box;
    }

    private Node createModelSelectBox() {
        ComboBox<String> modelDropdown = new ComboBox<>();
        modelDropdown.valueProperty().bindBidirectional(model.selectedModelProperty());
        // TODO: Update dropdown menu options when Llama3 path changes
        modelDropdown.getItems().addAll("Llama-3.2-1B-Instruct-Q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf"); // Hard-coded example strings for now
        HBox.setHgrow(modelDropdown, Priority.ALWAYS);
        modelDropdown.setMaxWidth(Double.MAX_VALUE);

        Button reloadButton = new Button("Reload");
        reloadButton.setOnAction(e -> {
            // TODO: Scan Llama3 path for models
            System.out.println("Reload pressed!");
        });

        HBox box = new HBox(8, new Label("Model:"), modelDropdown, reloadButton);
        box.setAlignment(Pos.CENTER_LEFT);
        return box;
    }

    private Node createPromptBox() {
        TextField promptField = boundTextField(model.promptTextProperty());
        HBox.setHgrow(promptField, Priority.ALWAYS);
        promptField.setMaxWidth(Double.MAX_VALUE);
        return new HBox(8, promptField);
    }

    private Node createRunButton() {
        Button runButton = new Button("Run");
        runButton.setMaxWidth(Double.MAX_VALUE);
        runButton.setOnAction(e -> {
            // TODO: Run inference
            System.out.println("Run pressed!");
            actionHandler.accept(() -> System.out.println("Finished running inference."));
        });
        return runButton;
    }

    private Node createOutputArea() {
        TextArea outputArea = new TextArea();
        outputArea.setEditable(false);
        outputArea.setWrapText(true);
        VBox.setVgrow(outputArea, Priority.ALWAYS);
        outputArea.textProperty().bind(model.outputTextProperty());
        return outputArea;
    }

    private Node createRightPanel() {
        VBox panel = new VBox(8);
        panel.setPadding(new Insets(24));
        panel.getChildren().addAll(
                createMonitorOutputArea(),
                createMonitorOptionsPanel()
        );
        return panel;
    }

    private TextArea createMonitorOutputArea() {
        TextArea textArea = new TextArea();
        textArea.setEditable(false);
        textArea.setWrapText(true);
        VBox.setVgrow(textArea, Priority.ALWAYS);
        return textArea;
    }

    private Node createMonitorOptionsPanel() {
        VBox box = new VBox();
        box.setPadding(new Insets(8));
        box.getChildren().addAll(
                createSubHeaderLabel("System Monitor"),
                createSystemMonitoringCheckBoxes()
        );
        return box;
    }

    private Node createSystemMonitoringCheckBoxes() {
        HBox checkBoxes = new HBox(8);
        checkBoxes.setAlignment(Pos.CENTER_LEFT);
        checkBoxes.getChildren().addAll(
                new CheckBox("htop"),
                new CheckBox("nvtop"),
                new CheckBox("GPU-Monitor")
        );
        return checkBoxes;
    }

    // Helper method for creating TextField objects with bound text property
    private TextField boundTextField(StringProperty boundProperty) {
        TextField textField = new TextField();
        textField.textProperty().bindBidirectional(boundProperty);
        return textField;
    }

    private Label createHeaderLabel(String text) {
        Label label = new Label(text);
        label.setStyle("-fx-font-size: 16pt; -fx-font-weight: bold;");
        return label;
    }

    private Label createSubHeaderLabel(String text) {
        Label label = new Label(text);
        label.setStyle("-fx-font-size: 12pt; -fx-font-weight: bold;");
        return label;
    }
}