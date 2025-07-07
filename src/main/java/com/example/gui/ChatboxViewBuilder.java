package com.example.gui;

import atlantafx.base.theme.Styles;
import javafx.beans.binding.Bindings;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.StringProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Control;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.util.Builder;

import java.io.File;
import java.util.function.Consumer;

public class ChatboxViewBuilder implements Builder<Region> {

    private static final int PANEL_WIDTH = 640;

    private final ChatboxModel model;
    private final BooleanProperty inferenceRunning = new SimpleBooleanProperty(false);
    private final Consumer<Runnable> actionHandler;

    public ChatboxViewBuilder(ChatboxModel model, Consumer<Runnable> actionHandler) {
        this.model = model;
        this.actionHandler = actionHandler;
    }

    @Override
    public Region build() {
        VBox results = new VBox();

        HBox panels = new HBox(createLeftPanel(), createRightPanel());
        VBox.setVgrow(panels, Priority.ALWAYS);
        results.getChildren().add(panels);

        return results;
    }

    private Node createLeftPanel() {
        VBox panel = new VBox(12);
        panel.setPrefWidth(PANEL_WIDTH);
        panel.setPadding(new Insets(24, 12, 24, 24));
        HBox.setHgrow(panel, Priority.ALWAYS);
        panel.getChildren().addAll(
                createHeaderLabel("TornadoVM Chat"),
                createEngineBox(),
                createChatModeBox(),
                createModelSelectBox(),
                createLabel("Prompt:"),
                createPromptBox(),
                createRunButton(),
                createLabel("Output:"),
                createOutputArea()
        );
        return panel;
    }

    private Node createEngineBox() {
        ComboBox<ChatboxModel.Engine> engineDropdown = new ComboBox<>();
        engineDropdown.disableProperty().bind(Bindings.createBooleanBinding(() -> (inferenceRunning.get() || model.getInteractiveSessionActive()),
                inferenceRunning,
                model.interactiveSessionActiveProperty()));
        engineDropdown.valueProperty().bindBidirectional(model.selectedEngineProperty());
        engineDropdown.getItems().addAll(ChatboxModel.Engine.values());
        engineDropdown.setMaxWidth(Double.MAX_VALUE);
        engineDropdown.setPrefWidth(152);
        HBox box = new HBox(8, createLabel("Engine:"), engineDropdown);
        box.setAlignment(Pos.CENTER_LEFT);
        return box;
    }

    private Node createChatModeBox() {
        ComboBox<ChatboxModel.Mode> modeDropdown = new ComboBox<>();
        modeDropdown.disableProperty().bind(Bindings.createBooleanBinding(() -> (inferenceRunning.get() || model.getInteractiveSessionActive()),
                inferenceRunning,
                model.interactiveSessionActiveProperty()));
        modeDropdown.valueProperty().bindBidirectional(model.selectedModeProperty());
        modeDropdown.getItems().addAll(ChatboxModel.Mode.values());
        modeDropdown.setMaxWidth(Double.MAX_VALUE);
        modeDropdown.setPrefWidth(152);
        HBox box = new HBox(8, createLabel("Chat:"), modeDropdown);
        box.setAlignment(Pos.CENTER_LEFT);
        return box;
    }

    private Node createModelSelectBox() {
        ComboBox<String> modelDropdown = new ComboBox<>();
        modelDropdown.disableProperty().bind(Bindings.createBooleanBinding(() -> (inferenceRunning.get() || model.getInteractiveSessionActive()),
                inferenceRunning,
                model.interactiveSessionActiveProperty()));
        modelDropdown.valueProperty().bindBidirectional(model.selectedModelProperty());
        HBox.setHgrow(modelDropdown, Priority.ALWAYS);
        modelDropdown.setMaxWidth(Double.MAX_VALUE);

        Button reloadButton = new Button("Reload");
        reloadButton.getStyleClass().add(Styles.ACCENT);
        reloadButton.setMinWidth(80);
        reloadButton.disableProperty().bind(Bindings.createBooleanBinding(() -> (inferenceRunning.get() || model.getInteractiveSessionActive()),
                inferenceRunning,
                model.interactiveSessionActiveProperty()));
        reloadButton.setOnAction(e -> {
            // Search for a /models folder containing model files.
            modelDropdown.getItems().clear();
            File llama3ModelsDir = new File("./models");
            if (llama3ModelsDir.exists() && llama3ModelsDir.isDirectory()) {
                File[] files = llama3ModelsDir.listFiles((dir, name) -> name.endsWith(".gguf"));
                if (files != null) {
                    for (File file : files) {
                        modelDropdown.getItems().add(file.getName());
                    }

                    int numModels = modelDropdown.getItems().size();
                    String message = String.format("Found %d %s in %s", numModels, (numModels == 1 ? "model" : "models"), llama3ModelsDir.toPath().normalize().toAbsolutePath());
                    String currentOutput = model.getOutputText();
                    if (currentOutput.isEmpty()) {
                        model.setOutputText(message);
                    } else {
                        model.setOutputText(String.format("%s\n%s", model.getOutputText(), message));
                    }

                    if (numModels == 0) {
                        modelDropdown.getSelectionModel().clearSelection();
                    } else {
                        modelDropdown.getSelectionModel().select(0);
                    }
                }
            }
        });

        reloadButton.fire(); // Trigger the reload once at the start.

        HBox box = new HBox(8, createLabel("Model:"), modelDropdown, reloadButton);
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
        runButton.getStyleClass().add(Styles.ACCENT);
        runButton.setMaxWidth(Double.MAX_VALUE);
        runButton.disableProperty().bind(inferenceRunning);
        runButton.setOnAction(e -> {
            inferenceRunning.set(true);
            actionHandler.accept(() -> inferenceRunning.set(false));
        });
        return runButton;
    }

    private Node createOutputArea() {
        TextArea outputArea = new TextArea();
        outputArea.setEditable(false);
        outputArea.setWrapText(true);
        VBox.setVgrow(outputArea, Priority.ALWAYS);
        model.outputTextProperty().subscribe((newValue) -> {
            outputArea.setText(newValue);
            // Autoscroll the text area to the bottom.
            outputArea.positionCaret(newValue.length());
        });
        return outputArea;
    }

    private Node createRightPanel() {
        VBox panel = new VBox(8);
        panel.setPrefWidth(PANEL_WIDTH);
        panel.setPadding(new Insets(24, 24, 24, 12));
        HBox.setHgrow(panel, Priority.ALWAYS);
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

    // Helper method to create Label objects with a minimum width
    private Label createLabel(String text) {
        Label label = new Label(text);
        label.setMinWidth(Control.USE_PREF_SIZE);
        return label;
    }

    private Label createHeaderLabel(String text) {
        Label label = createLabel(text);
        label.setStyle("-fx-font-size: 16pt; -fx-font-weight: bold;");
        return label;
    }

    private Label createSubHeaderLabel(String text) {
        Label label = createLabel(text);
        label.setStyle("-fx-font-size: 12pt; -fx-font-weight: bold;");
        return label;
    }
}