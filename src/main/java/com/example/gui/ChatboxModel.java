package com.example.gui;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

public class ChatboxModel {

    public enum Engine { TORNADO_VM, JVM }
    public enum Mode { INSTRUCT, INTERACTIVE }

    private final ObjectProperty<Engine> selectedEngine = new SimpleObjectProperty<>(Engine.TORNADO_VM);
    private final ObjectProperty<Mode> selectedMode = new SimpleObjectProperty<>(Mode.INSTRUCT);
    private final StringProperty selectedModel = new SimpleStringProperty("");
    private final StringProperty promptText = new SimpleStringProperty("");
    private final StringProperty outputText = new SimpleStringProperty("");
    private final BooleanProperty interactiveSessionActive = new SimpleBooleanProperty(false);

    public Engine getSelectedEngine() {
        return selectedEngine.get();
    }

    public ObjectProperty<Engine> selectedEngineProperty() {
        return selectedEngine;
    }

    public void setSelectedEngine(Engine engine) {
        this.selectedEngine.set(engine);
    }

    public Mode getSelectedMode() {
        return selectedMode.get();
    }

    public ObjectProperty<Mode> selectedModeProperty() {
        return selectedMode;
    }

    public void setSelectedMode(Mode mode) {
        this.selectedMode.set(mode);
    }

    public String getSelectedModel() {
        return selectedModel.get();
    }

    public StringProperty selectedModelProperty() {
        return selectedModel;
    }

    public void setSelectedModel(String selectedModel) {
        this.selectedModel.set(selectedModel);
    }

    public String getPromptText() {
        return promptText.get();
    }

    public StringProperty promptTextProperty() {
        return promptText;
    }

    public void setPromptText(String text) {
        this.promptText.set(text);
    }

    public String getOutputText() {
        return outputText.get();
    }

    public StringProperty outputTextProperty() {
        return outputText;
    }

    public void setOutputText(String text) {
        this.outputText.set(text);
    }

    public boolean getInteractiveSessionActive() {
        return interactiveSessionActive.get();
    }

    public BooleanProperty interactiveSessionActiveProperty() {
        return interactiveSessionActive;
    }

    public void setInteractiveSessionActive(boolean value) {
        this.interactiveSessionActive.set(value);
    }

}