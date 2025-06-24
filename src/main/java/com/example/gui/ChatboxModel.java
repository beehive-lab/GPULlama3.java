package com.example.gui;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

public class ChatboxModel {

    public enum Engine { TORNADO_VM, JVM }

    private final ObjectProperty<Engine> selectedEngine = new SimpleObjectProperty<>(Engine.TORNADO_VM);
    private final StringProperty llama3Path = new SimpleStringProperty("");
    private final StringProperty selectedModel = new SimpleStringProperty("");
    private final StringProperty promptText = new SimpleStringProperty("");
    private final StringProperty outputText = new SimpleStringProperty("");

    public Engine getSelectedEngine() {
        return selectedEngine.get();
    }

    public ObjectProperty<Engine> selectedEngineProperty() {
        return selectedEngine;
    }

    public void setSelectedEngine(Engine engine) {
        this.selectedEngine.set(engine);
    }

    public String getLlama3Path() {
        return llama3Path.get();
    }

    public StringProperty llama3PathProperty() {
        return llama3Path;
    }

    public void setLlama3Path(String path) {
        this.llama3Path.set(path);
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

}