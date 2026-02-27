# Integrations

This document showcases minimal integration examples demonstrating how to use GPULlama3.javaF with popular Java frameworks and AI orchestration libraries.

## LangChain4j Integration

**Repository**: [gpullama3-langchain4j-demo](https://github.com/beehive-lab/gpullama3-langchain4j-demo)

A pure Java demonstration of running LLaMA 3 language models on GPU through LangChain4j integration with TornadoVM for heterogeneous computing.

### Features

- Basic conversational AI interactions
- Memory-persistent multi-turn dialogues
- Real-time token streaming
- Agentic systems with tool-calling capabilities
- Comparative gameplay agents (CPU vs GPU)

### Requirements

- Java 21+
- Maven
- TornadoVM properly configured with GPU access
- ~20GB dedicated GPU memory for optimal performance

### Quick Start

```bash
# Build the project
mvn clean install

# Run with GPU acceleration (TornadoVM)
tornado --threadInfo --enableProfiler --printKernel -cp @cp.txt io.github.mikepapadim.YourDemo

# Run on CPU (for comparison)
java -Xmx20g -cp @cp.txt io.github.mikepapadim.YourDemo
```

### Performance

Benchmarks demonstrate consistent GPU advantages with speedups ranging from **3.5× to 5×** versus CPU execution across various model sizes using NVIDIA 5090 GPU.

---

## Quarkus + LangChain4j Integration

**Repository**: [gpullama3-quarkus-langchain4j-demo](https://github.com/beehive-lab/gpullama3-quarkus-langchain4j-demo)

Cloud-native demonstration combining Quarkus framework with LangChain4j extension for GPU-accelerated language model execution in containerized Java applications.

### Features

- **Chat Demo**: Interactive chat application with GPU acceleration
- **Streaming Demo**: Real-time streaming responses for conversational AI
- Quarkus-optimized deployment for cloud-native environments
- Hot-reload development mode

### Requirements

- Java 21 (suggested: 21.0.2-open)
- Maven
- TornadoVM with environment variables configured

### Quick Start

```bash
# Build all demo modules
mvn clean install

# Run the Chat Demo
java @$TORNADO_SDK/../../../tornado-argfile -jar demos/chat-demo/target/quarkus-app/quarkus-run.jar

# Run the Streaming Demo
java @$TORNADO_SDK/../../../tornado-argfile -jar demos/streaming-demo/target/quarkus-app/quarkus-run.jar

# Development mode (with hot reload)
mvn quarkus:dev
```

### Integration Benefits

- **Lightweight**: Quarkus minimizes memory footprint and startup time
- **Cloud-Ready**: Native Kubernetes integration and container-first design
- **Developer Experience**: Fast hot reload and unified configuration
- **AI-Optimized**: LangChain4j extension for seamless AI service integration

---

## Getting Started

Both integrations require:

1. **TornadoVM Setup**: Follow the [TornadoVM installation guide](https://github.com/beehive-lab/TornadoVM)
2. **GPU Drivers**: Ensure CUDA/OpenCL drivers are properly installed
3. **Model Files**: Download LLaMA 3 model files and configure paths

For detailed implementation examples, explore the respective repositories linked above.
