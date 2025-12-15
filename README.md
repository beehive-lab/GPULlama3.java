# GPULlama3.java powered by TornadoVM [![build JDK21](https://github.com/beehive-lab/GPULlama3.java/actions/workflows/build-and-run.yml/badge.svg)](https://github.com/beehive-lab/GPULlama3.java/actions/workflows/build-and-run.yml) [![Maven Central](https://img.shields.io/maven-central/v/io.github.beehive-lab/gpu-llama3?&logo=apache-maven&color=blue)](https://central.sonatype.com/artifact/io.github.beehive-lab/gpu-llama3)


![Java Version](https://img.shields.io/badge/java-21+-blue?style=&logo=openjdk)
[![LangChain4j](https://img.shields.io/badge/LangChain4j-1.7.1+-purple?&logo=link&logoColor=white)](https://docs.langchain4j.dev/)
![OpenCL](https://img.shields.io/badge/OpenCL-supported-blue?style=&logo=khronos)
![CUDA](https://img.shields.io/badge/CUDA/PTX-supported-76B900?style=&logo=nvidia)
[![Docker OpenCL](https://img.shields.io/badge/Docker-OpenCL-2496ED?&logo=docker&logoColor=white)](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-opencl)
[![Docker PTX](https://img.shields.io/badge/Docker-PTX-2496ED?&logo=docker&logoColor=white)](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-ptx)
[![GPULlama3.java DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/beehive-lab/GPULlama3.java)

-----------
<table style="border: none;">
<tr style="border: none;">
<td style="width: 40%; vertical-align: middle; border: none;">
<img src="docs/ll.gif" >
</td>
<td style="vertical-align: middle; padding-left: 20px; border: none;">  
<strong>Llama3</strong> models written in <strong>native Java</strong> automatically accelerated on GPUs with <a href="https://github.com/beehive-lab/TornadoVM" target="_blank"><strong>TornadoVM</strong></a>.
Runs Llama3 inference efficiently using TornadoVM's GPU acceleration.
<br><br>
Currently, supports <strong>Llama3</strong>, <strong>Mistral</strong>, <strong>Qwen2.5</strong>, <strong>Qwen3</strong> and <strong>Phi3</strong> models in the GGUF format.
Also, it is used as GPU inference engine in 
<a href="https://docs.quarkiverse.io/quarkus-langchain4j/dev/gpullama3-chat-model.html" target="_blank">Quarkus</a> 
and 
<a href="https://docs.langchain4j.dev/integrations/language-models/gpullama3-java" target="_blank">LangChain4J</a>.
<br><br>
Builds on <a href="https://github.com/mukel/llama3.java">Llama3.java</a> by <a href="https://github.com/mukel">Alfonso² Peterssen</a>.
Previous integration of TornadoVM and Llama2 it can be found in <a href="https://github.com/mikepapadim/llama2.tornadovm.java">llama2.tornadovm</a>.
</td>
</tr>
</table>

-----------
## <img src="https://github.com/user-attachments/assets/51b76554-0b01-4e18-a567-600901ab8c5f" alt="LangChain4j" height="38" style="vertical-align: middle; margin-right: 8px;"> Integration with LangChain4j

Since **LangChain4j v1.7.1**, `GPULlama3.java` is officially supported as a **model provider**.  
This means you can directly use *GPULlama3.java* inside your LangChain4j applications without extra glue code, just run on your GPU.

📖 Learn more: [LangChain4j Documentation](https://docs.langchain4j.dev/)

[Example agentic workflows with GPULlama3.java + LangChain4j 🚀](https://github.com/mikepapadim/devoxx25-demo-gpullama3-langchain4j/tree/main)

How to use:
```java
GPULlama3ChatModel model = GPULlama3ChatModel.builder()
        .modelPath(modelPath)
        .temperature(0.9)       // more creative
        .topP(0.9)              // more variety
        .maxTokens(2048)
        .onGPU(Boolean.TRUE) // if false, runs on CPU though a lightweight implementation of llama3.java
        .build();
```
-----------
#### **[Interactive-mode]** Running on a RTX 5090 with nvtop on bottom to track GPU utilization and memory usage.

![Demo](docs/inter-output.gif)

-----------

## Setup & Configuration

### Prerequisites

Ensure you have the following installed and configured:

- **Java 21**: Required for Vector API support & TornadoVM.
- [TornadoVM](https://github.com/beehive-lab/TornadoVM) with OpenCL or PTX backends.
- GCC/G++ 13 or newer: Required to build and run TornadoVM native components.

### Install, Build, and Run

When cloning this repository, use the `--recursive` flag to ensure that TornadoVM is properly included as submodule:

```bash
# Clone the repository with all submodules
git clone https://github.com/beehive-lab/GPULlama3.java.git
```

#### Install the TornadoVM SDK on Linux or macOS

Ensure that your JAVA_HOME points to a supported JDK before using the SDK. Download an SDK package matching your OS, architecture, and accelerator backend (opencl, ptx).
All pre-built SDKs are available on the TornadoVM [Releases Page](https://github.com/beehive-lab/TornadoVM/releases).
#After extracting the SDK, add its bin/ directory to your PATH so the `tornado` command becomes available.

##### Linux (x86_64)

```bash
wget https://github.com/beehive-lab/TornadoVM/releases/download/v2.1.0/tornadovm-2.1.0-opencl-linux-amd64.zip
unzip tornadovm-2.1.0-opencl-linux-amd64.zip
# Replace <path-to-sdk> manually with the absolute path of the extracted folder
export TORNADO_SDK="<path-to-sdk>/tornadovm-2.1.0-opencl"
export PATH=$TORNADO_SDK/bin:$PATH

tornado --devices
tornado --version
```

##### macOS (Apple Silicon)

```bash
wget https://github.com/beehive-lab/TornadoVM/releases/download/v2.1.0/tornadovm-2.1.0-opencl-mac-aarch64.zip
unzip tornadovm-2.1.0-opencl-mac-aarch64.zip
# Replace <path-to-sdk> manually with the absolute path of the extracted folder
export TORNADO_SDK="<path-to-sdk>/tornadovm-2.1.0-opencl"
export PATH=$TORNADO_SDK/bin:$PATH

tornado --devices
tornado --version
```

#### Build the GPULlama3.java

```bash
# Navigate to the project directory
cd GPULlama3.java

# Source the project-specific environment paths -> this will ensure the correct paths are set for the project and the TornadoVM SDK
# Expect to see: [INFO] Environment configured for Llama3 with TornadoVM at: $TORNADO_SDK
source set_paths

# Build the project using Maven (skip tests for faster build)
# mvn clean package -DskipTests or just make
make

# Run the model (make sure you have downloaded the model file first -  see below)
./llama-tornado --gpu  --verbose-init --opencl --model beehive-llama-3.2-1b-instruct-fp16.gguf --prompt "tell me a joke"
```


----------

### TornadoVM-Accelerated Inference Performance and Optimization Status

We are at the early stages of Java entering the AI world with features added to the JVM that enable faster execution such as GPU acceleration, Vector acceleration, high-performance access to off-heap memory and others.
<br><br>This repository provides the first Java-native implementation of Llama3 that automatically compiles and executes Java code on GPUs via TornadoVM. 
The baseline numbers presented below provide a solid starting point for achieving more competitive performance compared to llama.cpp or native CUDA implementations. 
[Our roadmap](https://github.com/beehive-lab/GPULlama3.java/blob/main/docs/GPULlama3_ROADMAP.md) provides the upcoming set of features that will dramatically improve the numbers below with the clear target being to achieve performance parity with the fastest implementations. 
<br><br>
If you achieve additional performance data points (e.g. new hardware or platforms) please let us know to add them below. 
<br><br>
In addition, if you are interested to learn more about the challenges of managed programming languages and GPU acceleration, you can read [our book](https://link.springer.com/book/10.1007/978-3-031-49559-5) or consult the [TornadoVM educational pages](https://www.tornadovm.org/resources). 


| Vendor / Backend             | Hardware     | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Optimizations |
|:----------------------------:|:------------:|:---------------------:|:---------------------:|:-------------:|
|                              |              |       **FP16**        |       **FP16**        |  **Support**  |
| **NVIDIA / OpenCL-PTX**      | RTX 3070     |      66 tokens/s      |    55.46 tokens/s     |       ✅      |
|                              | RTX 4090     |    86.11 tokens/s     |    75.32 tokens/s     |       ✅      |
|                              | RTX 5090     |    117.65 tokens/s    |    112.68 tokens/s    |       ✅      |
|                              | L4 Tensor    |    52.96 tokens/s     |    22.68 tokens/s     |       ✅      |
| **Intel / OpenCL**           | Arc A770     |    15.65 tokens/s     |     7.02 tokens/s     |      (WIP)    |
| **Apple Silicon / OpenCL**   | M3 Pro       |    14.04 tokens/s     |     6.78 tokens/s     |      (WIP)    |
|                              | M4 Pro       |    16.77 tokens/s     |     8.56 tokens/s     |      (WIP)    |
| **AMD / OpenCL**             | Radeon RX    |         (WIP)         |         (WIP)         |      (WIP)    |

##### ⚠️ Note on Apple Silicon Performance

TornadoVM currently runs on Apple Silicon via [OpenCL](https://developer.apple.com/opencl/), which has been officially deprecated since macOS 10.14.

Despite being deprecated, OpenCL can still run on Apple Silicon; albeit, with older drivers which do not support all optimizations of TornadoVM. Therefore, the performance is not optimal since TornadoVM does not have a Metal backend yet (it currently has OpenCL, PTX, and SPIR-V backends). We recommend using Apple silicon for development and for performance testing to use OpenCL/PTX compatible Nvidia GPUs for the time being (until we add a Metal backend to TornadoVM and start optimizing it).

-----------
## 📦 Maven Dependency

You can add **GPULlama3.java** directly to your Maven project by including the following dependency in your `pom.xml`:

```xml
<dependency>
    <groupId>io.github.beehive-lab</groupId>
    <artifactId>gpu-llama3</artifactId>
    <version>0.3.1</version>
</dependency>
```

## ☕ Integration with Your Java Codebase or Tools

To integrate it into your codebase or IDE (e.g., IntelliJ) or custom build system (like IntelliJ, Maven, or Gradle), use the `--show-command` flag.
This flag shows the exact Java command with all JVM flags that are being invoked under the hood to enable seamless execution on GPUs with TornadoVM.
Hence, it makes it simple to replicate or embed the invoked flags in any external tool or codebase.

```bash
llama-tornado --gpu --model beehive-llama-3.2-1b-instruct-fp16.gguf --prompt "tell me a joke" --show-command
```

<details>
<summary>📋 Click to see the JVM configuration </summary>

```java
/home/mikepapadim/.sdkman/candidates/java/current/bin/java \
    -server \
    -XX:+UnlockExperimentalVMOptions \
    -XX:+EnableJVMCI \
    -Xms20g -Xmx20g \
    --enable-preview \
    -Djava.library.path=/home/mikepapadim/manchester/TornadoVM/bin/sdk/lib \
    -Djdk.module.showModuleResolution=false \
    --module-path .:/home/mikepapadim/manchester/TornadoVM/bin/sdk/share/java/tornado \
    -Dtornado.load.api.implementation=uk.ac.manchester.tornado.runtime.tasks.TornadoTaskGraph \
    -Dtornado.load.runtime.implementation=uk.ac.manchester.tornado.runtime.TornadoCoreRuntime \
    -Dtornado.load.tornado.implementation=uk.ac.manchester.tornado.runtime.common.Tornado \
    -Dtornado.load.annotation.implementation=uk.ac.manchester.tornado.annotation.ASMClassVisitor \
    -Dtornado.load.annotation.parallel=uk.ac.manchester.tornado.api.annotations.Parallel \
    -Dtornado.tvm.maxbytecodesize=65536 \
    -Duse.tornadovm=true \
    -Dtornado.threadInfo=false \
    -Dtornado.debug=false \
    -Dtornado.fullDebug=false \
    -Dtornado.printKernel=false \
    -Dtornado.print.bytecodes=false \
    -Dtornado.device.memory=7GB \
    -Dtornado.profiler=false \
    -Dtornado.log.profiler=false \
    -Dtornado.profiler.dump.dir=/home/mikepapadim/repos/gpu-llama3.java/prof.json \
    -Dtornado.enable.fastMathOptimizations=true \
    -Dtornado.enable.mathOptimizations=false \
    -Dtornado.enable.nativeFunctions=fast \
    -Dtornado.loop.interchange=true \
    -Dtornado.eventpool.maxwaitevents=32000 \
    "-Dtornado.opencl.compiler.flags=-cl-denorms-are-zero -cl-no-signed-zeros -cl-finite-math-only" \
    --upgrade-module-path /home/mikepapadim/manchester/TornadoVM/bin/sdk/share/java/graalJars \
    @/home/mikepapadim/manchester/TornadoVM/bin/sdk/etc/exportLists/common-exports \
    @/home/mikepapadim/manchester/TornadoVM/bin/sdk/etc/exportLists/opencl-exports \
    --add-modules ALL-SYSTEM,tornado.runtime,tornado.annotation,tornado.drivers.common,tornado.drivers.opencl \
    -cp /home/mikepapadim/repos/gpu-llama3.java/target/gpu-llama3-1.0-SNAPSHOT.jar \
    org.beehive.gpullama3.LlamaApp \
    -m beehive-llama-3.2-1b-instruct-fp16.gguf \
    --temperature 0.1 \
    --top-p 0.95 \
    --seed 1746903566 \
    --max-tokens 512 \
    --stream true \
    --echo false \
    -p "tell me a joke" \
    --instruct
```

</details>

-----------

The above model can we swapped with one of the other models, such as `beehive-llama-3.2-3b-instruct-fp16.gguf` or `beehive-llama-3.2-8b-instruct-fp16.gguf`, depending on your needs.
Check models below.

## Collection of Tested Models

### Llama3.2 Collection 
[https://huggingface.co/collections/beehive-lab/llama3-gpullama3java](https://huggingface.co/collections/beehive-lab/llama3-gpullama3java)

### Qwen 2.5 Collection 
[https://huggingface.co/collections/beehive-lab/qwen-25-gpullama3java](https://huggingface.co/collections/beehive-lab/qwen-25-gpullama3java)

### Qwen 3 Collection 
[https://huggingface.co/collections/beehive-lab/llama3-gpullama3java](https://huggingface.co/collections/beehive-lab/qwen-3-gpullama3java)

### Phi-3 Collection 
[https://huggingface.co/collections/beehive-lab/llama3-gpullama3java](https://huggingface.co/collections/beehive-lab/phi-3-gpullama3java)

### Mistral Collection 
[https://huggingface.co/collections/beehive-lab/llama3-gpullama3java](https://huggingface.co/collections/beehive-lab/mistral-gpullama3java)

### DeepSeek-R1-Distill-Qwen Collection 
[https://huggingface.co/collections/beehive-lab/deepseek-r1-distill-qwen-gpullama3java](https://huggingface.co/collections/beehive-lab/deepseek-r1-distill-qwen-gpullama3java)

-----------

## Running `llama-tornado`

To execute Llama3, or Mistral models with TornadoVM on GPUs use the `llama-tornado` script with the `--gpu` flag.

### Usage Examples

#### Basic Inference
Run a model with a text prompt:

```bash
./llama-tornado --gpu --verbose-init --opencl --model beehive-llama-3.2-1b-instruct-fp16.gguf --prompt "Explain the benefits of GPU acceleration."
```

#### GPU Execution (FP16 Model)
Enable GPU acceleration with Q8_0 quantization:
```bash
./llama-tornado --gpu  --verbose-init --model beehive-llama-3.2-1b-instruct-fp16.gguf --prompt "tell me a joke"
```

-----------

## 🐳 Docker

You can run `GPULlama3.java` fully containerized with GPU acceleration enabled via **OpenCL** or **PTX** using pre-built Docker images.
More information as well as examples to run with the containers are available at [docker-gpullama3.java](https://github.com/beehive-lab/docker-gpullama3.java).

### 📦 Available Docker Images

| Backend | Docker Image | Pull Command |
|--------|---------------|---------------|
| **OpenCL** | [`beehivelab/gpullama3.java-nvidia-openjdk-opencl`](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-opencl) | `docker pull beehivelab/gpullama3.java-nvidia-openjdk-opencl` |
| **PTX (CUDA)** | [`beehivelab/gpullama3.java-nvidia-openjdk-ptx`](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-ptx) | `docker pull beehivelab/gpullama3.java-nvidia-openjdk-ptx` |

#### Example (OpenCL)

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/data \
  beehivelab/gpullama3.java-nvidia-openjdk-opencl \
  /gpullama3/GPULlama3.java/llama-tornado \
  --gpu --verbose-init \
  --opencl \
  --model /data/Llama-3.2-1B-Instruct.FP16.gguf \
  --prompt "Tell me a joke"
```
-----------

## Troubleshooting GPU Memory Issues

### Out of Memory Error

You may encounter an out-of-memory error like:
```
Exception in thread "main" uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException: Unable to allocate 100663320 bytes of memory.
To increase the maximum device memory, use -Dtornado.device.memory=<X>GB
```

This indicates that the default GPU memory allocation (7GB) is insufficient for your model.

### Solution

First, check your GPU specifications. If your GPU has high memory capacity, you can increase the GPU memory allocation using the `--gpu-memory` flag:

```bash
# For 3B models, try increasing to 15GB
./llama-tornado --gpu --model beehive-llama-3.2-3b-instruct-fp16.gguf --prompt "Tell me a joke" --gpu-memory 15GB

# For 8B models, you may need even more (20GB or higher)
./llama-tornado --gpu --model beehive-llama-3.2-8b-instruct-fp16.gguf --prompt "Tell me a joke" --gpu-memory 20GB
```

### GPU Memory Requirements by Model Size

| Model Size  | Recommended GPU Memory |
|-------------|------------------------|
| 1B models   | 7GB (default)          |
| 3-7B models | 15GB                   |
| 8B models   | 20GB+                  |

**Note**: If you still encounter memory issues, try:

1. Using Q4_0 instead of Q8_0 quantization (requires less memory).
2. Closing other GPU-intensive applications in your system.

-----------

## Command Line Options

Supported command-line options include:

```bash
cmd ➜ llama-tornado --help
usage: llama-tornado [-h] --model MODEL_PATH [--prompt PROMPT] [-sp SYSTEM_PROMPT] [--temperature TEMPERATURE] [--top-p TOP_P] [--seed SEED] [-n MAX_TOKENS]
                     [--stream STREAM] [--echo ECHO] [-i] [--instruct] [--gpu] [--opencl] [--ptx] [--gpu-memory GPU_MEMORY] [--heap-min HEAP_MIN] [--heap-max HEAP_MAX]
                     [--debug] [--profiler] [--profiler-dump-dir PROFILER_DUMP_DIR] [--print-bytecodes] [--print-threads] [--print-kernel] [--full-dump]
                     [--show-command] [--execute-after-show] [--opencl-flags OPENCL_FLAGS] [--max-wait-events MAX_WAIT_EVENTS] [--verbose]

GPU-accelerated LLaMA.java model runner using TornadoVM

options:
  -h, --help            show this help message and exit
  --model MODEL_PATH    Path to the LLaMA model file (e.g., beehive-llama-3.2-8b-instruct-fp16.gguf) (default: None)

LLaMA Configuration:
  --prompt PROMPT       Input prompt for the model (default: None)
  -sp SYSTEM_PROMPT, --system-prompt SYSTEM_PROMPT
                        System prompt for the model (default: None)
  --temperature TEMPERATURE
                        Sampling temperature (0.0 to 2.0) (default: 0.1)
  --top-p TOP_P         Top-p sampling parameter (default: 0.95)
  --seed SEED           Random seed (default: current timestamp) (default: None)
  -n MAX_TOKENS, --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate (default: 512)
  --stream STREAM       Enable streaming output (default: True)
  --echo ECHO           Echo the input prompt (default: False)
  --suffix SUFFIX       Suffix for fill-in-the-middle request (Codestral) (default: None)

Mode Selection:
  -i, --interactive     Run in interactive/chat mode (default: False)
  --instruct            Run in instruction mode (default) (default: True)

Hardware Configuration:
  --gpu                 Enable GPU acceleration (default: False)
  --opencl              Use OpenCL backend (default) (default: None)
  --ptx                 Use PTX/CUDA backend (default: None)
  --gpu-memory GPU_MEMORY
                        GPU memory allocation (default: 7GB)
  --heap-min HEAP_MIN   Minimum JVM heap size (default: 20g)
  --heap-max HEAP_MAX   Maximum JVM heap size (default: 20g)

Debug and Profiling:
  --debug               Enable debug output (default: False)
  --profiler            Enable TornadoVM profiler (default: False)
  --profiler-dump-dir PROFILER_DUMP_DIR
                        Directory for profiler output (default: /home/mikepapadim/repos/gpu-llama3.java/prof.json)

TornadoVM Execution Verbose:
  --print-bytecodes     Print bytecodes (tornado.print.bytecodes=true) (default: False)
  --print-threads       Print thread information (tornado.threadInfo=true) (default: False)
  --print-kernel        Print kernel information (tornado.printKernel=true) (default: False)
  --full-dump           Enable full debug dump (tornado.fullDebug=true) (default: False)
  --verbose-init        Enable timers for TornadoVM initialization (llama.EnableTimingForTornadoVMInit=true) (default: False)

Command Display Options:
  --show-command        Display the full Java command that will be executed (default: False)
  --execute-after-show  Execute the command after showing it (use with --show-command) (default: False)

Advanced Options:
  --opencl-flags OPENCL_FLAGS
                        OpenCL compiler flags (default: -cl-denorms-are-zero -cl-no-signed-zeros -cl-finite-math-only)
  --max-wait-events MAX_WAIT_EVENTS
                        Maximum wait events for TornadoVM event pool (default: 32000)
  --verbose, -v         Verbose output (default: False)

```

## Debug & Profiling Options
View TornadoVM's internal behavior:
```bash
# Print thread information during execution
./llama-tornado --gpu --model model.gguf --prompt "..." --print-threads

# Show bytecode compilation details
./llama-tornado --gpu --model model.gguf --prompt "..." --print-bytecodes

# Display generated GPU kernel code
./llama-tornado --gpu --model model.gguf --prompt "..." --print-kernel

# Enable full debug output with all details
./llama-tornado --gpu --model model.gguf --prompt "..." --debug --full-dump

# Combine debug options
./llama-tornado --gpu --model model.gguf --prompt "..." --print-threads --print-bytecodes --print-kernel
```

## Current Features & Roadmap

  - **Support for GGUF format models** with full FP16 and partial support for Q8_0 and Q4_0 quantization.
  - **Instruction-following and chat modes** for various use cases.
  - **Interactive CLI** with `--interactive` and `--instruct` modes.
  - **Flexible backend switching** - choose OpenCL or PTX at runtime (need to build TornadoVM with both enabled).
  - **Cross-platform compatibility**:
    - ✅ NVIDIA GPUs (OpenCL & PTX )
    - ✅ Intel GPUs (OpenCL)
    - ✅ Apple GPUs (OpenCL)

Click [here](https://github.com/beehive-lab/GPULlama3.java/tree/main/docs/TORNADOVM_TRANSFORMER_OPTIMIZATIONS.md) to view a more detailed list of the transformer optimizations implemented in TornadoVM.

Click [here](https://github.com/beehive-lab/GPULlama3.java/tree/main/docs/GPULlama3_ROADMAP.md) to see the roadmap of the project.

-----------

## Acknowledgments

This work is partially funded by the following EU & UKRI grants (most recent first):

- EU Horizon Europe & UKRI [AERO 101092850](https://aero-project.eu/).
- EU Horizon Europe & UKRI [P2CODE 101093069](https://p2code-project.eu/).
- EU Horizon Europe & UKRI [ENCRYPT 101070670](https://encrypt-project.eu).
- EU Horizon Europe & UKRI [TANGO 101070052](https://tango-project.eu).

-----------

## License

MIT
