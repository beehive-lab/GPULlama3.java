#!/bin/bash
# Phase 2: Qwen3 instrumentation capture + baseline benchmarks
# Runs all 8 Qwen3 model/quant combinations with --max-tokens 2048
# Output per model is captured to /tmp/qwen3_bench_<model>.txt

set -e
source ~/.sdkman/bin/sdkman-init.sh
sdk use java 25.0.2-open
sdk use tornadovm 4.0.0-jdk25-ptx
cd ~/GPULlama3.java
source set_paths

MODELS_DIR=~/LLMModels
OUT_DIR=/tmp/qwen3_bench
mkdir -p "$OUT_DIR"

MODELS=(
    "Qwen3-0.6B-f16.gguf"
    "Qwen3-0.6B-Q8_0.gguf"
    "Qwen3-1.7B-f16.gguf"
    "Qwen3-1.7B-Q8_0.gguf"
    "Qwen3-4B-f16.gguf"
    "Qwen3-4B-Q8_0.gguf"
    "Qwen3-8B-f16.gguf"
    "Qwen3-8B-Q8_0.gguf"
)

for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL%.gguf}"
    OUT="$OUT_DIR/${TAG}.txt"
    echo "=== Running $MODEL ==="
    JAVA_TOOL_OPTIONS="-Dllama.debugKernelSizes=true" \
        ./llama-tornado --gpu --ptx --gpu-memory 20GB \
        --model "$MODELS_DIR/$MODEL" \
        --max-tokens 2048 \
        --prompt "Explain the concept of entropy in one paragraph. /no_think" \
        > "$OUT" 2>&1
    echo "    -> $OUT (exit $?)"
done

echo ""
echo "All runs complete. Results in $OUT_DIR"
