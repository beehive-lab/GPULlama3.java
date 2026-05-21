#!/bin/bash
# Phase 4: LWS sweep across all 8 Qwen3 model/quant combinations
# Sweeps llama.localWorkGroupSize in {32, 64, 128}
# Output per run: /tmp/qwen3_lws/<model>_lws<N>.txt

source ~/.sdkman/bin/sdkman-init.sh
sdk use java 25.0.2-open
sdk use tornadovm 4.0.0-jdk25-ptx
cd ~/GPULlama3.java
source set_paths

MODELS_DIR=~/LLMModels
OUT_DIR=/tmp/qwen3_lws
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

LWS_VALUES=(32 64 128)

for LWS in "${LWS_VALUES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        TAG="${MODEL%.gguf}"
        OUT="$OUT_DIR/${TAG}_lws${LWS}.txt"
        echo "=== $MODEL  LWS=$LWS ==="
        JAVA_TOOL_OPTIONS="-Dllama.localWorkGroupSize=${LWS}" \
            ./llama-tornado --gpu --ptx --gpu-memory 20GB \
            --model "$MODELS_DIR/$MODEL" \
            --max-tokens 2048 \
            --prompt "Explain the concept of entropy in one paragraph. /no_think" \
            > "$OUT" 2>&1
        TOKS=$(grep "achieved tok/s" "$OUT" | tail -1)
        echo "    $TOKS"
        echo "    -> $OUT"
    done
done

echo ""
echo "=== Summary ==="
printf "%-28s %8s %8s %8s\n" "Model" "LWS=32" "LWS=64" "LWS=128"
for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL%.gguf}"
    R32=$(grep "achieved tok/s" "$OUT_DIR/${TAG}_lws32.txt" 2>/dev/null | tail -1 | grep -oP '[\d.]+(?= tok/s)' | head -1)
    R64=$(grep "achieved tok/s" "$OUT_DIR/${TAG}_lws64.txt" 2>/dev/null | tail -1 | grep -oP '[\d.]+(?= tok/s)' | head -1)
    R128=$(grep "achieved tok/s" "$OUT_DIR/${TAG}_lws128.txt" 2>/dev/null | tail -1 | grep -oP '[\d.]+(?= tok/s)' | head -1)
    printf "%-28s %8s %8s %8s\n" "$TAG" "${R32:-?}" "${R64:-?}" "${R128:-?}"
done
