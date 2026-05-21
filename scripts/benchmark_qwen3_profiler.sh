#!/bin/bash
# Collect TornadoVM profiler logs for all 8 Qwen3 model/quant combinations.
# Profiler runs in silent mode (log.profiler=false, dump to JSON file).
# Output: /tmp/qwen3_profiler/<model>.json  (profiler dump)
#         /tmp/qwen3_profiler/<model>.txt   (inference stdout/stderr)

source ~/.sdkman/bin/sdkman-init.sh
sdk use java 25.0.2-open
sdk use tornadovm 4.0.0-jdk25-ptx
cd ~/GPULlama3.java
source set_paths

MODELS_DIR=~/LLMModels
OUT_DIR=/tmp/qwen3_profiler
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
    DUMP="$OUT_DIR/${TAG}.json"
    LOG="$OUT_DIR/${TAG}.txt"
    echo "=== $MODEL ==="
    ./llama-tornado --gpu --ptx --gpu-memory 20GB \
        --profiler \
        --profiler-dump-dir "$DUMP" \
        --model "$MODELS_DIR/$MODEL" \
        --max-tokens 2048 \
        --prompt "Explain the concept of entropy in one paragraph. /no_think" \
        > "$LOG" 2>&1
    echo "    exit=$?  dump=$DUMP  log=$LOG"
    if [ -f "$DUMP" ]; then
        SIZE=$(wc -c < "$DUMP")
        echo "    profiler dump: ${SIZE} bytes"
    else
        echo "    [WARN] profiler dump not found: $DUMP"
    fi
done

echo ""
echo "All runs complete. Profiler dumps in $OUT_DIR"
ls -lh "$OUT_DIR"/*.json 2>/dev/null
