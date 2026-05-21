#!/bin/bash
# Profiler collection: one run per (attention variant × model × quant) with profiler enabled.
# Dumps TornadoVM profiler JSON to /tmp/qwen3_attn_profiler/<backend>/<model>_<variant>.json.
# Throughput logs to /tmp/qwen3_attn_profiler/<backend>/<model>_<variant>.txt.
#
# Usage: ./benchmark_qwen3_attn_profiler.sh [backend]
#   backend  ptx | opencl | metal  (default: ptx)
#
# Examples:
#   ./benchmark_qwen3_attn_profiler.sh ptx
#   ./benchmark_qwen3_attn_profiler.sh opencl
#   ./benchmark_qwen3_attn_profiler.sh metal

set -eo pipefail

BACKEND=${1:-ptx}

case "$BACKEND" in
    ptx)    BACKEND_FLAG="--ptx"    ;;
    opencl) BACKEND_FLAG="--opencl" ;;
    metal)  BACKEND_FLAG="--metal"  ;;
    *)      echo "Unknown backend: $BACKEND (use ptx | opencl | metal)"; exit 1 ;;
esac

source ~/.sdkman/bin/sdkman-init.sh
sdk use java 25.0.2-open
sdk use tornadovm 4.0.1-jdk25-${BACKEND}
cd ~/GPULlama3.java
source set_paths

MODELS_DIR=~/LLMModels
OUT_DIR=/tmp/qwen3_attn_profiler/${BACKEND}
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

VARIANTS=("v1" "opt" "optv2")

echo "Backend: $BACKEND  Output: $OUT_DIR"

for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "========== Profiling attention kernel: $VARIANT =========="
    for MODEL in "${MODELS[@]}"; do
        TAG="${MODEL%.gguf}"
        DUMP="$OUT_DIR/${TAG}_${VARIANT}.json"
        LOG="$OUT_DIR/${TAG}_${VARIANT}.txt"
        printf "  %-28s ... " "$TAG"
        JAVA_TOOL_OPTIONS="-Dllama.attentionKernel=${VARIANT}" \
            ./llama-tornado --gpu $BACKEND_FLAG --gpu-memory 20GB \
            --profiler \
            --profiler-dump-dir "$DUMP" \
            --model "$MODELS_DIR/$MODEL" \
            --max-tokens 2048 \
            --prompt "Explain the concept of entropy in one paragraph. /no_think" \
            > "$LOG" 2>&1
        TOKS=$(grep "achieved tok/s" "$LOG" | tail -1 | grep -oP '(?<=achieved tok/s: )[\d.]+')
        DUMP_SIZE=$(wc -c < "$DUMP" 2>/dev/null || echo 0)
        printf "tok/s=%-8s dump=%s bytes\n" "${TOKS:-ERR}" "$DUMP_SIZE"
    done
done

echo ""
echo "============================================================"
echo "  Profiler dump summary — backend=$BACKEND"
echo "============================================================"
printf "%-40s %12s\n" "File" "Size"
for f in "$OUT_DIR"/*.json; do
    printf "%-40s %12s\n" "$(basename "$f")" "$(wc -c < "$f") B"
done
echo "Dumps: $OUT_DIR"
