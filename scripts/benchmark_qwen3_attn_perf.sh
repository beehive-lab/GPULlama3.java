#!/bin/bash
# Performance benchmark: average tok/s over N runs per (attention variant × model × quant).
# Profiler is DISABLED. Results written to /tmp/qwen3_attn_perf/<backend>/.
#
# Usage: ./benchmark_qwen3_attn_perf.sh [runs] [backend]
#   runs    number of timed runs per combination (default: 5)
#   backend ptx | opencl | metal                 (default: ptx)
#
# Examples:
#   ./benchmark_qwen3_attn_perf.sh 5 ptx
#   ./benchmark_qwen3_attn_perf.sh 5 opencl
#   ./benchmark_qwen3_attn_perf.sh 5 metal

set -eo pipefail

RUNS=${1:-5}
BACKEND=${2:-ptx}

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
OUT_DIR=/tmp/qwen3_attn_perf/${BACKEND}
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

extract_toks() {
    grep "achieved tok/s" "$1" 2>/dev/null | tail -1 | grep -oP '(?<=achieved tok/s: )[\d.]+'
}

compute_avg() {
    echo "$@" | tr ' ' '\n' | awk 'NF{s+=$1; n++} END{if(n>0) printf "%.2f", s/n; else print "?"}'
}

declare -A RESULTS

echo "Backend: $BACKEND  Runs: $RUNS  Output: $OUT_DIR"

for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "========== Attention kernel: $VARIANT =========="
    for MODEL in "${MODELS[@]}"; do
        TAG="${MODEL%.gguf}"
        KEY="${TAG}__${VARIANT}"
        VALUES=()
        printf "  %-28s runs: " "$TAG"
        for ((i=1; i<=RUNS; i++)); do
            LOG="$OUT_DIR/${TAG}_${VARIANT}_run${i}.txt"
            JAVA_TOOL_OPTIONS="-Dllama.attentionKernel=${VARIANT}" \
                ./llama-tornado --gpu $BACKEND_FLAG --gpu-memory 20GB \
                --model "$MODELS_DIR/$MODEL" \
                --max-tokens 2048 \
                --prompt "Explain the concept of entropy in one paragraph. /no_think" \
                > "$LOG" 2>&1
            VAL=$(extract_toks "$LOG")
            if [[ -n "$VAL" ]]; then
                VALUES+=("$VAL")
                printf "%s " "$VAL"
            else
                printf "ERR "
            fi
        done
        AVG=$(compute_avg "${VALUES[@]}")
        RESULTS["$KEY"]="$AVG"
        echo "-> avg=$AVG tok/s"
    done
done

echo ""
echo "============================================================"
echo "  Average tok/s over $RUNS runs — backend=$BACKEND (profiler OFF)"
echo "============================================================"
printf "%-28s %10s %10s %10s %12s %12s\n" "Model" "v1" "opt" "optv2" "opt/v1" "optv2/v1"
for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL%.gguf}"
    V1="${RESULTS[${TAG}__v1]:-?}"
    OPT="${RESULTS[${TAG}__opt]:-?}"
    V2="${RESULTS[${TAG}__optv2]:-?}"
    if [[ "$V1" != "?" && "$OPT" != "?" ]]; then
        R_OPT=$(awk "BEGIN{printf \"%.1f%%\", ($OPT/$V1-1)*100}")
    else
        R_OPT="?"
    fi
    if [[ "$V1" != "?" && "$V2" != "?" ]]; then
        R_V2=$(awk "BEGIN{printf \"%.1f%%\", ($V2/$V1-1)*100}")
    else
        R_V2="?"
    fi
    printf "%-28s %10s %10s %10s %12s %12s\n" "$TAG" "$V1" "$OPT" "$V2" "$R_OPT" "$R_V2"
done
echo "============================================================"
echo "Raw logs: $OUT_DIR"
