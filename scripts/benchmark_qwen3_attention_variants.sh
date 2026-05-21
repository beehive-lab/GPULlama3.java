#!/bin/bash
# Collect profiler logs and throughput for all 3 attention kernel variants across all 8 Qwen3 models.
# Variant is selected via -Dllama.attentionKernel=v1|opt|optv2 — requires the Qwen3 layer classes
# to read this property. Since the property switch is not yet wired, we instead run each variant
# by temporarily overriding via JAVA_TOOL_OPTIONS and the approach below.
#
# NOTE: Because the kernel is hardcoded, we benchmark each variant by passing its name as a
# system property and relying on whatever wiring exists, OR we collect for the current default (v1)
# alongside opt and optv2 once wired. For now this script profiles the CURRENT default (v1) with
# profiler enabled, matching the already-collected profiler run, then produces throughput-only
# runs for comparison once the variants are wired.
#
# For now: collect throughput (no profiler) for each attention variant using the two existing
# alternative kernels. The user will wire the switch after reviewing results.

source ~/.sdkman/bin/sdkman-init.sh
sdk use java 25.0.2-open
sdk use tornadovm 4.0.0-jdk25-ptx
cd ~/GPULlama3.java
source set_paths

MODELS_DIR=~/LLMModels
OUT_DIR=/tmp/qwen3_attn_variants
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

for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "=============================="
    echo "Attention variant: $VARIANT"
    echo "=============================="
    for MODEL in "${MODELS[@]}"; do
        TAG="${MODEL%.gguf}"
        PROF_DUMP="$OUT_DIR/${TAG}_${VARIANT}.json"
        LOG="$OUT_DIR/${TAG}_${VARIANT}.txt"
        echo "  $MODEL ..."
        JAVA_TOOL_OPTIONS="-Dllama.attentionKernel=${VARIANT}" \
            ./llama-tornado --gpu --ptx --gpu-memory 20GB \
            --profiler \
            --profiler-dump-dir "$PROF_DUMP" \
            --model "$MODELS_DIR/$MODEL" \
            --max-tokens 2048 \
            --prompt "Explain the concept of entropy in one paragraph. /no_think" \
            > "$LOG" 2>&1
        TOKS=$(grep "achieved tok/s" "$LOG" | tail -1)
        echo "    $TOKS"
    done
done

echo ""
echo "=== Throughput Summary ==="
printf "%-28s %12s %12s %12s\n" "Model" "v1 (tok/s)" "opt (tok/s)" "optv2 (tok/s)"
for MODEL in "${MODELS[@]}"; do
    TAG="${MODEL%.gguf}"
    R_V1=$(grep   "achieved tok/s" "$OUT_DIR/${TAG}_v1.txt"    2>/dev/null | tail -1 | grep -oP '[\d.]+(?=\. Tokens)')
    R_OPT=$(grep  "achieved tok/s" "$OUT_DIR/${TAG}_opt.txt"   2>/dev/null | tail -1 | grep -oP '[\d.]+(?=\. Tokens)')
    R_V2=$(grep   "achieved tok/s" "$OUT_DIR/${TAG}_optv2.txt" 2>/dev/null | tail -1 | grep -oP '[\d.]+(?=\. Tokens)')
    printf "%-28s %12s %12s %12s\n" "$TAG" "${R_V1:-?}" "${R_OPT:-?}" "${R_V2:-?}"
done

echo ""
echo "Profiler dumps: $OUT_DIR"
ls -lh "$OUT_DIR"/*.json 2>/dev/null
