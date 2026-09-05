#!/usr/bin/env bash
# Local RTX 5090 baseline for the architecture refactor, TornadoVM 5.2.0 floor.
# Matrix: model family x quantization x backend. 3 runs per config, median tok/s.

set -u

SDK_ROOT=/home/orion/.sdkman/candidates/tornadovm
export JAVA_HOME=/home/orion/.sdkman/candidates/java/current
export LLAMA_ROOT=/home/orion/GPULlama3.java
M=/home/orion/LLMModels

OUT_DIR="$1"
mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/results.jsonl"
LOGS="$OUT_DIR/logs"
mkdir -p "$LOGS"

PROMPT="Write a detailed 400-word essay about the history of computing."
NTOK=256
RUNS=3

# family|label|quant|path
CONFIGS=(
"LLAMA_3|Llama-3.2-1B-Instruct|F16|$M/Llama-3.2-1B-Instruct-F16.gguf"
"LLAMA_3|Llama-3.2-1B-Instruct|Q8_0|$M/Llama-3.2-1B-Instruct-Q8_0.gguf"
"QWEN_3|Qwen3-0.6B|F16|$M/Qwen3-0.6B-f16.gguf"
"QWEN_3|Qwen3-0.6B|Q8_0|$M/Qwen3-0.6B-Q8_0.gguf"
"QWEN_2|Qwen2.5-0.5B-Instruct|F16|$M/Qwen2.5-0.5B-Instruct-f16.gguf"
"QWEN_2|Qwen2.5-0.5B-Instruct|Q8_0|$M/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
"DEEPSEEK_R1_DISTILL_QWEN|DeepSeek-R1-Distill-Qwen-1.5B|F16|$M/DeepSeek-R1-Distill-Qwen-1.5B-F16.gguf"
"DEEPSEEK_R1_DISTILL_QWEN|DeepSeek-R1-Distill-Qwen-1.5B|Q8_0|$M/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
"GRANITE|granite-4.0-1b|F16|$M/granite-4.0-1b-F16.gguf"
"GRANITE|granite-4.0-1b|Q8_0|$M/granite-4.0-1b-Q8_0.gguf"
"PHI_3|Phi-3-mini-4k-instruct|F16|$M/Phi-3-mini-4k-instruct-fp16.gguf"
"PHI_3|Phi-3-mini-4k-instruct|Q8_0|$M/Phi-3-mini-4k-instruct-Q8_0.gguf"
"MISTRAL|Mistral-7B-Instruct-v0.3|F16|$M/Mistral-7B-Instruct-v0.3.fp16.gguf"
"MISTRAL|Mistral-7B-Instruct-v0.3|Q8_0|$M/Mistral-7B-Instruct-v0.3.Q8_0.gguf"
)

gpu_mem_for() {
  # size-aware device memory budget, capped below the 23.4GB device limit
  local bytes gb
  bytes=$(stat -c %s "$1")
  gb=$(( (bytes / 1073741824) * 2 + 6 ))
  [ "$gb" -lt 14 ] && gb=14
  [ "$gb" -gt 22 ] && gb=22
  echo "${gb}GB"
}

median3() { printf '%s\n' "$@" | sort -n | sed -n 2p; }

for BACKEND in cuda opencl; do
  export TORNADOVM_HOME="$SDK_ROOT/5.2.0-jdk21-$BACKEND"
  export PATH="$TORNADOVM_HOME/bin:$JAVA_HOME/bin:/usr/bin:/bin"
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r FAMILY LABEL QUANT PATH_GGUF <<< "$CFG"
    [ -f "$PATH_GGUF" ] || { echo "SKIP missing $PATH_GGUF"; continue; }
    GMEM=$(gpu_mem_for "$PATH_GGUF")
    TAG="${BACKEND}_${LABEL}_${QUANT}"
    echo "=== $TAG (gpu-memory=$GMEM) ==="
    RATES=(); STATUS="ok"; LOAD=""; JIT=""; WARM=""; COPYIN=""
    for i in $(seq 1 $RUNS); do
      LOG="$LOGS/${TAG}_run${i}.log"
      timeout 900 "$LLAMA_ROOT/llama-tornado" --gpu --verbose-init \
        --gpu-memory "$GMEM" --model "$PATH_GGUF" \
        --prompt "$PROMPT" -n $NTOK > "$LOG" 2>&1
      RC=$?
      R=$(grep -oP 'achieved tok/s: \K[0-9]+(\.[0-9]+)?' "$LOG" | head -1)
      if [ $RC -ne 0 ] || [ -z "$R" ]; then
        STATUS="fail_rc${RC}"
        echo "  run$i FAILED rc=$RC"
        break
      fi
      RATES+=("$R")
      echo "  run$i ${R} tok/s"
      LOAD=$(grep -oP 'GGUF Model Load: \K[0-9.]+' "$LOG" | head -1)
      JIT=$(grep -oP 'Compilation & CodeGen: \K[0-9.]+' "$LOG" | head -1)
      WARM=$(grep -oP 'Warmup: \K[0-9.]+' "$LOG" | head -1)
      COPYIN=$(grep -oP 'Read-only weights Copy-in: \K[0-9.]+' "$LOG" | head -1)
    done
    if [ "$STATUS" = "ok" ]; then
      MED=$(median3 "${RATES[@]}")
      ALL=$(printf '%s,' "${RATES[@]}"); ALL="[${ALL%,}]"
    else
      MED="null"; ALL="[]"
    fi
    printf '{"backend":"%s","family":"%s","model":"%s","quantization":"%s","status":"%s","eval_rate_median":%s,"eval_rate_runs":%s,"tokens_requested":%s,"gpu_memory":"%s","load_ms":%s,"codegen_ms":%s,"warmup_ms":%s,"copyin_ms":%s}\n' \
      "$BACKEND" "$FAMILY" "$LABEL" "$QUANT" "$STATUS" "$MED" "$ALL" "$NTOK" "$GMEM" \
      "${LOAD:-null}" "${JIT:-null}" "${WARM:-null}" "${COPYIN:-null}" >> "$RESULTS"
  done
done
echo "DONE -> $RESULTS"
