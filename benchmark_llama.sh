#!/bin/bash

# Array of models to test
models=(
    "Llama-3.2-3B-Instruct-Q4_0.gguf"
    "Llama-3.2-1B-Instruct-Q4_0.gguf"
    "Llama-3.2-1B-Instruct-Q8_0.gguf"
)

# Number of runs per model
runs=20

# Base command
base_cmd="./llama-tornado --gpu --prompt \"tell me a joke\" --max-tokens 1024"

echo "Starting LLaMA Model Benchmark"
echo "Running each model $runs times..."
echo "================================================"

# Function to extract tok/s from output
extract_tokps() {
    echo "$1" | grep "achieved tok/s:" | sed -n 's/.*achieved tok\/s: \([0-9]*\.[0-9]*\).*/\1/p'
}

# Loop through each model
for model in "${models[@]}"; do
    echo "Testing model: $model"
    echo "------------------------------------------------"

    total_tokps=0
    successful_runs=0

    # Run the model 20 times
    for i in $(seq 1 $runs); do
        echo -n "Run $i/$runs... "

        # Execute command and capture output
        output=$(eval "$base_cmd --model \"$model\"" 2>&1)

        # Extract tok/s value
        tokps=$(extract_tokps "$output")

        if [[ -n "$tokps" && "$tokps" != "0" ]]; then
            echo "tok/s: $tokps"
            total_tokps=$(awk "BEGIN {print $total_tokps + $tokps}")
            ((successful_runs++))
        else
            echo "FAILED or no tok/s found"
        fi
    done

    # Calculate average
    if [[ $successful_runs -gt 0 ]]; then
        average_tokps=$(awk "BEGIN {printf \"%.2f\", $total_tokps / $successful_runs}")
        echo ""
        echo "Results for $model:"
        echo "  Successful runs: $successful_runs/$runs"
        echo "  Average tok/s: $average_tokps"
        echo ""
    else
        echo ""
        echo "Results for $model:"
        echo "  No successful runs!"
        echo ""
    fi

    echo "================================================"
done

echo "Benchmark completed!"