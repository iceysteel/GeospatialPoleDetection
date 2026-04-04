#!/bin/bash
# Start vLLM server for Qwen 3.5 VLM classification
# Usage: ./scripts/start_vllm.sh [9b|27b]

MODEL_SIZE="${1:-9b}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/../.venv/bin/python"

case "$MODEL_SIZE" in
    9b)
        MODEL="Qwen/Qwen3.5-9B"
        GPU_UTIL=0.85
        ;;
    27b)
        MODEL="Qwen/Qwen3.5-27B-AWQ"
        GPU_UTIL=0.90
        ;;
    *)
        echo "Usage: $0 [9b|27b]"
        exit 1
        ;;
esac

echo "Starting vLLM with $MODEL (TP=2 across both GPUs)..."
exec $VENV -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 4096 \
    --trust-remote-code \
    --port 8000
