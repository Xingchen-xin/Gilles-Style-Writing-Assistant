#!/bin/bash
# Start vLLM server with OpenAI-compatible API
#
# Usage:
#   ./scripts/start_vllm.sh
#   VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2 ./scripts/start_vllm.sh

set -e

MODEL=${VLLM_MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}
PORT=${VLLM_PORT:-8000}
GPU_MEMORY=${GPU_MEMORY_UTIL:-0.9}

echo "========================================"
echo "Starting vLLM Server"
echo "========================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY"
echo ""

# Check if vllm is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM not installed. Install with: pip install vllm"
    exit 1
fi

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --trust-remote-code \
    --host 0.0.0.0
