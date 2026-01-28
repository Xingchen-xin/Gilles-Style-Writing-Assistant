#!/bin/bash
# Start vLLM server with OpenAI-compatible API
# Supports LoRA adapter serving for fine-tuned models
#
# Usage:
#   ./scripts/start_vllm.sh                    # Base model only
#   ./scripts/start_vllm.sh --lora             # Auto-discover LoRA adapters
#   ./scripts/start_vllm.sh --lora --background # Start in background
#   ./scripts/start_vllm.sh --check            # Check if running
#   VLLM_MODEL=mistralai/Mistral-Nemo-Instruct-2407 ./scripts/start_vllm.sh --lora

set -e

MODEL=${VLLM_MODEL:-"mistralai/Mistral-Nemo-Instruct-2407"}
PORT=${VLLM_PORT:-8000}
GPU_MEMORY=${GPU_MEMORY_UTIL:-0.9}
MAX_MODEL_LEN=${VLLM_MAX_LEN:-8192}  # Limit context to save memory
MODELS_DIR=${MODELS_DIR:-"./models"}
ENABLE_LORA=""
BACKGROUND=""
CHECK_ONLY=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --lora)
            ENABLE_LORA="1"
            ;;
        --background|-b)
            BACKGROUND="1"
            ;;
        --check)
            CHECK_ONLY="1"
            ;;
    esac
done

# Function to check if vLLM is running on the port
check_vllm_running() {
    if curl -s --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1 || \
       curl -s --max-time 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Check-only mode
if [ -n "$CHECK_ONLY" ]; then
    if check_vllm_running; then
        echo "vLLM is running on port $PORT"
        exit 0
    else
        echo "vLLM is NOT running on port $PORT"
        exit 1
    fi
fi

# Check if already running
if check_vllm_running; then
    echo "========================================"
    echo "vLLM Already Running"
    echo "========================================"
    echo "vLLM server is already running on port $PORT"
    echo "Use 'curl http://localhost:$PORT/v1/models' to check available models"
    exit 0
fi

echo "========================================"
echo "Starting vLLM Server"
echo "========================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY"
echo "Max Model Length: $MAX_MODEL_LEN"

# Check if we should use micromamba
PYTHON_CMD="python"
if command -v micromamba &>/dev/null && micromamba env list 2>/dev/null | grep -q "gswa"; then
    PYTHON_CMD="micromamba run -n gswa python"
fi

# Check if vllm is installed
if ! $PYTHON_CMD -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM not installed. Install with: pip install vllm"
    exit 1
fi

# Auto-discover LoRA adapters if --lora flag is set
LORA_MODULES=""
if [ -n "$ENABLE_LORA" ]; then
    echo ""
    echo "Scanning for LoRA adapters in $MODELS_DIR..."

    # Use Python to scan and match adapters to the base model
    # Write adapters to temp file to avoid stdout/stderr complexity
    TEMP_ADAPTERS=$(mktemp)
    $PYTHON_CMD -c "
import json, sys
from pathlib import Path

models_dir = Path('$MODELS_DIR')
base_model = '$MODEL'

if not models_dir.exists():
    sys.exit(0)

adapters = []
for d in sorted(models_dir.iterdir()):
    if not d.is_dir() or not d.name.startswith('gswa-'):
        continue
    config_file = d / 'training_config.json'
    adapter_config = d / 'adapter_config.json'
    if not config_file.exists() or not adapter_config.exists():
        continue
    has_weights = (d / 'adapter_model.safetensors').exists() or (d / 'adapter_model.bin').exists()
    if not has_weights:
        continue
    with open(config_file) as f:
        config = json.load(f)
    if config.get('base_model', '') == base_model:
        adapters.append(f'{d.name}={d}')
        print(f'  Found: {d.name} (r={config.get(\"lora_r\", \"?\")}, epochs={config.get(\"epochs\", \"?\")})')

with open('$TEMP_ADAPTERS', 'w') as f:
    f.write(' '.join(adapters))
"
    LORA_MODULES=$(cat "$TEMP_ADAPTERS")
    rm -f "$TEMP_ADAPTERS"

    if [ -n "$LORA_MODULES" ]; then
        echo ""
        echo "LoRA adapters will be served. Select via model name in API calls."
        echo "Example: {\"model\": \"gswa-lora-Mistral-20260123-012408\", ...}"
    else
        echo "  No compatible adapters found for $MODEL"
        echo "  (Adapters must be trained on the same base model)"
        ENABLE_LORA=""
    fi
fi

echo ""
echo "Starting server..."
echo ""

# Build the command
if [ -n "$ENABLE_LORA" ] && [ -n "$LORA_MODULES" ]; then
    VLLM_CMD="$PYTHON_CMD -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $PORT \
        --gpu-memory-utilization $GPU_MEMORY \
        --max-model-len $MAX_MODEL_LEN \
        --trust-remote-code \
        --host 0.0.0.0 \
        --enable-lora \
        --lora-modules $LORA_MODULES \
        --max-lora-rank 64"
else
    VLLM_CMD="$PYTHON_CMD -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $PORT \
        --gpu-memory-utilization $GPU_MEMORY \
        --max-model-len $MAX_MODEL_LEN \
        --trust-remote-code \
        --host 0.0.0.0"
fi

# Start in background or foreground
if [ -n "$BACKGROUND" ]; then
    LOG_FILE="${MODELS_DIR}/../logs/vllm.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    echo "Starting vLLM in background..."
    echo "Log file: $LOG_FILE"

    # Start in background with nohup
    nohup $VLLM_CMD > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for server to be ready (max 120 seconds for model loading)
    echo ""
    echo "Waiting for vLLM to be ready (this may take a while for large models)..."
    for i in $(seq 1 120); do
        if check_vllm_running; then
            echo ""
            echo "========================================"
            echo "vLLM Started Successfully!"
            echo "========================================"
            echo "Server running on port $PORT (PID: $VLLM_PID)"
            echo "Log file: $LOG_FILE"
            exit 0
        fi

        # Check if process is still running
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo ""
            echo "Error: vLLM process died. Check log file: $LOG_FILE"
            tail -20 "$LOG_FILE" 2>/dev/null || true
            exit 1
        fi

        printf "."
        sleep 2
    done

    echo ""
    echo "Warning: vLLM is starting but not ready yet. Check log: $LOG_FILE"
    echo "Server may still be loading the model. PID: $VLLM_PID"
    exit 0
else
    # Foreground mode
    exec $VLLM_CMD
fi
