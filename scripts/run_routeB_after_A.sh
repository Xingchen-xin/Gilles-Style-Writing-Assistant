#!/bin/bash
set -e
export NCCL_P2P_DISABLE=1 PYTHONUNBUFFERED=1

ROUTE_A_PID=$1
echo "Waiting for Route A (PID $ROUTE_A_PID) to finish..."

# Wait for Route A to complete
while kill -0 $ROUTE_A_PID 2>/dev/null; do
    sleep 30
done
echo "Route A finished. Starting Route B..."

eval "$(micromamba shell hook -s bash)"
micromamba activate gswa

echo ""
echo "=== Route B: Style-Enhanced training (context-window, rank=32) ==="
echo "Data: context-window_train.jsonl (3,950 entries, with V1+V2+V3 variants)"
echo "Mode: style-enhanced (rank=32, alpha=64, max_len=4096, epochs=4)"
echo ""

accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=no --dynamo_backend=no --multi_gpu \
  scripts/finetune_lora.py \
  --model mistral-nemo \
  --style-enhanced \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --disable-tqdm

echo ""
echo "=== Route B Complete ==="
