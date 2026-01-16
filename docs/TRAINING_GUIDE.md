# GSWA Training Guide - Foolproof One-Click Tutorial

## Quick Start (Solving Memory Issues)

If you encounter `Insufficient Memory` or `OOM` errors, use this command:

```bash
# One-click memory-safe training (RECOMMENDED)
make train-safe
```

This command automatically:
1. Parses your corpus files
2. Prepares training data
3. **Splits long text sequences**
4. **Uses conservative memory settings**
5. **Retries with reduced settings on OOM**

---

## Common Issues and Solutions

### Issue 1: `Insufficient Memory` / `kIOGPUCommandBufferCallbackErrorOutOfMemory`

**Cause**: Training data contains sequences that are too long for GPU memory.

**Solutions**:

```bash
# Method 1: Use memory-safe mode (RECOMMENDED)
make train-safe

# Method 2: Manual preprocessing
make analyze-data          # View data statistics
make preprocess-data       # Auto-split long sequences
make finetune-mlx-safe     # Memory-safe training

# Method 3: Specify max token length
make preprocess-data MAX_TOKENS=1024
```

### Issue 2: Training is too slow

**Solutions**:
- Close other GPU-intensive apps (browsers, Photoshop, etc.)
- Use a smaller model: `python scripts/finetune_mlx_mac.py --model phi3`

### Issue 3: Want higher quality training

If your Mac has enough memory (32GB+), use more aggressive settings:

```bash
# List available profiles
python scripts/finetune_mlx_mac.py --list-profiles

# Use a specific profile
python scripts/finetune_mlx_mac.py --profile mac_m1_32gb
```

---

## Memory Configuration Reference

| RAM | Profile | batch_size | max_seq_length | num_layers |
|-----|---------|------------|----------------|------------|
| 8GB | mac_m1_8gb | 1 | 512 | 4 |
| 16GB | mac_m1_16gb | 1 | 1024 | 8 |
| 32GB | mac_m1_32gb | 2 | 1536 | 12 |
| 64GB | mac_m1_64gb | 4 | 2048 | 16 |
| 128GB+ | mac_m1_128gb | 8 | 4096 | 24 |

---

## Complete Training Workflow

### Step 1: Prepare Corpus
```bash
# Place PDF/DOCX files in:
# data/corpus/raw/

# Important files (higher weight):
# data/corpus/raw/important_examples/
```

### Step 2: Parse Corpus
```bash
make parse-corpus
```

### Step 3: Prepare Training Data
```bash
make prepare-training
```

### Step 4: Analyze Data (Optional but Recommended)
```bash
make analyze-data
```

### Step 5: Preprocess Data (if long sequences exist)
```bash
make preprocess-data
```

### Step 6: Train
```bash
make finetune-mlx-safe
```

### Step 7: Create Ollama Model
```bash
ollama create gswa-gilles -f models/gswa-mlx-*/Modelfile
```

### Step 8: Configure and Run
```bash
echo 'VLLM_MODEL_NAME=gswa-gilles' >> .env
make run
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `make train-safe` | Foolproof one-click training (RECOMMENDED) |
| `make train-auto` | Auto training (no confirmation) |
| `make train` | Interactive training wizard |
| `make analyze-data` | Analyze training data |
| `make preprocess-data` | Preprocess long sequences |
| `make finetune-mlx-safe` | Memory-safe MLX training |
| `make finetune-all-safe` | Full pipeline (memory-safe) |

---

## Manual Parameters

```bash
python scripts/finetune_mlx_mac.py \
    --model mistral \
    --batch-size 1 \
    --max-seq-length 1024 \
    --num-layers 8 \
    --iters 500 \
    --memory-safe
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 4 | Batch size, reduce to save memory |
| `--max-seq-length` | 1024 | Max sequence length, reduce to save memory |
| `--num-layers` | 16 | LoRA layers, reduce to save memory |
| `--iters` | 1000 | Training iterations |
| `--memory-safe` | False | Enable memory-safe mode |
| `--retry-on-oom` | True | Auto-retry on OOM |

---

## Troubleshooting

```bash
# Check dependencies
make check-mlx

# View status
make status

# List models
make models

# Clean and restart
make clean
rm -rf data/training/mlx/
make train-safe
```
