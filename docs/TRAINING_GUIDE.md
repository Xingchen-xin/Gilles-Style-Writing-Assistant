# GSWA Training Guide - Foolproof One-Click Tutorial

## Quick Start (Recommended)

```bash
# One-click training with auto-configuration
python -m gswa.train train --preprocess --auto-plan -y

# Or use the Makefile shortcut
make train-safe
```

This command automatically:
1. Detects your hardware (Apple Silicon/CUDA)
2. Selects optimal training parameters
3. Preprocesses long sequences (no truncation!)
4. Runs training with OOM fallback protection
5. Generates visualizations and reports

---

## New Unified CLI

The new `gswa.train` CLI provides all training functionality in one place:

```bash
# Show hardware information
python -m gswa.train info

# Preprocess data only
python -m gswa.train preprocess --max-tokens 2048

# Run training planner
python -m gswa.train plan --max-candidates 5

# Full training with all features
python -m gswa.train train --preprocess --auto-plan

# Generate visualizations from a run
python -m gswa.train visualize --run-dir runs/20240115-123456

# List recent runs
python -m gswa.train list
```

---

## Memory Safety Features

### Automatic OOM Fallback

If Metal OOM occurs during training, the system automatically:

1. **First**: Reduces `eval_batch_size`
2. **Then**: Reduces `batch_size`
3. **Then**: Reduces `max_seq_length`
4. **Finally**: Reduces `num_layers`

All fallback decisions are logged with explicit reasons.

### Data Preprocessing

Long sequences are now intelligently split instead of truncated:

```bash
# Analyze data without modifying
python -m gswa.train preprocess --analyze-only

# Preprocess with specific max tokens
python -m gswa.train preprocess --max-tokens 1024 --strategy paragraph

# Strategies available: auto, paragraph, sentence, token
```

**Statistics reported**:
- Token distribution: min/median/p90/p95/p99/max
- Truncation percentage before vs after
- Chunks per original entry
- Markdown report with full analysis

---

## Training Planner

The planner finds optimal configuration through dry-runs:

```bash
python -m gswa.train plan \
    --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
    --max-candidates 5 \
    --dry-run-steps 10
```

**Scoring formula**:
- `score = throughput_score * stability_score * effective_batch_factor`
- Rejects plans near OOM threshold (80% margin by default)
- Deterministic with seed control

---

## Run Directory Structure

Each training run creates a structured directory:

```
runs/<timestamp>/
├── config/
│   ├── run_config.json      # Full configuration
│   └── hardware_info.json   # Hardware snapshot
├── logs/
│   ├── train_steps.jsonl    # Step-by-step metrics
│   ├── eval_steps.jsonl     # Evaluation metrics
│   └── events.jsonl         # OOM events, config changes
├── plots/
│   ├── loss.png             # Loss curves
│   ├── throughput.png       # Tokens/sec
│   ├── memory.png           # Peak memory
│   └── length_distribution.png
├── reports/
│   └── report.html          # Interactive HTML report
├── adapters/                 # LoRA adapters
├── stats/
│   └── preprocess_stats.json
└── metadata.json
```

---

## Configuration Files

### Using YAML/JSON Configs

```yaml
# configs/my_run.yaml
model_id: mlx-community/Mistral-7B-Instruct-v0.2-4bit
batch_size: 2
max_seq_length: 1536
num_layers: 12
iters: 500
learning_rate: 1e-5
lora_rank: 8
enable_oom_fallback: true
preprocess_enabled: true
seed: 42
```

```bash
python -m gswa.train train --config configs/my_run.yaml
```

### Reproducing a Run

```bash
# Use exact config from previous run
python -m gswa.train train --config runs/20240115-123456/config/run_config.json
```

---

## Memory Configuration Reference

| RAM | Recommended Settings | Notes |
|-----|---------------------|-------|
| 8GB | batch=1, seq=512, layers=4 | Minimal, use --memory-safe |
| 16GB | batch=1, seq=768-1024, layers=8 | Standard configuration |
| 32GB | batch=2, seq=1536, layers=12 | Good balance |
| 64GB | batch=4, seq=2048, layers=16 | Full capability |
| 128GB+ | batch=8, seq=4096, layers=24 | Maximum settings |

---

## Interpreting Logs and Plots

### Training Logs (JSONL)

```json
{
  "step": 100,
  "timestamp": "2024-01-15T10:30:45",
  "train_loss": 1.234,
  "learning_rate": 1e-5,
  "tokens_per_sec": 150.5,
  "peak_memory_gb": 12.3,
  "trained_tokens": 100000
}
```

### Loss Plot
- **Train loss** (blue): Should decrease smoothly
- **Eval loss** (green): Should track train loss without diverging
- **OOM markers** (red dashed): Show fallback events

### Throughput Plot
- Stable throughput indicates healthy training
- Drops may indicate memory pressure

### Memory Plot
- Filled area shows peak memory usage
- Should stay below available memory

---

## Common Issues and Solutions

### Issue 1: `Insufficient Memory` / OOM

```bash
# Solution: Use auto-fallback (default) or memory-safe mode
python -m gswa.train train --preprocess

# Or manually specify conservative settings
python -m gswa.train train \
    --batch-size 1 \
    --max-seq-length 768 \
    --num-layers 4
```

### Issue 2: Long sequences being truncated

```bash
# Solution: Enable preprocessing
python -m gswa.train train --preprocess

# Or preprocess separately first
python -m gswa.train preprocess --max-tokens 1024
```

### Issue 3: Noisy training loss

```bash
# Solution: Use gradient accumulation
# Edit config or use planner which considers effective batch size
python -m gswa.train plan --max-candidates 10
```

### Issue 4: Validation OOM (training works)

The fallback system handles this automatically by reducing eval_batch_size first.

---

## Complete Workflow

### 1. Prepare Corpus
```bash
# Place files in data/corpus/raw/
# Priority files in data/corpus/raw/important_examples/
```

### 2. Parse and Prepare
```bash
make parse-corpus
make prepare-training
```

### 3. Train with Full Pipeline
```bash
python -m gswa.train train \
    --preprocess \
    --auto-plan \
    --name my-experiment
```

### 4. Review Results
```bash
# Open HTML report
open runs/<your-run>/reports/report.html

# Or regenerate visualizations
python -m gswa.train visualize --run-dir runs/<your-run>
```

### 5. Create Ollama Model
```bash
ollama create gswa-gilles -f runs/<your-run>/adapters/Modelfile
```

### 6. Use the Model
```bash
echo 'VLLM_MODEL_NAME=gswa-gilles' >> .env
make run
```

---

## Unit Tests

Run tests to verify the training infrastructure:

```bash
pytest tests/test_training.py -v
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `python -m gswa.train info` | Show hardware information |
| `python -m gswa.train preprocess` | Preprocess training data |
| `python -m gswa.train plan` | Run training planner |
| `python -m gswa.train train` | Run training |
| `python -m gswa.train visualize` | Generate visualizations |
| `python -m gswa.train list` | List recent runs |
| `make train-safe` | Shortcut for safe training |
| `make analyze-data` | Analyze training data |
