# GSWA Training Guide - Foolproof One-Click Tutorial

## Environment Setup (IMPORTANT - Read First!)

### Linux Server (No sudo)

在 Linux 服务器上训练前，请确保环境正确配置：

```bash
# 方式一：使用一键设置脚本（推荐）
make setup-cuda      # 交互式安装
make setup-cuda-auto # 全自动安装

# 方式二：手动使用 micromamba
micromamba create -n gswa python=3.11 -y
micromamba activate gswa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,similarity]" pymupdf
```

### 常见问题：`_ctypes` 模块缺失

如果遇到以下错误：
```
ModuleNotFoundError: No module named '_ctypes'
```

**原因**：pyenv 编译的 Python 缺少 libffi 支持（服务器无 sudo 无法安装 libffi-devel）

**解决方案**：使用 micromamba 代替 pyenv
```bash
# micromamba 自带完整的 Python 环境，不需要系统库
curl -L micro.mamba.pm/install.sh | bash
source ~/.bashrc
micromamba create -n gswa python=3.11 -y
micromamba activate gswa
```

### 验证 CUDA 环境

```bash
# 检查 CUDA 是否可用
micromamba run -n gswa python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 应该输出：CUDA: True
# 如果是 False，重新安装 PyTorch：
micromamba run -n gswa pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

详细安装指南请参考：[INSTALL.md](INSTALL.md)

---

## Quick Start (Recommended)

### Mac (Apple Silicon)

```bash
# One-click training with auto-configuration
make train-safe

# Or use the CLI directly
python -m gswa.train train --preprocess --auto-plan -y
```

### Linux (NVIDIA GPU)

```bash
# One-click Linux training with auto-configuration
make train-linux

# Or memory-safe mode with OOM fallback (RECOMMENDED)
make train-linux-safe

# Or use the CLI directly
python -m gswa.train train --preprocess -y
```

Both commands automatically:
1. Detects your hardware (Apple Silicon/NVIDIA CUDA)
2. Selects optimal training parameters
3. Preprocesses long sequences (no truncation!)
4. Runs training with OOM fallback protection
5. Generates visualizations and reports

---

## Linux Training Guide (傻瓜式一键Linux训练)

### Prerequisites

**方式一：使用 conda 环境（推荐）**
```bash
# 激活 conda 环境
micromamba activate gswa

# 验证 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**方式二：使用 venv 环境**
```bash
# 激活 venv
source venv/bin/activate

# 安装 CUDA 训练依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft datasets accelerate bitsandbytes

# 验证 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### One-Click Training

```bash
# Simplest method - fully automatic
make train-linux

# This does everything:
# 1. Parses your corpus files (PDF/DOCX/TXT)
# 2. Prepares training data
# 3. Detects your GPU and VRAM
# 4. Selects optimal parameters
# 5. Preprocesses long sequences
# 6. Trains with progress display
# 7. Generates visualizations
```

### Memory-Safe Mode (Recommended for OOM)

```bash
# Memory-safe training with automatic fallback
make train-linux-safe

# This mode will:
# - Use conservative initial settings
# - Automatically retry with reduced settings on OOM
# - Log all fallback decisions
```

### NVIDIA GPU Memory Reference

| GPU VRAM | Recommended Settings | Example GPUs |
|----------|---------------------|--------------|
| 4GB | batch=1, seq=512, 4bit quant | GTX 1650 |
| 8GB | batch=1, seq=1024, 4bit quant | RTX 3060, RTX 4060 |
| 16GB | batch=2, seq=1536, 4bit quant | RTX 4080, A4000 |
| 24GB | batch=4, seq=2048, 4bit quant | RTX 3090, RTX 4090, A5000 |
| 48GB | batch=8, seq=2048, no quant | A6000, A40 |

### OOM Fallback Strategy (CUDA)

If CUDA OOM occurs during training, the system automatically:

1. **First**: Halves `batch_size`, doubles `gradient_accumulation`
2. **Then**: Halves `batch_size` again if possible
3. **Then**: Reduces `max_seq_length` by 25%
4. **Finally**: Doubles `gradient_accumulation` further

All fallback decisions are logged in `logs/events.jsonl`.

### Manual Configuration

```bash
# Train with specific settings
python -m gswa.train train \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --batch-size 2 \
    --max-seq-length 1024 \
    --preprocess \
    -y
```

### Supported Models for Linux

| Model | VRAM Required | Quality |
|-------|--------------|---------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 4GB | Basic |
| `microsoft/Phi-3.5-mini-instruct` | 8GB | Good |
| `Qwen/Qwen2.5-7B-Instruct` | 16GB | Very Good |
| `mistralai/Mistral-7B-Instruct-v0.3` | 16GB | Very Good |
| `Qwen/Qwen2.5-14B-Instruct` | 24GB | Excellent |
| `mistralai/Mistral-Large-Instruct-2407` | 48GB | Best |

---

## Unified CLI

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

## Real-time Training Visualization

### During Training

Training now shows real-time metrics:

```
──────────────────────────────────────────────────────────────
[ 25.0%] Step   250: Loss=1.8234 | 456.7 tok/s | 12.3GB
```

When training completes, you'll see:
- **Metrics Summary**: Final/min/mean loss, throughput stats
- **ASCII Loss Curve**: Terminal-friendly loss visualization

### Standalone Visualization Script

Use the visualization script for detailed analysis:

```bash
# Visualize latest run
python scripts/visualize_training.py --latest

# Watch training in real-time
python scripts/visualize_training.py --run-dir runs/xxx --watch

# Show ASCII loss graph in terminal
python scripts/visualize_training.py --run-dir runs/xxx --ascii

# Generate specific plots
python scripts/visualize_training.py --run-dir runs/xxx --plots loss,throughput
```

### Generated Reports

After training, open the HTML report:

```bash
open models/gswa-mlx-xxx/reports/report.html
```

The report includes:
- **Loss curves** with smoothing
- **Throughput** over time
- **Memory usage** profile
- **Configuration** summary
- **Training statistics**

---

## Interpreting Logs and Plots

### Training Logs (JSONL)

Structured logs are saved in `logs/train_steps.jsonl`:

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
- **Smoothed line**: Moving average for clearer trend

### Throughput Plot
- Stable throughput indicates healthy training
- Drops may indicate memory pressure or batch size changes

### Memory Plot
- Filled area shows peak memory usage
- Should stay below available memory
- Spikes may trigger OOM fallback

### ASCII Loss Graph (Terminal)

When matplotlib is not available, you get an ASCII graph:

```
Loss: 1.234 - 2.567
┌────────────────────────────────┐
│ ████                           │
│ █████                          │
│ ██████                         │
│ ████████                       │
│ ██████████                     │
│ █████████████                  │
│ ████████████████               │
│ ██████████████████████████████ │
└────────────────────────────────┘
 Step 0                    Step 500
```

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

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m gswa.train info` | Show hardware information |
| `python -m gswa.train preprocess` | Preprocess training data |
| `python -m gswa.train plan` | Run training planner |
| `python -m gswa.train train` | Run training |
| `python -m gswa.train visualize` | Generate visualizations |
| `python -m gswa.train list` | List recent runs |

### Mac Makefile Commands

| Command | Description |
|---------|-------------|
| `make train-safe` | Memory-safe Mac training (RECOMMENDED) |
| `make finetune-mlx` | MLX training (auto-detect settings) |
| `make finetune-mlx-safe` | MLX with memory-safe mode |
| `make finetune-all-safe` | Full pipeline with safe mode |

### Linux Makefile Commands

| Command | Description |
|---------|-------------|
| `make train-linux` | One-click Linux/CUDA training |
| `make train-linux-safe` | Memory-safe Linux training (RECOMMENDED) |
| `make train-linux-full` | Full pipeline with planner |
| `make train-info` | Show hardware info and recommendations |
| `make finetune-lora` | LoRA training (auto-detect settings) |
| `make check-lora` | Check LoRA dependencies |

### Utility Commands

| Command | Description |
|---------|-------------|
| `make analyze-data` | Analyze training data for long sequences |
| `make preprocess-data` | Preprocess data to split long sequences |
| `make prepare-training` | Prepare training data from corpus |
| `make parse-corpus` | Parse corpus files (PDF/DOCX/TXT) |
