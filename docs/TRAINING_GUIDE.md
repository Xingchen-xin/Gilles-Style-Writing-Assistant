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

| GPU VRAM | Recommended Model | Settings | Example GPUs |
|----------|------------------|----------|--------------|
| 4GB | Qwen 1.5B | batch=1, seq=512, 4bit | GTX 1650 |
| 8GB | Phi-3.5 Mini | batch=1, seq=1024, 4bit | RTX 3060, RTX 4060 |
| 16GB | **Mistral 7B** | batch=2, seq=1536, 4bit | RTX 4080, A4000 |
| 24GB | Mistral Nemo 12B | batch=4, seq=2048, 4bit | RTX 3090, RTX 4090, A5000 |
| 48GB | Mistral Large | batch=8, seq=2048, no quant | A6000, A40 |
| 32GB+ (multi-GPU) | Mistral Nemo 12B | QLoRA single GPU | 2x RTX 5000 Ada |
| 60GB+ (multi-GPU) | Llama 3.3 70B (default) | DeepSpeed ZeRO-3 | 2x RTX 5000 Ada |

### Multi-GPU Training

For multi-GPU setups:
- **7B-14B models**: Uses **DDP (Data Distributed Parallel)** via `accelerate launch`
- **70B+ models**: Uses **DeepSpeed ZeRO-3** with CPU offloading

**DDP for 7B-14B models (2+ GPUs):**
- Each GPU loads a full copy of the model (QLoRA 4-bit fits in ~22GB per GPU)
- Data is split across GPUs, gradients are synchronized via all-reduce
- Gives ~1.6x speedup with 2 GPUs, allows larger batch size and LoRA rank
- Requires `NCCL_P2P_DISABLE=1` on non-NVLink systems (handled automatically)

**Important: Never use `device_map="auto"` for training:**
- `device_map="auto"` is **inference-style pipeline parallelism** - doesn't handle gradient sync
- QLoRA + device_map + multi-GPU = **CUDA assertion errors**
- DDP is the correct approach for multi-GPU training

```bash
# Default: Auto picks model and GPU strategy
make finetune-smart

# Or specify model explicitly
python scripts/smart_finetune.py --model mistral
python scripts/smart_finetune.py --model mistral-nemo  # 12B, DDP with 2 GPUs
```

### Background Training (Recommended for Long Runs)

训练大模型需要数小时，建议使用后台模式防止终端关闭导致中断：

```bash
# 一键后台训练 (使用 tmux，自动写日志)
make finetune-background

# 指定模型 + DeepSpeed + 日志
make finetune-background MODEL=llama3.3 DEEPSPEED=1 LOG=logs/llama3.3-deepspeed.log

# 或手动指定参数
python scripts/smart_finetune.py --background
python scripts/smart_finetune.py --model llama3.3 --deepspeed --background
python scripts/smart_finetune.py --model llama3.3 --no-deepspeed --background

# 跳过确认提示 (适合脚本/自动化)
python scripts/smart_finetune.py --model mistral -y --background
```

**管理后台训练：**
```bash
# 查看训练进度
tmux attach -t gswa-training

# 脱离会话 (训练继续运行): 按 Ctrl+B，然后按 D

# 停止训练
tmux kill-session -t gswa-training

# 检查是否有训练在运行
tmux list-sessions
```

**注意：** 需要安装 tmux (`apt install tmux` 或 `yum install tmux`)

**如果日志里一直显示 0%**（tqdm 不刷新）：
```bash
python scripts/finetune_lora.py --disable-tqdm --log-every 1
```

---

### 70B+ Models (Advanced)

For 70B+ models on multi-GPU, you need **DeepSpeed ZeRO-3**.
`smart_finetune.py` will auto-enable DeepSpeed unless you pass `--no-deepspeed`.

**Requirements:**
- 2+ GPUs with total 60GB+ VRAM
- Matching CUDA versions (PyTorch CUDA version = system CUDA version)
- Large CPU RAM (256GB+ recommended for CPU offloading)

```bash
# Explicit DeepSpeed mode for 70B models
python scripts/smart_finetune.py --model llama3.3 --deepspeed

# Direct DeepSpeed command
accelerate launch --config_file config/accelerate_deepspeed.yaml \
    scripts/finetune_deepspeed.py --model meta-llama/Llama-3.3-70B-Instruct
```

**If DeepSpeed fails with CUDA mismatch:**
- Check: `nvcc --version` vs `python -c "import torch; print(torch.version.cuda)"`
- If they differ, use a smaller model instead (e.g., Mistral 7B)

**Install DeepSpeed:**
```bash
./scripts/setup.sh --cuda
# Or manually
pip install deepspeed accelerate pyyaml
```

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

| Model | VRAM Required | Quality | Notes |
|-------|--------------|---------|-------|
| `mistralai/Mistral-7B-Instruct-v0.3` | 16GB | Very Good | **RECOMMENDED** - Stable, no login |
| `mistralai/Mistral-Nemo-Instruct-2407` | 24GB | Excellent | Good for academic writing |
| `Qwen/Qwen2.5-7B-Instruct` | 16GB | Very Good | No login required |
| `microsoft/Phi-3.5-mini-instruct` | 8GB | Good | Fast training |
| `Qwen/Qwen2.5-1.5B-Instruct` | 4GB | Basic | Minimum viable |
| `mistralai/Mistral-Large-Instruct-2407` | 48GB | Excellent | Needs more VRAM |
| `meta-llama/Llama-3.3-70B-Instruct` | 64GB+ | Best | Requires DeepSpeed + HF login |

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

## Post-Training Visualization (CUDA/Transformers)

After training with `finetune_lora.py` or `smart_finetune.py`, plots are automatically
generated in the `Parameter_Tuning/` folder of the model output directory.

### Output Structure

```
models/gswa-lora-Mistral-20260123-012408/
├── Parameter_Tuning/
│   ├── loss_curve.png          # Train + eval loss with hyperparameter annotations
│   ├── learning_rate.png       # LR schedule (warmup + cosine decay)
│   ├── grad_norm.png           # Gradient norms over training
│   ├── training_summary.png    # Combined 2x2 grid of all metrics
│   └── training_report.txt     # Text summary with loss stats
├── training_metrics.json       # Complete training log_history
├── training_config.json        # Hyperparameters used
└── checkpoint-*/
    └── trainer_state.json      # HuggingFace Trainer state
```

### Manual Visualization

```bash
# Generate plots for a completed training run
make visualize MODEL_DIR=models/gswa-lora-Mistral-20260123-012408

# Or directly:
python scripts/plot_training.py models/gswa-lora-Mistral-20260123-012408/

# Compare multiple runs (overlaid loss curves)
make compare-runs
# Or: python scripts/plot_training.py models/gswa-lora-*/ --compare
```

### Model Evaluation

Generate text samples to assess style quality:

```bash
# Quick evaluation (5 samples from validation set)
make evaluate MODEL_DIR=models/gswa-lora-Mistral-20260123-012408

# More samples with custom settings
python scripts/evaluate_model.py models/gswa-lora-Mistral-20260123-012408/ \
    --num-samples 10 --max-new-tokens 512

# Custom prompts
python scripts/evaluate_model.py models/gswa-lora-Mistral-20260123-012408/ \
    --prompts-file my_prompts.jsonl
```

### Interpreting Results

- **Loss Curve**: Train loss should decrease smoothly. If eval loss diverges upward while train loss continues dropping, the model is overfitting.
- **Train-Eval Gap**: Gap < 0.15 = good generalization; Gap > 0.3 = likely overfitting (try fewer epochs or more data)
- **Learning Rate**: Verify warmup phase (~10% of training) and smooth cosine decay
- **Gradient Norm**: Spikes indicate instability. Consistent growth may indicate exploding gradients (reduce LR).
- **Comparison Plots**: Lower final eval loss generally indicates better generalization.

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

## Complete Workflow (Style Transfer Training)

GSWA 的训练采用 **style-transfer pair** 方法：使用本地 LLM 生成 Gilles 原始段落的"通用"版本，
然后训练模型学习从通用学术英语转换为 Gilles 风格。

### 训练数据格式

每条训练数据的结构（使用模型原生 chat template）：

```
<s>[INST]Rewrite the following scientific paragraph in a clear, precise academic style:

<通用学术英文版本>[/INST]<Gilles 原始风格文本></s>
```

- `[INST]...[/INST]` 部分：instruction + input（被 mask，不参与 loss 计算）
- `[/INST]` 之后的部分：response（模型实际学习生成的部分）

### Step 1: 准备语料库

```bash
# 放入 Gilles 的论文 (PDF/DOCX/TXT)
data/corpus/raw/                      <- 普通文章
data/corpus/raw/important_examples/   <- 重要文章 (2.5x 权重)

# 解析语料库
make parse-corpus
```

### Step 2: 生成 Style-Transfer Pairs

这一步使用本地 LLM (ollama) 将 Gilles 的每个段落生成一个"通用"版本。
支持断点续传（可随时中断并重新运行）。

```bash
# 使用 qwen3-coder:30b (推荐，速度快)
make generate-pairs OLLAMA_MODEL=qwen3-coder:30b

# 或使用 llama3:70b (质量更高，但慢 5 倍)
make generate-pairs OLLAMA_MODEL=llama3:70b

# 直接运行脚本（更多选项）
python scripts/prepare_training_data.py --generate-pairs \
    --ollama-model qwen3-coder:30b \
    --max-para-length 1500

# 监控进度
tail -f /tmp/pair_generation.log  # 如果后台运行
```

**生成时间参考：**
| 模型 | 速度 | 2968 段落预计时间 |
|------|------|-------------------|
| qwen3-coder:30b | ~5s/段落 | ~4 小时 |
| llama3:70b | ~30s/段落 | ~25 小时 |

**原理：** 长文本块会自动拆分为 100-1500 字符的独立段落。
约 1134 个原始文本块 → 拆分后约 2968 个训练段落。

### Step 3: 准备训练数据

```bash
# 从 style pairs 生成 Alpaca 格式训练数据
make prepare-training

# 或手动运行（带权重和验证集拆分）
python scripts/prepare_training_data.py --format alpaca --weighted --split
```

输出文件：
- `data/training/alpaca_train.jsonl` - 训练集
- `data/training/alpaca_val.jsonl` - 验证集

**数据质量保障：** 脚本自动过滤以下内容：
- **参考文献列表** (Reference sections) - 包含多个 "Author et al. (YYYY)" 模式的段落
- **DOI 和文献条目** - 带有 "10.xxxx/xxx" 或 journal volume-page 格式的内容
- **过短段落** - 少于 100 字符的片段

过滤后训练数据中的参考文献内容 < 1%，确保模型学习写作风格而非记忆文献格式。

### Step 4: 训练模型

```bash
# 一键智能训练（推荐，自动检测硬件和参数）
make finetune-smart

# 后台训练（长时间运行推荐）
make finetune-background

# 查看训练进度
tmux attach -t gswa-training
```

### Step 4b: 风格增强训练（Style-Enhanced Mode）

标准训练（Step 4）使用单段落对学习基本词汇和句法风格。风格增强模式额外学习：
- **转折模式** (transition patterns) - 段间过渡和话语标记词
- **论证思路** (argument structure) - 跨段落逻辑推进
- **铺垫手法** (foreshadowing) - 设悬-揭示模式

```bash
# 一键风格增强训练（自动生成多段落训练数据 + 大LoRA rank训练）
make finetune-style-enhanced

# 后台运行
make finetune-style-enhanced-bg

# 或分步执行：
# 1. 生成多段落上下文窗口训练数据
python scripts/prepare_training_data.py --format context-window --section-aware --split

# 2. 用风格增强模式训练
python scripts/smart_finetune.py --style-enhanced
python scripts/smart_finetune.py --style-enhanced --model mistral-nemo --background
```

**风格增强模式与标准模式的区别：**

| 参数 | 标准模式 | 风格增强模式 |
|------|---------|-------------|
| LoRA rank | 16 | 32 |
| LoRA alpha | 32 | 64 |
| Max序列长度 | 1024-2048 | 4096 |
| 训练轮数 | 3 | 4 |
| 训练数据 | 单段落对 | 多段落上下文窗口（默认3段） |
| Section标签 | 无 | 自动检测并加入prompt |
| 学习维度 | 用词+句法 | 用词+句法+转折+部分思路 |

**自定义上下文窗口大小：**
```bash
# 使用2段落窗口（适合小数据集或VRAM不足）
python scripts/prepare_training_data.py --format context-window --context-window 2 --section-aware --split

# 使用5段落窗口（需要更多VRAM和更长训练）
python scripts/prepare_training_data.py --format context-window --context-window 5 --section-aware --split
```

**注意：** 风格增强模式需要更多VRAM（max_length=4096），建议24GB+显存。如果OOM，减小context-window为2或降低batch_size。

### Step 5: 评估和可视化

```bash
# 查看训练曲线
make visualize MODEL_DIR=models/gswa-lora-Mistral-<timestamp>

# 生成样本评估风格质量
make evaluate MODEL_DIR=models/gswa-lora-Mistral-<timestamp>

# 多次训练对比
make compare-runs
```

### Step 6: 使用模型

```bash
# 配置 .env 使用新模型
LORA_ADAPTER_PATH=./models/gswa-lora-Mistral-<timestamp>

# 启动服务
make run
```

### Label Masking 机制

训练时只有 response 部分（Gilles 风格文本）参与 loss 计算：

```
Token:  <s> [INST] Rewrite...  generic_text [/INST] gilles_text </s>  [PAD]...
Label:  -100  -100   -100...    -100         -100    token_ids   EOS   -100...
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^
        masked (不参与训练)                            trainable (学习生成)
```

这确保模型的全部学习容量用于学习如何生成 Gilles 风格的文本。

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
| `make finetune-smart` | Smart training (auto-selects backend) |
| `make finetune-style-enhanced` | Style-enhanced (multi-paragraph, rank=32, 4096 context) |
| `make finetune-style-enhanced-bg` | Style-enhanced in background (tmux) |
| `make finetune-deepspeed` | DeepSpeed ZeRO-3 for multi-GPU 70B+ |
| `make finetune-background` | Background training (tmux, survives terminal close) |
| `make check-lora` | Check LoRA dependencies |

### Utility Commands

| Command | Description |
|---------|-------------|
| `make analyze-data` | Analyze training data for long sequences |
| `make preprocess-data` | Preprocess data to split long sequences |
| `make prepare-training` | Prepare training data from corpus |
| `make prepare-style-enhanced` | Prepare multi-paragraph context-window data |
| `make parse-corpus` | Parse corpus files (PDF/DOCX/TXT) |
