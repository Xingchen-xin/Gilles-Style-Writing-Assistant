# GSWA Fine-tuning Guide / 微调指南

## TL;DR 傻瓜式操作 (4 Steps)

```bash
# 只需 4 步 / Just 4 steps:

# 1. 放文章到文件夹 / Add your documents
#    data/corpus/raw/                    <- 普通文章 / Regular articles
#    data/corpus/raw/important_examples/ <- 重要文章 (2.5x权重) / Important examples

# 2. 生成风格转换对 / Generate style-transfer pairs (一次性，支持断点续传)
make parse-corpus
make generate-pairs OLLAMA_MODEL=qwen3-coder:30b

# 3. 一键智能训练 / One-click smart training
make finetune-smart

# 4. 按照输出提示完成配置 / Follow the output instructions
```

### 🚀 智能训练特性 / Smart Training Features

- **自动检测平台**: Mac → MLX, Linux/Windows → LoRA
- **自动检测硬件**: GPU型号、显存大小、系统内存
- **自动选择参数**: batch_size, learning_rate, 量化等级
- **自动推荐模型**: 根据硬件推荐最佳基底模型

---

## 文件夹结构说明

```
data/corpus/raw/                      <- 普通 Gilles 文章
├── paper1.pdf
├── paper2.docx
├── paper3.txt
│
└── important_examples/               <- 重要/代表性文章 (自动 2.5x 权重)
    ├── best_review.pdf
    └── classic_paper.pdf
```

**权重说明：**
| 位置 | 自动权重 | 说明 |
|------|----------|------|
| `raw/` | 1.0x | 普通文章 |
| `raw/important_examples/` | 2.5x | 重要文章，训练时出现更多次 |

**支持的文件格式：** `.pdf`, `.docx`, `.txt`

---

## 为什么需要微调？

当前问题：
1. **AI 检测器识别** - 生成的文本被识别为纯 AI 生成
2. **风格不匹配** - 输出不像 Gilles 的写作风格
3. **通用性过强** - 模型没有学习 Gilles 特有的表达方式

解决方案：**Style-Transfer Fine-tuning (风格转换微调)**

### 训练原理

使用 **Approach B: Synthetic Pairs** 方法：
1. 用本地 LLM 将 Gilles 的每个段落"简化"为通用学术英语
2. 训练模型学习从{通用输入 → Gilles 风格输出}的映射
3. 使用模型原生 chat template (`[INST]...[/INST]`) 确保训练和推理格式一致
4. Label masking 确保只训练 response tokens

**示例:**
```
Input (通用):  "SEM analysis confirmed earlier aerial hyphae development in the mutant."
Output (Gilles): "The precocious erection of aerial hyphae in the redD mutant was confirmed
                  by scanning electron microscopy (SEM)."
```

模型学到的转换：
- "earlier development" → "precocious erection" (精确、生动的词汇)
- 被动句 → 复杂从属结构
- 添加 discourse markers (Indeed, Notably, Together)

---

## 跨平台支持 / Multi-Platform Support

| 平台 | 训练方式 | 检测命令 | 说明 |
|------|----------|----------|------|
| **Mac** (M1/M2/M3/M4) | MLX | `make check-mlx` | Apple Silicon 专用优化 |
| **Linux** (NVIDIA GPU) | LoRA/QLoRA | `make check-lora` | CUDA 加速训练 |
| **Windows** (NVIDIA GPU) | LoRA/QLoRA | `make check-lora` | 需安装 CUDA |
| **无 GPU** | CPU LoRA | - | 非常慢，仅供测试 |

## 推荐基底模型 / Recommended Base Models

| 显存/内存 | 推荐模型 | 说明 |
|-----------|----------|------|
| 8GB | `Qwen/Qwen2.5-1.5B-Instruct` | 最小可用，基础质量 |
| 16GB | `mistralai/Mistral-7B-Instruct-v0.3` | 推荐入门用户 |
| 24GB+ | `mistralai/Mistral-Nemo-Instruct-2407` | **推荐** - 12B 模型，最佳性价比 |
| 48GB+ | `mistralai/Mistral-Large-Instruct-2407` | 高质量输出 |
| 60GB+ | `meta-llama/Llama-3.3-70B-Instruct` | 可选 (需要 `--model llama3.3`) |

**为什么推荐 Mistral-Nemo 12B?**
- **模型容量与数据量匹配**: 12B 参数对 ~1000 样本更合适，避免过拟合
- **可用更大 batch size**: batch=4 vs 70B 的 batch=1，梯度更稳定
- **训练速度快**: 比 70B 快 3-4 倍
- **英文学术写作质量优秀**: 在科学写作任务上表现出色
- **支持长上下文**: 32K tokens

**关于 70B+ 大模型**
- 70B 模型适合数据量充足 (>5000 样本) 的场景
- 对于 ~1000 样本的数据集，12B 模型通常效果更好
- 如需使用 70B，请显式指定: `--model llama3.3`
- 多 GPU 系统会自动启用 DeepSpeed ZeRO-3

```bash
# 自动选择最佳模型（推荐 Mistral-Nemo 12B）
make finetune-smart

# 后台运行（保存日志）
python scripts/smart_finetune.py --background -y

# 手动指定 70B 模型（需要大量数据）
python scripts/smart_finetune.py --model llama3.3 -y

# 查看训练日志
tail -f logs/finetune-background-*.log
```

**关于 Mistral tokenizer 警告**
- Mistral 系列模型的 tokenizer 警告已自动处理
- 脚本会自动应用 `fix_mistral_regex=True` 并抑制警告

## 训练参数说明

| 参数 | Mistral-Nemo 12B | Llama 70B | 说明 |
|------|------------------|-----------|------|
| batch_size | 4 | 1 | 更大 batch = 更稳定梯度 |
| gradient_accumulation | 4 | 8 | 有效 batch = batch × accum |
| lora_r | 32 | 16 | LoRA 秩，越大容量越大 |
| lora_alpha | 64 | 32 | 通常 = 2 × lora_r |
| learning_rate | 1e-4 | 5e-5 | QLoRA 标准值 |
| epochs | 3 | 1-2 | 根据数据量调整 |
| max_length | 2048 | 1024 | 学术文章通常较长 |

## 微调方案对比

| 方案 | 硬件要求 | 训练时间 | 质量 | 难度 | 推荐场景 |
|------|----------|----------|------|------|----------|
| **MLX (Mac)** | M1/M2/M3 16GB+ | 1-2小时 | ⭐⭐⭐⭐ | 低 | **Mac 用户首选** |
| **LoRA** | GPU 16GB+ | 2-4小时 | ⭐⭐⭐⭐ | 中 | Linux/Windows |
| **QLoRA** | GPU 8GB+ | 3-6小时 | ⭐⭐⭐ | 中 | 显存有限 |
| **Full Fine-tuning** | GPU 48GB+ | 8-24小时 | ⭐⭐⭐⭐⭐ | 高 | 最佳质量 |

---

## Mac 用户傻瓜式教程

### 第一步：放入文章

1. 打开 Finder，进入项目目录
2. 打开 `data/corpus/raw/` 文件夹
3. 把 Gilles 的 PDF 文章拖进去
4. 如果有最能代表 Gilles 风格的文章，放入 `raw/important_examples/`

```bash
# 或者用命令行
cp ~/Downloads/*.pdf data/corpus/raw/

# 重要文章放这里
cp ~/Downloads/important_paper.pdf data/corpus/raw/important_examples/
```

### 第二步：安装依赖（首次运行）

```bash
# 安装 MLX（Apple Silicon 专用机器学习库）
pip install mlx mlx-lm
```

### 第三步：一键微调

```bash
# 这一条命令完成所有工作：解析文章 → 生成训练数据 → 微调模型
make finetune-all
```

**看到的输出：**
```
============================================================
GSWA Corpus Parser
============================================================
Input (regular):  ./data/corpus/raw
Input (priority): ./data/corpus/raw/important_examples

Found 15 documents:
  - Regular articles:  12
  - Priority articles: 3 (in important_examples/)

Processing: paper1.pdf...
  Extracted 45 paragraphs
Processing: best_review.pdf ⭐...
  Extracted 120 paragraphs
...

============================================================
Starting MLX Fine-tuning
============================================================
Epoch 1/3: loss=2.45
Epoch 2/3: loss=1.89
Epoch 3/3: loss=1.23

Model saved to: models/gswa-mlx-mistral/
```

### 第四步：创建 Ollama 模型

```bash
# 根据输出的模型路径创建 Ollama 模型
ollama create gswa-gilles -f models/gswa-mlx-mistral/Modelfile
```

### 第五步：更新配置并运行

```bash
# 更新 .env 使用新模型
echo "VLLM_MODEL_NAME=gswa-gilles" >> .env

# 重启 GSWA
make run
```

**恭喜！现在 GSWA 使用的是微调后的模型！**

---

## Linux 用户傻瓜式教程

### 第一步：放入文章

同 Mac 用户，放入 `data/corpus/raw/` 和 `raw/important_examples/`

### 第二步：安装依赖

```bash
# 安装训练依赖
make setup-cuda-auto
# 或手动
micromamba create -n gswa python=3.11 -y && micromamba activate gswa
pip install -e ".[dev,similarity]" pymupdf
```

### 第三步：生成 Style-Transfer Pairs

```bash
# 解析语料库
make parse-corpus

# 生成风格对 (一次性操作，支持断点续传，~4小时)
make generate-pairs OLLAMA_MODEL=qwen3-coder:30b

# 或后台运行
nohup micromamba run -n gswa python -u scripts/prepare_training_data.py \
    --generate-pairs --ollama-model qwen3-coder:30b > /tmp/pair_generation.log 2>&1 &
tail -f /tmp/pair_generation.log  # 监控进度
```

### 第四步：训练模型

```bash
# 一键智能训练（自动检测GPU并选择参数）
make finetune-smart

# 后台训练（推荐，关闭终端不中断）
make finetune-background
```

### 第五步：评估和部署

```bash
# 评估模型效果
make evaluate MODEL_DIR=models/gswa-lora-Mistral-<timestamp>

# 查看训练曲线
make visualize MODEL_DIR=models/gswa-lora-Mistral-<timestamp>

# 部署：配置 .env 使用 LoRA adapter
LORA_ADAPTER_PATH=./models/gswa-lora-Mistral-<timestamp>
```

---

## Windows 用户傻瓜式教程

### 前置要求

1. **NVIDIA GPU** (8GB+ VRAM)
2. **CUDA Toolkit** (推荐 12.1+)
3. **Python 3.10+**

### 第一步：安装 CUDA

1. 下载 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. 安装并重启
3. 验证: `nvidia-smi`

### 第二步：安装 PyTorch with CUDA

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate bitsandbytes-windows
```

### 第三步：放入文章

同 Mac/Linux 用户，放入 `data/corpus/raw/` 和 `raw/important_examples/`

### 第四步：一键智能微调

```powershell
# 在 PowerShell 或 CMD 中运行
python scripts/smart_finetune.py
```

或者使用 make (需安装 [GNU Make for Windows](http://gnuwin32.sourceforge.net/packages/make.htm)):
```powershell
make finetune-smart
```

---

## 硬件配置文件（自动检测）

系统会自动检测你的硬件并选择最佳训练参数：

| 硬件 | 内存 | batch_size | num_layers | iters |
|------|------|------------|------------|-------|
| M1/M2/M3 8GB | 8GB | 1 | 4 | 300 |
| M1/M2/M3 16GB | 16GB | 2 | 8 | 500 |
| M1/M2/M3 Max 32GB+ | 32GB+ | 4 | 16 | 1000 |
| M1/M2/M3 Ultra 64GB+ | 64GB+ | 8 | 32 | 1500 |

**自定义配置：** 编辑 `config/training_profiles.json` 文件。

**查看系统检测结果：**
```bash
python scripts/finetune_mlx_mac.py --auto --check-only
```

---

## 手动配置权重（高级）

如果你想精确控制每篇文章的权重，可以编辑 `data/corpus/priority_weights.json`：

```json
{
  "default_weight": 1.0,
  "priority_folder_weight": 2.5,

  "priority_docs": {
    "Barka_MicrobiolMolBiolRev2016": {
      "weight": 3.0,
      "reason": "最能代表 Gilles 风格的综述"
    }
  },

  "exclude_docs": {
    "some_bad_paper": {
      "reason": "太短，不能代表风格"
    }
  }
}
```

**查看所有文章 ID：**
```bash
make list-docs
```

---

## 完整 Makefile 命令

```bash
# === 语料管理 ===
make corpus            # 查看语料库状态
make corpus-guide      # 显示添加文件指南
make corpus-validate   # 验证所有语料文件
make parse-corpus      # 解析 raw/ 中的文章
make list-docs         # 列出所有文章 ID
make training-stats    # 查看训练数据统计

# === 数据准备 ===
make generate-pairs    # 生成 style-transfer pairs (一次性，~4小时)
make prepare-training  # 从 pairs 生成 Alpaca 格式训练数据

# === 智能训练 ===
make finetune-smart    # 一键智能训练（自动检测平台和硬件）
make finetune-background  # 后台训练（关闭终端不中断）
make finetune-all      # Mac 一键训练（parse + prepare + mlx）

# === 评估和可视化 ===
make visualize MODEL_DIR=models/gswa-lora-...  # 训练曲线
make evaluate MODEL_DIR=models/gswa-lora-...   # 生成样本评估
make compare-runs      # 多次训练对比

# === 分步训练 ===
make finetune-mlx      # Mac MLX 微调
make finetune-lora     # Linux/Windows LoRA 微调
make finetune-deepspeed  # 多卡 70B+ 模型

# === 环境检查 ===
make check-mlx         # 检查 MLX 依赖 (Mac)
make check-lora        # 检查 LoRA 依赖 (Linux/Windows)
make train-info        # 查看硬件信息和推荐
```

---

## 如何减少 AI 检测？

微调后的模型会更好地模仿人类写作风格，但还可以采取以下措施：

### 1. 使用高质量语料

- 放入更多 Gilles 的文章（越多越好）
- 把最能代表风格的放入 `important_examples/`
- 排除不典型的文章

### 2. 调整生成参数

在 `.env` 中设置：
```bash
TEMPERATURE_BASE=0.4      # 略高的温度增加变化
TEMPERATURE_VARIANCE=0.2  # 变体间更大差异
```

### 3. 后处理

- 轻微编辑生成的文本
- 添加个人表达
- 调整句子结构

---

## DPO 进阶训练（偏好对齐）

使用后，你可以通过反馈进一步优化：

1. 使用 GSWA 生成变体
2. 在 UI 中为变体评分（Best/Good/Bad）
3. 提交反馈
4. 导出并训练：

```bash
make export-dpo
python scripts/prepare_training_data.py --format dpo --from-feedback
make finetune-lora
```

---

## 故障排除

### Q: 没有检测到文章？

A: 检查文件位置和格式：
```bash
ls data/corpus/raw/
ls data/corpus/raw/important_examples/
```
确保是 `.pdf`, `.docx`, 或 `.txt` 文件。

### Q: MLX 训练太慢？

A: 减少迭代次数或使用更小的模型：
```bash
python scripts/finetune_mlx_mac.py --model phi --iters 500
```

### Q: 内存不足 (Mac)?

A: 系统会自动检测你的硬件并选择合适的配置。如需手动调整：
```bash
# 查看可用配置
python scripts/finetune_mlx_mac.py --list-profiles

# 使用保守配置（最低内存）
python scripts/finetune_mlx_mac.py --profile conservative

# 或者手动设置参数
python scripts/finetune_mlx_mac.py --batch-size 1 --num-layers 4 --max-seq-length 512
```

### Q: 显存不足 (CUDA OOM)？

A: 使用 4-bit 量化：
```bash
python scripts/finetune_lora.py --quantize 4bit --batch-size 1
```

### Q: 训练卡在 0%？/ Training stuck at 0%?

A: 可能的原因和解决方案：

1. **Mistral 模型兼容性问题** - 已在最新版本中修复
   ```bash
   git pull  # 更新到最新版本
   ```

2. **显存不足** - 尝试使用更小的模型
   ```bash
   python scripts/smart_finetune.py --model mistral  # 使用 7B 模型
   ```

3. **多 GPU 冲突** - 强制使用单卡
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/finetune_lora.py --model mistral
   ```

4. **梯度检查点问题** - 脚本已自动处理 Mistral 模型的兼容性

5. **日志中只显示 0%** - tqdm 在日志文件中不会持续刷新
   ```bash
   # 关闭 tqdm 并强制每步输出
   python scripts/finetune_lora.py --disable-tqdm --log-every 1
   ```

### Q: 生成质量下降？

A: 可能是过拟合，尝试：
- 减少训练轮数
- 增加更多文章
- 使用验证集

### Q: 如何回滚到原模型？

A: 修改 `.env`：
```bash
VLLM_MODEL_NAME=mistral  # 使用原始模型
```

---

## 参考资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [MLX 文档](https://ml-explore.github.io/mlx/)
- [Ollama 模型创建](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
