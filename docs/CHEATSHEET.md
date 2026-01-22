# GSWA 快速参考 / Quick Reference

## 一键操作 (One-Click Workflow)

```bash
# Step 1: 安装 (Setup) - 只需运行一次
make setup-cuda-auto    # Linux GPU (全自动)
make setup-auto         # Mac (全自动)

# Step 2: 激活环境 (Activate)
micromamba activate gswa

# Step 3: 训练 (Train)
make finetune-smart     # 自动检测平台和硬件

# Step 4: 运行 (Run)
make run                # http://localhost:8080
```

---

## 环境说明 (重要!)

**使用 micromamba 后不需要 venv！**

```bash
# 正确 ✓ - 只有 micromamba 环境
(gswa) $ make finetune-smart

# 错误 ✗ - 同时激活了 venv 和 micromamba
(/home/public_data/conda_envs/gswa) (venv) $ ...

# 如果看到上面的情况，先退出 venv
deactivate              # 退出 venv
micromamba activate gswa  # 只用 micromamba
```

---

## 常用命令 (Common Commands)

| 命令 | 说明 |
|------|------|
| `make help` | 显示所有命令 |
| `make train-info` | 查看硬件信息 |
| `make finetune-smart` | 一键训练 (自动选择后端) |
| `make finetune-deepspeed` | 多卡70B+训练 (DeepSpeed) |
| `make finetune-background` | **后台训练 (关闭终端不中断)** |
| `make run` | 启动服务 |
| `make test` | 运行测试 |

---

## 环境操作 (Environment)

```bash
# 激活 micromamba 环境（推荐）
micromamba activate gswa

# 不激活直接运行命令
micromamba run -n gswa make test
micromamba run -n gswa make run

# 退出环境
micromamba deactivate
```

---

## 常见问题快速解决

### `_ctypes` 模块缺失
```bash
# 使用 micromamba 代替 pyenv
make setup-cuda-auto
```

### CUDA 未检测到
```bash
# 检查
python -c "import torch; print(torch.cuda.is_available())"

# 重装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 内存不足 (OOM)
```bash
# finetune-smart 会自动 OOM 回退
make finetune-smart
```

### 多卡训练失败
```bash
# 问题：多卡 QLoRA 可能不稳定
# 解决：默认使用单卡模式 (Mistral 7B)

# 推荐：使用默认设置 (Mistral 7B，稳定)
make finetune-smart  # 自动选择 Mistral 7B

# 或手动指定模型
python scripts/smart_finetune.py --model mistral

# 70B 模型需要 DeepSpeed (高级)
# 注意：需要 CUDA 版本匹配
python scripts/smart_finetune.py --model llama3.3 --deepspeed
```

### 后台训练 (Background Training)
```bash
# 问题：关闭终端或 SSH 断开导致训练中断
# 解决：使用 --background 在 tmux 中后台运行

# 一键后台训练
make finetune-background

# 或手动指定参数
python scripts/smart_finetune.py --model llama3.3 --deepspeed --background

# 查看训练进度
tmux attach -t gswa-training

# 脱离会话 (训练继续运行)
# 按 Ctrl+B，然后按 D

# 停止训练
tmux kill-session -t gswa-training
```

---

## 文件位置

| 用途 | 路径 |
|------|------|
| 语料文件 | `data/corpus/raw/` |
| 重要语料 | `data/corpus/raw/important_examples/` |
| 训练输出 | `runs/` 或 `models/` |
| 配置文件 | `.env` |

---

## 模型选择 (Model Selection)

### 推荐模型（按 VRAM）

| VRAM | 推荐模型 | 命令 |
|------|----------|------|
| 16GB+ | **Mistral 7B** (推荐) | `make finetune-smart` (默认) |
| 24GB+ | Mistral Nemo 12B | `make finetune-smart --model mistral-nemo` |
| 48GB+ | Mistral Large | `make finetune-smart --model mistral-large` |
| 8GB+ | Phi-3.5 Mini | `make finetune-smart --model phi` |
| 60GB+ | Llama 3.3 70B (高级) | `--model llama3.3 --deepspeed` |

### 模型快捷名

```bash
# 无需登录（Ungated）
mistral, mistral-large, mistral-nemo, qwen, qwen-14b, phi

# 需要 HuggingFace 登录（Gated）
llama3.3      # Llama 3.3 70B - 最佳英文写作
llama3-8b     # Llama 3.1 8B - 快速测试
gemma         # Google Gemma 2 9B
```

### 使用 Gated 模型

```bash
# 1. 登录 HuggingFace
huggingface-cli login

# 2. 访问模型页面接受许可
# https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

# 3. 训练
make finetune-smart
```

---

## 详细文档

- 安装指南: [docs/INSTALL.md](INSTALL.md)
- 训练指南: [docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md)
