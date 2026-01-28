# GSWA 快速参考 / Quick Reference

## 一键操作 (Complete Workflow)

```bash
# Step 1: 安装 (Setup) - 只需运行一次
make setup-cuda-auto    # Linux GPU (全自动)
make setup-auto         # Mac (全自动)

# Step 2: 激活环境 (Activate)
micromamba activate gswa

# Step 3: 准备数据 (Prepare) - 放入 Gilles 论文后运行
make parse-corpus                                  # 解析 PDF/DOCX
make generate-pairs OLLAMA_MODEL=qwen3-coder:30b   # 生成风格对 (~4小时,一次性)

# Step 4: 训练 (Train)
make finetune-smart     # 自动检测平台和硬件

# Step 5: 评估 (Evaluate)
make evaluate MODEL_DIR=models/gswa-lora-Mistral-<timestamp>
make visualize MODEL_DIR=models/gswa-lora-Mistral-<timestamp>

# Step 6: 运行 (Run)
make serve              # 一键启动! vLLM + API + 登录保护
                        # 默认账号: gilles / IBLGilles2026
                        # 访问: http://<IP>:8080

# 公网访问
make tunnel             # 创建 Cloudflare 隧道 (任何地方访问)
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
| `make parse-corpus` | 解析语料库 (PDF/DOCX/TXT) |
| `make generate-pairs` | 生成风格转换对 (一次性, ~4小时) |
| `make finetune-smart` | 一键训练 (自动选择后端) |
| `make finetune-background` | **后台训练 (关闭终端不中断)** |
| `make visualize MODEL_DIR=...` | 训练曲线可视化 |
| `make evaluate MODEL_DIR=...` | 生成样本评估 |
| `make compare-runs` | 多次训练对比 |
| `make serve` | **一键启动服务 (vLLM + API + 登录)** |
| `make tunnel` | 创建公网隧道 (Cloudflare) |
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

## 服务器部署 (Server Deployment)

```bash
# 一键启动 (带登录保护)
make serve
# 默认账号: gilles
# 默认密码: IBLGilles2026
# 访问: http://<服务器IP>:8080

# 自定义账号密码
make serve-secure USER=admin PASS=mypassword

# 公网访问 (通过 Cloudflare 隧道)
make tunnel
# 会显示类似: https://xxx-xxx.trycloudflare.com

# 无密码模式 (仅本地开发)
make serve-noauth
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
# 解决：70B 默认 DeepSpeed，其他默认单卡

# 推荐：使用默认设置 (自动选择)
make finetune-smart

# 或手动指定模型
python scripts/smart_finetune.py --model mistral

# 70B 模型默认 DeepSpeed (注意 CUDA 版本匹配)
python scripts/smart_finetune.py --model llama3.3 --deepspeed
python scripts/smart_finetune.py --model llama3.3 --no-deepspeed  # 强制单卡
```

### 后台训练 (Background Training)
```bash
# 问题：关闭终端或 SSH 断开导致训练中断
# 解决：使用 --background 在 tmux 中后台运行

# 一键后台训练
make finetune-background

# 指定模型 + DeepSpeed + 日志
make finetune-background MODEL=llama3.3 DEEPSPEED=1 LOG=logs/llama3.3-deepspeed.log

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
| 语料文件 (PDF/DOCX) | `data/corpus/raw/` |
| 重要语料 (2.5x 权重) | `data/corpus/raw/important_examples/` |
| 解析后语料 | `data/corpus/parsed/corpus.jsonl` |
| 风格转换对 | `data/training/style_pairs.jsonl` |
| 训练数据 | `data/training/alpaca_train.jsonl` |
| 模型输出 | `models/gswa-lora-*/` |
| 训练曲线 | `models/gswa-lora-*/Parameter_Tuning/` |
| 评估结果 | `models/gswa-lora-*/eval_results.txt` |
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
