# GSWA 训练指南 - 傻瓜式一键教程

## 环境配置（首先阅读！）

### Linux 服务器（无 sudo 权限）

在服务器上训练模型前，请先配置好环境：

```bash
# 一键安装（推荐）
make setup-cuda-auto    # 全自动安装，支持 CUDA

# 安装完成后激活环境
micromamba activate gswa

# 验证 CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 常见问题：`_ctypes` 模块缺失

如果看到以下错误：
```
ModuleNotFoundError: No module named '_ctypes'
```

**这是什么问题？**
- pyenv 编译的 Python 缺少 `libffi` 库支持
- 服务器没有 sudo 权限，无法安装 `libffi-devel`

**怎么解决？**
```bash
# 使用 micromamba（自带完整 Python，不需要系统库）
curl -L micro.mamba.pm/install.sh | bash
source ~/.bashrc

# 创建环境
micromamba create -n gswa python=3.11 -y
micromamba activate gswa

# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装项目
pip install -e ".[dev,similarity]" pymupdf
```

### Mac 用户

Mac 用户可以直接使用一键安装：
```bash
make setup      # 自动检测并配置环境
source venv/bin/activate
```

详细安装指南：[INSTALL.md](INSTALL.md)

---

## 快速开始（解决内存不足问题）

如果你遇到了 `Insufficient Memory` 或 `OOM` 错误，请使用以下命令：

```bash
# 一键傻瓜式训练（推荐）
make train-safe
```

这个命令会自动：
1. 解析你的语料文件
2. 准备训练数据
3. **自动分割过长的文本序列**
4. **使用保守的内存设置**
5. **OOM时自动重试并降低配置**

---

## 常见问题解决方案

### 问题 1: `Insufficient Memory` / `kIOGPUCommandBufferCallbackErrorOutOfMemory`

**原因**：训练数据包含过长的序列，超出了GPU内存限制。

**解决方案**：

```bash
# 方法 1: 使用内存安全模式（推荐）
make train-safe

# 方法 2: 手动预处理数据
make analyze-data          # 查看数据统计
make preprocess-data       # 自动分割长序列
make finetune-mlx-safe     # 内存安全训练

# 方法 3: 指定最大token长度
make preprocess-data MAX_TOKENS=1024
```

### 问题 2: 训练太慢

**解决方案**：
- 确保关闭其他占用GPU的应用（如浏览器、Photoshop等）
- 使用更小的模型：`python scripts/finetune_mlx_mac.py --model phi3`

### 问题 3: 想要更高质量的训练

如果你的Mac有足够的内存（32GB+），可以使用更积极的设置：

```bash
# 查看可用配置
python scripts/finetune_mlx_mac.py --list-profiles

# 手动指定配置
python scripts/finetune_mlx_mac.py --profile mac_m1_32gb
```

---

## 训练配置说明

### 内存配置对照表

| 内存大小 | 推荐配置 | batch_size | max_seq_length | num_layers |
|---------|---------|------------|----------------|------------|
| 8GB     | mac_m1_8gb | 1 | 512 | 4 |
| 16GB    | mac_m1_16gb | 1 | 1024 | 8 |
| 32GB    | mac_m1_32gb | 2 | 1536 | 12 |
| 64GB    | mac_m1_64gb | 4 | 2048 | 16 |
| 128GB+  | mac_m1_128gb | 8 | 4096 | 24 |

### 内存安全配置

如果你不确定该用什么配置，使用 `--memory-safe` 标志：

```bash
python scripts/finetune_mlx_mac.py --auto --memory-safe
```

这会根据你的硬件自动选择最保守的安全配置。

---

## 完整训练流程

### 步骤 1: 准备语料

```bash
# 将PDF/DOCX文件放入：
# data/corpus/raw/

# 重要文件放入（权重更高）：
# data/corpus/raw/important_examples/
```

### 步骤 2: 解析语料

```bash
make parse-corpus
```

### 步骤 3: 准备训练数据

```bash
make prepare-training
```

### 步骤 4: 分析数据（可选但推荐）

```bash
make analyze-data
```

这会显示：
- 数据总条数
- Token长度分布
- 超过阈值的序列数量
- 推荐的max_tokens设置

### 步骤 5: 预处理数据（如果有长序列）

```bash
# 自动检测并分割
make preprocess-data

# 或指定最大长度
make preprocess-data MAX_TOKENS=1024
```

### 步骤 6: 开始训练

```bash
# 内存安全模式（推荐）
make finetune-mlx-safe

# 或标准模式
make finetune-mlx
```

### 步骤 7: 创建Ollama模型

```bash
ollama create gswa-gilles -f models/gswa-mlx-*/Modelfile
```

### 步骤 8: 配置并运行

```bash
echo 'VLLM_MODEL_NAME=gswa-gilles' >> .env
make run
```

---

## 一键命令汇总

| 命令 | 说明 |
|------|------|
| `make train-safe` | 傻瓜式一键训练（推荐） |
| `make train-auto` | 自动训练（不询问确认） |
| `make train` | 交互式训练向导 |
| `make analyze-data` | 分析训练数据 |
| `make preprocess-data` | 预处理长序列 |
| `make finetune-mlx-safe` | 内存安全MLX训练 |
| `make finetune-all-safe` | 完整流程（内存安全） |

---

## 高级选项

### 手动指定参数

```bash
python scripts/finetune_mlx_mac.py \
    --model mistral \
    --batch-size 1 \
    --max-seq-length 1024 \
    --num-layers 8 \
    --iters 500 \
    --memory-safe
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 4 | 批大小，减小可节省内存 |
| `--max-seq-length` | 1024 | 最大序列长度，减小可节省内存 |
| `--num-layers` | 16 | LoRA层数，减小可节省内存 |
| `--iters` | 1000 | 训练迭代次数 |
| `--learning-rate` | 1e-5 | 学习率 |
| `--memory-safe` | False | 启用内存安全模式 |
| `--retry-on-oom` | True | OOM时自动重试 |

---

## 故障排除

### 检查依赖

```bash
make check-mlx
```

### 查看训练状态

```bash
make status
```

### 查看可用模型

```bash
make models
```

### 清理并重新开始

```bash
make clean
rm -rf data/training/mlx/
make train-safe
```

---

## 技术支持

如果问题仍未解决，请：

1. 运行 `make analyze-data` 并保存输出
2. 运行 `sysctl hw.memsize` 查看内存大小
3. 在 GitHub Issues 中提交问题，附上上述信息

---

## 英文版 / English Version

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the English version.
