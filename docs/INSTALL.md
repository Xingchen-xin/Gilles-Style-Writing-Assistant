# GSWA 安装指南 / Installation Guide

> 详细的环境配置和安装说明，支持无 sudo 权限的服务器环境

## 目录

- [系统要求](#系统要求)
- [一键安装（推荐）](#一键安装推荐)
- [手动安装](#手动安装)
- [CUDA/GPU 配置](#cudagpu-配置)
- [常见问题排查](#常见问题排查)
- [环境验证](#环境验证)

---

## 系统要求

### 最低要求

| 组件 | 要求 |
|------|------|
| Python | 3.10+ (必需) |
| 内存 | 16GB+ |
| 磁盘 | 20GB+ 可用空间 |

### 推荐配置

| 平台 | 推荐配置 |
|------|----------|
| **Mac** | Apple Silicon (M1/M2/M3), 16GB+ RAM |
| **Linux** | NVIDIA GPU (24GB+ VRAM), 32GB+ RAM |
| **Windows** | NVIDIA GPU (16GB+ VRAM), 32GB+ RAM |

### 重要说明

- **无需 sudo 权限**：本项目所有操作都可以在用户目录下完成
- **推荐使用 micromamba/conda**：在 Linux 服务器上，micromamba 比 pyenv 更可靠

---

## 一键安装（推荐）

### 基础安装

```bash
# 克隆仓库
git clone <repository-url>
cd Gilles-Style-Writing-Assistant

# 一键安装（自动检测环境）
make setup

# 或全自动模式（无需确认）
make setup-auto
```

### CUDA/GPU 安装（Linux 服务器）

```bash
# 带 CUDA 支持的安装（推荐用于训练）
make setup-cuda

# 全自动 CUDA 安装
make setup-cuda-auto
```

### 安装脚本做了什么？

1. **检测 Python 3.10+**
   - 按顺序尝试：python3.13 → python3.12 → python3.11 → python3.10 → python3
   - 验证 `ctypes` 模块可用（PyTorch 必需）

2. **如果 Python 不满足要求**
   - Linux：自动安装 micromamba 并创建 conda 环境
   - Mac：提供 pyenv 或 micromamba 选项

3. **检测 NVIDIA GPU**
   - 自动识别 GPU 型号和 VRAM
   - 提示安装 CUDA 版本的 PyTorch

4. **创建虚拟环境**
   - 优先使用 conda 环境（更可靠）
   - 备选使用 Python venv

5. **安装依赖**
   - 核心依赖：FastAPI, Pydantic, httpx
   - 训练依赖：PyTorch, transformers, sentence-transformers
   - 工具依赖：PyMuPDF (PDF 解析)

---

## 手动安装

### 方式一：使用 micromamba（推荐，Linux 服务器）

```bash
# 1. 安装 micromamba（无需 sudo）
curl -L micro.mamba.pm/install.sh | bash

# 2. 重新加载 shell 配置
source ~/.bashrc  # 或 source ~/.zshrc

# 3. 创建环境
micromamba create -n gswa python=3.11 -y

# 4. 激活环境
micromamba activate gswa

# 5. 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch torchvision

# 6. 安装项目依赖
pip install -e ".[dev,similarity]" pymupdf
```

### 方式二：使用系统 Python + venv

```bash
# 确保有 Python 3.10+
python3.11 --version

# 创建虚拟环境
python3.11 -m venv venv

# 激活
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -e ".[dev,similarity]"
```

### 方式三：使用 pyenv（Mac 推荐）

```bash
# 安装 pyenv
curl https://pyenv.run | bash

# 添加到 shell 配置
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# 安装 Python
pyenv install 3.11.9
pyenv local 3.11.9

# 创建 venv
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,similarity]"
```

---

## CUDA/GPU 配置

### 检测 GPU

```bash
# 查看 GPU 信息
nvidia-smi

# 示例输出：
# NVIDIA RTX 5000 Ada Generation (32GB VRAM)
# CUDA Version: 13.1
```

### PyTorch CUDA 版本选择

| 系统 CUDA 版本 | PyTorch 安装命令 |
|----------------|------------------|
| CUDA 13.x | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |

### 验证 CUDA

```bash
# 使用 conda 环境
micromamba run -n gswa python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

---

## 常见问题排查

### 问题 1: `ModuleNotFoundError: No module named '_ctypes'`

**原因**：pyenv 编译的 Python 缺少 `libffi` 支持

**症状**：
```
Traceback (most recent call last):
  ...
ModuleNotFoundError: No module named '_ctypes'
```

**解决方案**：使用 micromamba 代替 pyenv

```bash
# 1. 安装 micromamba
curl -L micro.mamba.pm/install.sh | bash
source ~/.bashrc

# 2. 创建新环境
micromamba create -n gswa python=3.11 -y

# 3. 安装 PyTorch
micromamba run -n gswa pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 安装项目
micromamba run -n gswa pip install -e ".[dev,similarity]" pymupdf
```

**为什么 micromamba 有效**：
- micromamba 自带预编译的 Python 和所有依赖库（包括 libffi）
- 不需要系统级的 `libffi-devel` 包
- 完全独立，不依赖系统环境

### 问题 2: CUDA 未检测到

**症状**：
```
CUDA Available: No
Training Backend: CPU
```

**排查步骤**：

```bash
# 1. 检查 nvidia-smi
nvidia-smi
# 应该能看到 GPU 信息

# 2. 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
# 应该输出 12.4 或类似版本

# 3. 如果输出 None，重新安装 PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 问题 3: pip 依赖解析超时

**症状**：
```
ResolutionTooDeep: 2000000
```

**解决方案**：
```bash
# 升级 pip
pip install --upgrade pip

# 如果仍然失败，使用 conda 安装
micromamba install pytorch torchvision -c pytorch -c nvidia
```

### 问题 4: 内存不足 (OOM)

**症状**：训练时出现 CUDA out of memory

**解决方案**：
```bash
# 使用内存安全模式训练
make train-safe

# 或手动调整参数
python scripts/smart_finetune.py --batch-size 1 --max-seq-length 512
```

---

## 环境验证

### 快速检查

```bash
# 使用 conda 环境
micromamba run -n gswa make test

# 或激活后运行
micromamba activate gswa
make test
```

### 完整验证

```bash
# 1. 检查 Python 版本
python --version  # 应该是 3.10+

# 2. 检查关键包
pip show gswa torch fastapi pymupdf

# 3. 检查 CUDA（如果有 GPU）
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 4. 运行测试
make test

# 5. 检查训练环境
make train-info
```

### 预期输出

```
======================================================================
GSWA Smart Fine-tuning System - System Information
======================================================================

Operating System:         Linux
CPU:                      AMD EPYC 9534 64-Core Processor
System RAM:               629.0 GB

----------------------------------------------------------------------
GPU Information
----------------------------------------------------------------------
GPU Type:                 NVIDIA
GPU Name:                 NVIDIA RTX 5000 Ada Generation
GPU VRAM:                 32.0 GB
CUDA Available:           Yes

----------------------------------------------------------------------
Recommended Configuration
----------------------------------------------------------------------
Training Backend:         CUDA
Model Tier:               tier_2
Recommended Model:        Qwen2.5 14B
```

---

## 使用环境

### 使用 conda 环境

```bash
# 激活环境
micromamba activate gswa

# 运行命令
make run
make test
make finetune-smart

# 退出环境
micromamba deactivate
```

### 不激活直接运行

```bash
# 使用 micromamba run
micromamba run -n gswa make test
micromamba run -n gswa make run
micromamba run -n gswa make finetune-smart
```

### 使用 venv 环境

```bash
# 激活
source venv/bin/activate

# 运行
make run

# 退出
deactivate
```

---

## 下一步

安装完成后：

1. **添加语料**：将 PDF 文件放入 `data/corpus/raw/`
2. **训练模型**：`make finetune-smart`
3. **启动服务**：`make run`
4. **访问界面**：http://localhost:8080

详细训练指南请参考：[TRAINING_GUIDE.md](TRAINING_GUIDE.md)
