# GSWA - Gilles-Style Writing Assistant

> 完全本地离线运行的科学论文段落重写工具

## 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
  - [Mac 用户 (Apple Silicon M1/M2/M3)](#mac-用户-apple-silicon-m1m2m3)
  - [Linux 用户 (NVIDIA GPU)](#linux-用户-nvidia-gpu)
  - [Windows 用户](#windows-用户)
- [使用 Web 界面](#使用-web-界面)
- [系统架构](#系统架构)
- [API 文档](#api-文档)
- [开发者指南](#开发者指南)
- [常见问题](#常见问题)

---

## 项目概述

GSWA 是一个本地部署的 AI 写作助手，专门用于将科学论文段落重写为符合特定写作风格（Gilles 风格）的版本。

### 核心特性

- **完全离线** - 无外部 API 调用，数据不出本地
- **多版本输出** - 同一输入生成 3-5 个不同组织策略的候选版本
- **语义保持** - 不改变数值、实验条件、结论强度
- **去重保护** - N-gram + 向量相似度检测，自动回退重写
- **多平台支持** - Mac (Ollama)、Linux (vLLM)、Windows (LM Studio)
- **AI 检测规避** - 自动检测和修正 AI 写作痕迹，降低被 AI 检测器识别的风险
- **风格指纹分析** - 自动提取目标作者的写作特征，生成更像人类的输出

### 支持的 LLM 后端

| 后端 | 平台 | 硬件要求 | 推荐指数 |
|------|------|----------|----------|
| **Ollama** | Mac (Apple Silicon) | 16GB+ RAM | ⭐⭐⭐⭐⭐ |
| **vLLM** | Linux | NVIDIA GPU 24GB+ | ⭐⭐⭐⭐⭐ |
| **LM Studio** | Windows/Mac | 16GB+ RAM | ⭐⭐⭐⭐ |
| **Ollama** | Linux | 16GB+ RAM | ⭐⭐⭐⭐ |

---

## 快速开始

### 环境要求

- **Python 3.10+** （必需）
- 16GB+ 内存
- Mac: Apple Silicon (M1/M2/M3) 或 Intel
- Linux: NVIDIA GPU（推荐 24GB+ VRAM）

### 傻瓜式一键设置（推荐）

不管是什么平台，都可以使用一键设置脚本：

```bash
# 克隆仓库
git clone <repository-url>
cd Gilles-Style-Writing-Assistant

# 一键设置（自动检测/安装 Python 3.10+，创建环境，安装依赖）
make setup

# 或全自动模式（无需确认）
make setup-auto

# 有 NVIDIA GPU？使用 CUDA 版本（推荐用于训练）
make setup-cuda

# CUDA 全自动模式
make setup-cuda-auto
```

脚本会自动：
1. 检测系统中的 Python 3.10+（尝试 python3.13/3.12/3.11/3.10）
2. 如果未找到，**自动使用 micromamba/conda 安装**（无需 sudo 权限）
3. 检测 NVIDIA GPU 并提示安装 CUDA 支持
4. 创建虚拟环境或 conda 环境
5. 安装所有依赖（包括 PyTorch）

> **服务器用户注意**：脚本支持无 sudo 权限的环境。在 Linux 服务器上推荐使用 micromamba（自动安装），它会自带完整的 Python 环境，不需要系统级的 libffi-devel 等依赖。

#### 使用 conda 环境

如果使用 micromamba/conda 环境，激活方式略有不同：

```bash
# 激活 conda 环境
micromamba activate gswa

# 或直接运行命令
micromamba run -n gswa make test
micromamba run -n gswa make run
```

---

### Mac 用户 (Apple Silicon M1/M2/M3)

Mac 用户使用 **Ollama** 作为 LLM 后端，Ollama 对 Apple Silicon 有原生优化。

#### 第一步：安装 Ollama

```bash
# 方式一：使用 Homebrew（推荐）
brew install ollama

# 方式二：从官网下载
# 访问 https://ollama.ai/download 下载 Mac 版本
```

#### 第二步：下载模型

```bash
# 启动 Ollama 服务
ollama serve

# 在新终端窗口下载模型（推荐 mistral，约 4GB）
ollama pull mistral

# 或下载其他模型
# ollama pull llama2      # 通用模型
# ollama pull mixtral     # 更大更好，需要更多内存
```

#### 第三步：克隆并安装 GSWA

```bash
# 克隆仓库
git clone <repository-url>
cd Gilles-Style-Writing-Assistant

# 一键设置（推荐，自动安装 micromamba 环境）
make setup-auto

# 激活环境
micromamba activate gswa
```

#### 第四步：配置 GSWA 使用 Ollama

```bash
# 创建配置文件
cp .env.example .env

# 编辑 .env，设置以下内容：
```

编辑 `.env` 文件，确保包含：
```bash
LLM_BACKEND=ollama
VLLM_MODEL_NAME=mistral
```

#### 第五步：启动 GSWA

```bash
# 确保 Ollama 正在运行（在另一个终端）
ollama serve

# 启动 GSWA 服务器
make run
```

#### 第六步：打开浏览器

访问 `http://localhost:8080`，开始使用！

#### 一键设置（可选）

如果上述步骤太多，可以使用自动设置脚本：

```bash
# 自动安装和配置 Ollama
make setup-mac
```

---

### Linux 用户 (NVIDIA GPU)

Linux 服务器推荐使用 **micromamba** 管理环境（无需 sudo 权限）。

#### 第一步：一键设置（推荐）

```bash
git clone <repository-url>
cd Gilles-Style-Writing-Assistant

# 一键设置（自动安装 micromamba + Python + CUDA 依赖）
make setup-cuda-auto
```

脚本会自动：
1. 安装 micromamba（如果没有）
2. 创建 `gswa` conda 环境
3. 安装 PyTorch with CUDA
4. 安装所有训练依赖

#### 第二步：激活环境

```bash
# 激活 micromamba 环境（每次新终端都需要）
micromamba activate gswa

# 注意：使用 micromamba 后不需要 venv！
# 如果看到 (venv) 提示，先运行 deactivate
```

#### 第三步：配置（默认即可）

```bash
cp .env.example .env
# 默认配置已经是 vLLM，无需修改
```

#### 第四步：启动 vLLM 服务器

```bash
# 启动 vLLM（在单独的终端）
make start-vllm

# 或手动启动
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --port 8000
```

#### 第五步：启动 GSWA

```bash
make run
```

访问 `http://localhost:8080`

---

### Windows 用户

Windows 用户推荐使用 **LM Studio** 或 **Ollama for Windows**。

#### 选项 A：使用 LM Studio

1. 下载安装 [LM Studio](https://lmstudio.ai/)
2. 在 LM Studio 中下载 Mistral 7B 模型
3. 启动本地服务器（LM Studio → Local Server → Start Server）
4. 配置 GSWA：
   ```bash
   LLM_BACKEND=lm-studio
   VLLM_MODEL_NAME=local-model
   ```

#### 选项 B：使用 Ollama for Windows

1. 下载安装 [Ollama for Windows](https://ollama.ai/download/windows)
2. 打开 PowerShell，运行：
   ```powershell
   ollama pull mistral
   ollama serve
   ```
3. 配置 GSWA：
   ```bash
   LLM_BACKEND=ollama
   VLLM_MODEL_NAME=mistral
   ```

---

## 使用 Web 界面

### 基本操作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    GSWA 使用流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐  │
│   │ 1. 粘贴   │───▶│ 2. 选择   │───▶│ 3. 生成   │───▶│ 4. 复制 │  │
│   │   段落    │    │   部分    │    │   变体    │    │  结果   │  │
│   └──────────┘    └──────────┘    └──────────┘    └─────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 详细步骤

1. **打开浏览器**访问 `http://localhost:8080`

2. **粘贴段落**到文本框中，例如：
   ```
   We observed a 2.5-fold increase in enzyme activity (p < 0.01)
   when cells were treated with compound X at 10 μM for 24 hours.
   This suggests that compound X may enhance catalytic efficiency
   through allosteric modulation.
   ```

3. **选择设置**：
   - **Section（论文部分）**: Results / Methods / Discussion 等
   - **Variants（变体数量）**: 3 / 4 / 5

4. **点击 "Generate Variants"** 按钮

5. **查看结果**：
   - 每个变体显示使用的策略（A/B/C/D）
   - 显示相似度分数
   - 回退重写会显示 "Fallback" 标签

6. **使用结果**：
   - 点击 **"Copy"** 复制满意的变体
   - 为变体评分（Best/Good/Bad）
   - 可编辑变体后提交反馈

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + Enter` | 生成变体 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI (HTML/JS)                         │
│                    paste → generate → copy                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP (localhost:8080)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Orchestrator                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ POST /v1/rewrite/variants - 生成重写变体                  │    │
│  │ POST /v1/feedback        - 提交反馈                       │    │
│  │ GET  /v1/health          - 健康检查                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                  │
│  ┌───────────────┬────────────┴─────────────┬───────────────┐   │
│  │  Prompt       │    Similarity Service    │   Fallback    │   │
│  │  Constructor  │    (n-gram + embed)      │   Logic       │   │
│  └───────────────┴──────────────────────────┴───────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ OpenAI-compatible API
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Local LLM Server (Ollama / vLLM / LM Studio)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## API 文档

### POST /v1/rewrite/variants

生成多个重写变体。

**请求：**
```json
{
  "text": "We observed a 2.5-fold increase in enzyme activity...",
  "section": "Results",
  "n_variants": 3
}
```

**响应：**
```json
{
  "variants": [
    {
      "text": "A 2.5-fold enhancement in enzymatic activity was detected...",
      "strategy": "A",
      "scores": {
        "ngram_max_match": 3,
        "embed_top1": 0.72
      },
      "fallback": false
    }
  ],
  "model_version": "mistral@v1",
  "processing_time_ms": 2450
}
```

### GET /v1/health

检查系统健康状态。

**响应：**
```json
{
  "status": "healthy",
  "llm_server": "connected",
  "model_loaded": "mistral",
  "corpus_paragraphs": 0,
  "index_loaded": false
}
```

---

## 重写策略说明

| 策略 | 名称 | 描述 | 适用场景 |
|------|------|------|----------|
| A | Conclusion-first | 先陈述结论，再提供细节 | Results, Abstract |
| B | Background-first | 先提供背景，再引入观点 | Introduction |
| C | Methods-first | 先描述方法，再报告发现 | Methods |
| D | Cautious-first | 先提出局限，保守陈述 | Discussion |

---

## 开发者指南

### 项目结构

```
gswa/
├── src/gswa/              # 源代码
│   ├── api/               # API 路由和模型
│   ├── services/          # 业务逻辑
│   ├── utils/             # 工具函数
│   ├── config.py          # 配置管理
│   └── main.py            # FastAPI 入口
├── web/                   # Web UI
├── scripts/               # 脚本工具
├── tests/                 # 测试文件
└── data/                  # 数据目录
```

### 常用命令

```bash
# === 一键操作（傻瓜式） ===
make setup-cuda-auto  # Linux GPU 全自动安装
make setup-auto       # Mac 全自动安装
make finetune-smart   # 一键训练（自动检测平台和硬件）
make run              # 启动服务器

# === 其他 ===
make train-info       # 查看硬件信息
make test             # 运行测试
make parse-corpus     # 解析语料
```

> **conda 环境用户**：激活后运行 `micromamba activate gswa`，或直接 `micromamba run -n gswa make test`

---

## 微调模型（减少 AI 检测）

默认模型可能被 AI 检测器识别为纯 AI 生成。通过微调，模型可以学习 Gilles 的写作风格，使输出更自然、更像人类写作。

### 傻瓜式操作（只需 3 步）

```
1. 放文章到文件夹
   data/corpus/raw/                    <- 普通文章
   data/corpus/raw/important_examples/ <- 重要文章（自动 2.5x 权重）

2. 运行一条命令
   make finetune-all

3. 部署并重启
   ollama create gswa-gilles -f models/gswa-mlx-*/Modelfile
   echo "VLLM_MODEL_NAME=gswa-gilles" >> .env
   make run
```

### 文件夹结构

```
data/corpus/raw/                      <- 普通 Gilles 文章 (权重 1.0x)
├── paper1.pdf
├── paper2.docx
│
└── important_examples/               <- 重要文章 (自动权重 2.5x)
    ├── best_review.pdf
    └── classic_paper.pdf
```

**支持格式：** `.pdf`, `.docx`, `.txt`

### 微调方案对比

| 方案 | 平台 | 硬件 | 时间 | 命令 |
|------|------|------|------|------|
| **MLX** | Mac | M1/M2/M3 16GB+ | 1-2h | `make finetune-all` |
| **QLoRA** | Linux | GPU 8GB+ | 3-6h | `make finetune-lora` |
| **LoRA** | Linux | GPU 16GB+ | 2-4h | 见文档 |

### 常用命令

```bash
make finetune-all      # 一键完成（解析 + 训练 + 微调）
make parse-corpus      # 仅解析文章
make prepare-training  # 仅准备训练数据
make list-docs         # 列出所有文章 ID
make training-stats    # 查看训练数据统计
```

详细文档：[docs/FINETUNING.md](docs/FINETUNING.md)

---

## AI 检测规避 (Anti-AI Detection)

GSWA 内置**基于学术研究**的科学化 AI 检测规避系统。

### 检测原理 (基于学术研究)

系统使用四个核心指标，权重基于研究有效性：

| 指标 | 权重 | 原理 | 人类特征 | AI 特征 |
|------|------|------|----------|---------|
| **Burstiness** | 30% | 句子长度变化 (CV) | > 0.4 | < 0.3 |
| **Perplexity** | 25% | 文本可预测性 | 20-80 | 5-15 |
| **Vocabulary** | 20% | 词汇多样性 (TTR) | 0.5-0.8 | 0.3-0.5 |
| **Style Match** | 15% | 与作者风格匹配 | 高 | 低 |
| **Patterns** | 10% | AI 典型短语 | 无 | 有 |

> 参考文献: GPTZero, DetectGPT, StyloAI, DIPPER 等研究

### 使用命令

```bash
# 科学化 AI 分析 (显示各维度分数)
make ai-check

# 一键人性化文本 (自动降低 AI 分数)
make humanize

# 分析作者风格
make analyze-style
```

### 输出示例

```
=== Scientific AI Detection ===
OVERALL: AI Score = 0.42 | Risk: MODERATE
Confidence: 85%

METRICS (lower score = more human-like):
  Perplexity:      28.5  (score: 0.35)
  Burstiness:      0.312 (score: 0.58)  ← 需要改进!
  Vocab Diversity: 0.523 (score: 0.28)
  Style Match:     0.650 (score: 0.35)

Sentence lengths: [18, 22, 19, 21, 20]  ← 太均匀!

SUGGESTIONS:
  - CRITICAL: Sentence lengths too uniform. Mix short and long.
  - Remove/replace: 'Furthermore'
```

### 最关键: Burstiness (句子变化)

研究表明 **句子长度均匀性是最强的 AI 检测信号**：

```
❌ AI 输出 (Burstiness = 0.18):
"The results showed significant improvements. The modified
approach demonstrated enhanced performance. The experimental
conditions were carefully controlled."
→ 长度: [5, 5, 5] 太均匀

✅ 人类化 (Burstiness = 0.52):
"Results were striking. We observed a 47% increase in binding
affinity when the modified peptide was introduced, suggesting
the conformational change plays a key role. This matters."
→ 长度: [3, 24, 2] 变化大
```

### 最佳实践

1. **优先关注 Burstiness** - 每段混合短句 (5-10词) 和长句 (25-35词)
2. **使用微调本地模型** - 避免直接使用 ChatGPT/GPT-4 API
3. **运行风格分析** - `make analyze-style` 生成作者指纹
4. **一键人性化** - `make humanize` 自动降低 AI 分数

---

## 风格指纹分析

系统会自动分析目标作者的写作特征，生成风格指纹。

### 分析的特征

- **句子统计**: 平均长度、分布、变化程度
- **词汇偏好**: 常用动词、形容词、过渡词
- **结构特征**: 被动语态比例、hedge 使用频率
- **短语模式**: 常见开头、过渡方式

### 使用方式

```bash
# 一键训练时自动分析
make train

# 单独运行分析
make analyze-style

# 查看详细指纹
make style-show
```

### 输出示例

```
=== Style Fingerprint ===
Author:          Gilles
Corpus size:     156 paragraphs
Total words:     45,230

Sentence Statistics:
  Average length:  22.5 words (std: 8.3)
  Distribution:    15% short, 60% medium, 25% long

Structure:
  Passive voice:   35.2% of sentences
  Hedge frequency: 1.8 per 100 words

Vocabulary:
  Top verbs:       show, demonstrate, indicate, suggest
  Transitions:     however, also, while, although
```

---

## 常见问题

### Q: Mac 上 Ollama 无法启动？

A: 检查以下几点：
1. 确保已安装 Ollama：`brew install ollama`
2. 启动 Ollama 服务：`ollama serve`
3. 检查端口：`curl http://localhost:11434/api/tags`

### Q: 显示 "System Degraded" 怎么办？

A: 这表示 LLM 服务器未连接：

**Mac 用户：**
```bash
# 确保 Ollama 正在运行
ollama serve

# 检查连接
curl http://localhost:11434/api/tags
```

**Linux 用户：**
```bash
# 确保 vLLM 正在运行
make start-vllm

# 检查连接
curl http://localhost:8000/v1/models
```

### Q: 如何切换模型？

A:
```bash
# Mac (Ollama)
ollama pull llama2
# 然后编辑 .env: VLLM_MODEL_NAME=llama2

# Linux (vLLM)
# 修改 scripts/start_vllm.sh 中的 MODEL 变量
```

### Q: 内存不足怎么办？

A: 尝试使用更小的模型：
- `mistral` (4GB) - 推荐
- `phi` (1.6GB) - 更小
- `tinyllama` (637MB) - 最小

### Q: 如何添加语料库？

A:
```bash
# 1. 添加普通文档
cp papers/*.pdf data/corpus/raw/

# 2. 添加重要/代表性文档（会自动获得 2.5x 权重）
cp important_papers/*.pdf data/corpus/raw/important_examples/

# 3. 解析并构建索引
make parse-corpus
make build-index

# 或者直接一键微调（包含解析）
make finetune-all
```

### Q: 如何完全离线运行？

A: GSWA 设计为完全离线：
1. 模型下载后存储在本地
2. 所有推理在本地进行
3. 无任何外部 API 调用
4. 无遥测数据发送

### Q: `ModuleNotFoundError: No module named '_ctypes'` 怎么办？

A: 这是因为 pyenv 编译的 Python 缺少 `libffi` 支持。服务器没有 sudo 权限时无法安装 `libffi-devel`。

**解决方案**：使用 micromamba 代替 pyenv
```bash
# micromamba 自带完整 Python，不需要系统库
curl -L micro.mamba.pm/install.sh | bash
source ~/.bashrc

# 创建新环境
micromamba create -n gswa python=3.11 -y
micromamba activate gswa

# 重新安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,similarity]" pymupdf
```

详细说明请参考：[docs/INSTALL.md](docs/INSTALL.md)

### Q: CUDA 未检测到 / GPU 训练显示 CPU？

A: 可能的原因和解决方案：

**检查 CUDA 是否可用**：
```bash
# 1. 检查 nvidia-smi
nvidia-smi

# 2. 检查 PyTorch CUDA 版本
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Version: {torch.version.cuda}')"
```

**如果输出 CUDA: False**：
```bash
# 重新安装带 CUDA 的 PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**如果看到 `_ctypes` 错误**：
参考上一个问题的解决方案。

### Q: Linux 服务器没有 sudo 权限怎么办？

A: 本项目完全支持无 sudo 环境：

```bash
# 一键安装（自动使用 micromamba）
make setup-cuda-auto

# micromamba 会安装在用户目录下
# 不需要任何系统级权限
```

---

## 安全约束

| 约束 | 说明 |
|------|------|
| 无外部调用 | `ALLOW_EXTERNAL_API=false` 硬编码 |
| 仅本地 URL | 只允许 localhost |
| 无遥测 | 不发送任何数据 |
| 日志脱敏 | 只记录 hash |

---

## 许可证

本项目仅供内部使用。

---

## 技术支持

- `docs/SPEC.md` - 完整技术规格
- `docs/TASKS.md` - 开发任务列表
- `docs/API.md` - API 详细文档
