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

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e ".[dev,similarity]"
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

Linux 用户使用 **vLLM** 作为 LLM 后端，需要 NVIDIA GPU。

#### 第一步：安装 vLLM

```bash
# 安装 vLLM（需要 CUDA）
pip install vllm
```

#### 第二步：克隆并安装 GSWA

```bash
git clone <repository-url>
cd Gilles-Style-Writing-Assistant

python3 -m venv venv
source venv/bin/activate

pip install -e ".[dev,similarity]"
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
# === 安装 ===
make install          # 核心依赖
make dev              # 开发依赖

# === Mac 设置 ===
make setup-mac        # 一键设置 Ollama
make setup-ollama     # 仅配置 Ollama

# === Linux 设置 ===
make start-vllm       # 启动 vLLM

# === 运行 ===
make run              # 开发模式
make run-prod         # 生产模式

# === 测试 ===
make test             # 单元测试
make lint             # 代码检查
make smoke-test       # 端到端测试

# === 语料库 ===
make parse-corpus     # 解析文档
make build-index      # 构建索引

# === 训练数据 ===
make export-dpo       # 导出 DPO 数据
```

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

GSWA 内置了多层 AI 检测规避机制，帮助生成更像人类写作的输出。

### 工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI 检测规避流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐  │
│   │ 1. 生成   │───▶│ 2. 检测   │───▶│ 3. 修正   │───▶│ 4. 输出 │  │
│   │ (带规则)  │    │ AI 痕迹   │    │ (自动)    │    │ (安全)  │  │
│   └──────────┘    └──────────┘    └──────────┘    └─────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 检测的 AI 痕迹类型

| 类型 | 典型例子 | 修正方式 |
|------|----------|----------|
| **过度使用的过渡词** | Furthermore, Moreover, Additionally | 替换为 Also, And, 或删除 |
| **AI 惯用短语** | It is worth noting that... | 直接陈述内容 |
| **过度正式词汇** | utilize, leverage, facilitate | 使用简单词: use, help |
| **句子长度均匀** | 所有句子 15-25 词 | 混合短句和长句 |
| **完美枚举结构** | First... Second... Third... | 变化表达方式 |
| **过度 hedge** | may potentially possibly | 限制每段最多 2 个 |

### 使用命令

```bash
# 分析文本的 AI 痕迹
make ai-check

# 分析作者风格
make analyze-style

# 查看当前风格指纹
make style-show
```

### AI 分数解读

| 分数范围 | 含义 | 建议 |
|----------|------|------|
| 0.0 - 0.2 | 很像人类写作 | 无需修改 |
| 0.2 - 0.4 | 正常范围 | 可接受 |
| 0.4 - 0.6 | 有明显 AI 特征 | 建议修正 |
| 0.6 - 1.0 | 强烈 AI 特征 | 必须修正 |

### 最佳实践

1. **选择合适的基础模型**
   - 推荐: Qwen 2.5-7B (英文学术写作能力强)
   - 避免: GPT-4/Claude API (输出有明显指纹)

2. **收集足够的训练语料**
   - 至少 10+ 篇目标作者的论文
   - 将代表性文章放入 `important_examples/` (2.5x 权重)

3. **利用 DPO 训练**
   - 将"听起来像 AI"的输出标记为 `ai_like`
   - 系统会自动检测高 AI 分数的输出并加入拒绝集

4. **后处理检查**
   ```bash
   make ai-check  # 粘贴文本检查
   ```

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
