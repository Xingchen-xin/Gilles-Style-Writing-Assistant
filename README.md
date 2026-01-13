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
# 1. 添加文档到 data/corpus/raw/
cp papers/*.pdf data/corpus/raw/

# 2. 解析文档
python scripts/parse_corpus.py

# 3. 构建索引
python scripts/build_index.py
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
