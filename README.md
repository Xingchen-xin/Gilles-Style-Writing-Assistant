# GSWA - Gilles-Style Writing Assistant

> 完全本地离线运行的科学论文段落重写工具

## 目录

- [项目概述](#项目概述)
- [快速开始（傻瓜式操作指南）](#快速开始傻瓜式操作指南)
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
- **反馈收集** - 支持用户反馈以进行 DPO 训练

### 安全约束

| 约束 | 说明 |
|------|------|
| 无外部调用 | `ALLOW_EXTERNAL_API=false` 硬编码默认值 |
| 仅本地 URL | 只允许 localhost/127.0.0.1 |
| 无遥测 | 不发送任何使用数据 |
| 日志脱敏 | 不记录原文，只记录 hash |

---

## 快速开始（傻瓜式操作指南）

### 前置要求

- Python 3.10 或更高版本
- GPU 服务器（推荐 24GB+ 显存用于 7B 模型）
- 或者使用 Mock 模式进行测试（无需 GPU）

### 第一步：克隆并安装

```bash
# 1. 克隆仓库
git clone <repository-url>
cd Gilles-Style-Writing-Assistant

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -e ".[dev,similarity]"
```

**验证安装成功：**
```bash
python -c "import gswa; print(f'GSWA v{gswa.__version__} installed successfully!')"
```

### 第二步：准备语料库（可选但推荐）

语料库用于检测生成文本与已有文献的相似度，防止抄袭。

```bash
# 1. 将 Gilles 的论文（PDF/DOCX/TXT）放入 data/corpus/raw/ 目录
cp /path/to/gilles/papers/*.pdf data/corpus/raw/

# 2. 解析文档为段落
python scripts/parse_corpus.py

# 3. 构建相似度索引
python scripts/build_index.py
```

**没有语料库？** 系统仍可运行，但不会进行相似度检测。可以跳过此步骤。

### 第三步：启动 vLLM 服务器（需要 GPU）

GSWA 需要一个本地 vLLM 服务器来运行 LLM 推理。

```bash
# 方式一：使用默认模型启动
./scripts/start_vllm.sh

# 方式二：指定模型
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2 ./scripts/start_vllm.sh

# 方式三：使用 Makefile
make start-vllm
```

**没有 GPU？**
- 可以跳过此步骤
- GSWA 会显示 "System Degraded" 但仍可进行安全性测试
- 生产环境使用需要 GPU

### 第四步：启动 GSWA 服务器

```bash
# 开发模式（支持热重载，推荐）
make run

# 或直接使用 uvicorn
uvicorn gswa.main:app --reload --port 8080 --host 0.0.0.0
```

**成功启动后会显示：**
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### 第五步：使用 Web 界面

1. **打开浏览器**访问 `http://localhost:8080`

2. **粘贴段落**到文本框中，例如：
   ```
   We observed a 2.5-fold increase in enzyme activity (p < 0.01)
   when cells were treated with compound X at 10 μM for 24 hours.
   This suggests that compound X may enhance catalytic efficiency
   through allosteric modulation.
   ```

3. **选择设置**：
   - Section（论文部分）: Results / Methods / Discussion 等
   - Variants（变体数量）: 3 / 4 / 5

4. **点击 "Generate Variants"** 按钮

5. **查看结果**：
   - 每个变体会显示使用的策略（A/B/C/D）
   - 显示相似度分数（N-gram 匹配、向量相似度）
   - 如果触发了回退重写，会显示 "Fallback" 标签

6. **使用结果**：
   - 点击 "Copy" 复制满意的变体
   - 可以为每个变体评分（Best/Good/Bad）
   - 可以编辑变体后提交反馈

### 第六步：验证系统正常工作

```bash
# 运行单元测试（86 个测试）
make test

# 运行端到端测试（需要服务器运行）
python scripts/smoke_test.py

# 仅测试安全性（无需 vLLM）
python scripts/smoke_test.py --skip-llm
```

---

## 完整操作流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    GSWA 使用流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 1. 安装   │───▶│ 2. 语料库 │───▶│ 3. vLLM  │───▶│ 4. GSWA  │  │
│  │ 依赖     │    │ (可选)   │    │ (需GPU)  │    │ 服务器    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                       │         │
│                                                       ▼         │
│                                              ┌──────────────┐   │
│                                              │ 5. Web UI    │   │
│                                              │ 粘贴 → 生成   │   │
│                                              │ → 评分 → 复制 │   │
│                                              └──────────────┘   │
│                                                       │         │
│                                                       ▼         │
│                                              ┌──────────────┐   │
│                                              │ 6. 导出训练   │   │
│                                              │ 数据 (可选)   │   │
│                                              └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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
│  │ POST /v1/reply           - 简单聊天                       │    │
│  │ POST /v1/feedback        - 提交反馈                       │    │
│  │ GET  /v1/health          - 健康检查                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                  │
│  ┌───────────────┬────────────┴─────────────┬───────────────┐   │
│  │  Prompt       │    Similarity Service    │   Fallback    │   │
│  │  Constructor  │    (n-gram + embed)      │   Logic       │   │
│  └───────────────┴──────────────────────────┴───────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ OpenAI-compatible API (localhost:8000)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 vLLM Inference Server                            │
│            (local model, OpenAI-compatible)                      │
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
  "n_variants": 3,
  "constraints": {
    "preserve_numbers": true,
    "no_new_facts": true
  }
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
        "ngram_overlap": 0.02,
        "embed_top1": 0.72
      },
      "fallback": false,
      "fallback_reason": null
    }
  ],
  "model_version": "mistral-7b-instruct@v1",
  "processing_time_ms": 2450
}
```

### POST /v1/feedback

提交变体反馈用于 DPO 训练。

**请求：**
```json
{
  "session_id": "session_12345",
  "input_text": "Original paragraph...",
  "section": "Results",
  "variants": [
    {"variant_index": 0, "feedback_type": "best"},
    {"variant_index": 1, "feedback_type": "good"},
    {"variant_index": 2, "feedback_type": "bad"}
  ]
}
```

### GET /v1/health

检查系统健康状态。

**响应：**
```json
{
  "status": "healthy",
  "llm_server": "connected",
  "model_loaded": "mistral-7b-instruct",
  "corpus_paragraphs": 1234,
  "index_loaded": true
}
```

---

## 重写策略说明

GSWA 使用四种不同的组织策略来生成多样化的重写变体：

| 策略 | 名称 | 描述 | 适用场景 |
|------|------|------|----------|
| A | Conclusion-first | 先陈述主要结论，再提供支持细节 | Results, Abstract |
| B | Background-first | 先提供背景/动机，再引入主要观点 | Introduction, Discussion |
| C | Methods-first | 先描述方法/设置，再报告发现 | Methods, Results |
| D | Cautious-first | 先提出局限性，保守陈述观点 | Discussion, Conclusion |

---

## 相似度阈值

| 指标 | 阈值 | 触发动作 |
|------|------|----------|
| N-gram 最长匹配 | ≥ 12 tokens | 回退重写 |
| 向量相似度 top-1 | ≥ 0.88 | 回退重写 |

---

## 开发者指南

### 项目结构

```
gswa/
├── src/gswa/              # 源代码
│   ├── api/               # API 路由和模型
│   │   ├── routes.py      # FastAPI 路由
│   │   └── schemas.py     # Pydantic 模型
│   ├── services/          # 业务逻辑
│   │   ├── llm_client.py  # vLLM 客户端
│   │   ├── similarity.py  # 相似度服务
│   │   ├── prompt.py      # Prompt 模板
│   │   ├── rewriter.py    # 重写编排器
│   │   └── feedback.py    # 反馈收集
│   ├── utils/             # 工具函数
│   │   ├── ngram.py       # N-gram 检测
│   │   ├── embedding.py   # 向量相似度
│   │   └── logging.py     # 审计日志
│   ├── config.py          # 配置管理
│   └── main.py            # FastAPI 入口
├── web/                   # Web UI
├── scripts/               # 脚本工具
├── tests/                 # 测试文件
├── data/                  # 数据目录
│   ├── corpus/raw/        # 原始文档
│   ├── corpus/parsed/     # 解析后的 JSONL
│   └── index/             # 相似度索引
└── logs/                  # 日志目录
```

### 常用命令

```bash
# 安装依赖
make install          # 仅核心依赖
make dev              # 包含开发和相似度依赖

# 开发
make run              # 启动开发服务器
make test             # 运行测试
make lint             # 代码检查

# 语料库管理
make parse-corpus     # 解析文档
make build-index      # 构建索引

# 训练数据
make export-dpo       # 导出 DPO 训练数据
```

---

## 常见问题

### Q: vLLM 不可用怎么办？

A: 可以在没有 vLLM 的情况下运行基本功能测试：
```bash
python scripts/smoke_test.py --skip-llm
```

### Q: 没有 GPU 怎么测试？

A:
1. 相似度服务可以在 CPU 上完全运行
2. 可以使用 `--skip-llm` 选项跳过 LLM 相关测试
3. 生产使用需要 GPU 服务器

### Q: 如何添加新的语料库文档？

A:
```bash
# 1. 添加文档
cp new_paper.pdf data/corpus/raw/

# 2. 重新解析
python scripts/parse_corpus.py

# 3. 重建索引
python scripts/build_index.py

# 4. 重启服务器
make run
```

### Q: 如何导出反馈数据进行模型训练？

A:
```bash
# 导出 DPO 训练数据
python scripts/export_dpo_data.py

# 或指定输出路径和格式
python scripts/export_dpo_data.py --output ./training_data.jsonl --format huggingface
```

### Q: 如何自定义相似度阈值？

A: 创建 `.env` 文件：
```bash
cp .env.example .env
```

然后编辑 `.env`：
```bash
THRESHOLD_NGRAM_MAX_MATCH=10
THRESHOLD_EMBED_TOP1=0.85
```

### Q: 为什么显示 "System Degraded"？

A: 这通常表示 vLLM 服务器不可用。检查：
1. vLLM 是否正在运行
2. 端口 8000 是否正确
3. 运行 `curl http://localhost:8000/health` 测试连接

### Q: 如何查看详细日志？

A: 日志文件位于 `logs/` 目录：
```bash
# 查看审计日志
cat logs/audit.log

# 查看反馈记录
ls logs/feedback/
```

---

## 许可证

本项目仅供内部使用。

---

## 技术支持

如有问题，请查阅：
- `docs/SPEC.md` - 完整技术规格
- `docs/TASKS.md` - 开发任务列表
- `docs/API.md` - API 详细文档
