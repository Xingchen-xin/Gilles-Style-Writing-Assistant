# GSWA - Gilles-Style Writing Assistant

> 完全本地离线运行的科学论文段落重写工具

## 🎯 项目概述

GSWA 是一个本地部署的 AI 写作助手，专门用于将科学论文段落重写为符合特定写作风格（Gilles 风格）的版本。

### 核心特性

- 🔒 **完全离线** - 无外部 API 调用，数据不出本地
- 📝 **多版本输出** - 同一输入生成 3-5 个不同组织策略的候选版本
- 🎯 **语义保持** - 不改变数值、实验条件、结论强度
- 🔄 **去重保护** - N-gram + 向量相似度检测，自动回退重写

## 🏗️ 系统架构

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Web UI    │────▶│  FastAPI Server  │────▶│ vLLM Server │
└─────────────┘     │  (Orchestrator)  │     │ (Local LLM) │
                    │        │         │     └─────────────┘
                    │        ▼         │
                    │ ┌─────────────┐  │
                    │ │  Similarity │  │
                    │ │   Service   │  │
                    │ └─────────────┘  │
                    └──────────────────┘
```

## 📁 文档结构

```
docs/
├── SPEC.md              # 完整技术规格
├── TASKS.md             # 任务拆解（11个PR）
└── CLAUDE_CODE_PROMPT.md # Claude Code 开发指令
```

## 🚀 快速开始

### 1. 阅读规格文档

```bash
# 理解项目需求
cat docs/SPEC.md
```

### 2. 按任务顺序开发

```bash
# 查看任务列表
cat docs/TASKS.md
```

### 3. 遵循开发指令

```bash
# Claude Code 专用指令
cat docs/CLAUDE_CODE_PROMPT.md
```

## 🔐 安全约束

| 约束 | 说明 |
|------|------|
| 无外部调用 | `ALLOW_EXTERNAL_API=false` 硬编码默认值 |
| 仅本地 URL | 只允许 localhost/127.0.0.1 |
| 无遥测 | 不发送任何使用数据 |
| 日志脱敏 | 不记录原文，只记录 hash |

## 📋 MVP 交付物

1. ✅ FastAPI 服务器
   - `POST /v1/rewrite/variants` - 生成重写变体
   - `POST /v1/reply` - 简单聊天
   - `GET /v1/health` - 健康检查

2. ✅ 相似度服务
   - N-gram 重叠检测
   - 向量相似度 (FAISS)
   - 自动回退机制

3. ✅ Web UI
   - 粘贴 → 生成 → 复制
   - 显示相似度分数
   - 回退状态标记

4. ✅ vLLM 集成
   - OpenAI 兼容 API
   - 本地模型推理

5. ✅ 测试脚本
   - 单元测试
   - 端到端 smoke test

## 🎨 API 示例

### 重写请求

```bash
curl -X POST http://localhost:8080/v1/rewrite/variants \
  -H "Content-Type: application/json" \
  -d '{
    "text": "We observed a 2.5-fold increase in enzyme activity...",
    "section": "Results",
    "n_variants": 3
  }'
```

### 响应

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
  "model_version": "mistral-7b-instruct@v1",
  "processing_time_ms": 2450
}
```

## 📊 相似度阈值

| 指标 | 阈值 | 触发动作 |
|------|------|----------|
| N-gram 最长匹配 | ≥ 12 tokens | 回退重写 |
| 向量相似度 top-1 | ≥ 0.88 | 回退重写 |

## 🛠️ 开发命令

```bash
# 安装依赖
make install

# 安装开发依赖
make dev

# 运行测试
make test

# 启动服务器
make run

# 运行 smoke test
make smoke-test
```

## 📝 开发任务清单

- [ ] PR #1: 项目骨架
- [ ] PR #2: Pydantic 数据模型
- [ ] PR #3: N-gram 服务
- [ ] PR #4: 向量相似度服务
- [ ] PR #5: 组合相似度服务
- [ ] PR #6: LLM 客户端
- [ ] PR #7: Prompt 服务
- [ ] PR #8: 重写编排器
- [ ] PR #9: FastAPI 路由
- [ ] PR #10: Web UI
- [ ] PR #11: Smoke Test

---

**For Claude Code:** 请从 `docs/CLAUDE_CODE_PROMPT.md` 开始，按照 `docs/TASKS.md` 中的任务顺序实现。