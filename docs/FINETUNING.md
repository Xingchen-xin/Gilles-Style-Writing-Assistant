# GSWA Fine-tuning Guide

## 为什么需要微调？

当前问题：
1. **AI 检测器识别** - 生成的文本被识别为纯 AI 生成
2. **风格不匹配** - 输出不像 Gilles 的写作风格
3. **通用性过强** - 模型没有学习 Gilles 特有的表达方式

解决方案：**微调模型使其学习 Gilles 的写作风格**

---

## 微调方案对比

| 方案 | 硬件要求 | 训练时间 | 质量 | 难度 | 推荐场景 |
|------|----------|----------|------|------|----------|
| **LoRA** | GPU 16GB+ | 2-4小时 | ⭐⭐⭐⭐ | 中 | Linux 服务器 |
| **QLoRA** | GPU 8GB+ | 3-6小时 | ⭐⭐⭐ | 中 | 显存有限 |
| **MLX (Mac)** | M1/M2/M3 16GB+ | 1-2小时 | ⭐⭐⭐⭐ | 低 | Mac 用户推荐 |
| **Full Fine-tuning** | GPU 48GB+ | 8-24小时 | ⭐⭐⭐⭐⭐ | 高 | 最佳质量 |
| **RAG 增强** | 无需训练 | 即时 | ⭐⭐⭐ | 低 | 快速部署 |

---

## 快速开始（Mac 用户）

### 第一步：配置优先文档权重

编辑 `data/corpus/priority_weights.json`，设置 Gilles 认为最能代表其风格的文章：

```json
{
  "priority_docs": {
    "Barka_MicrobiolMolBiolRev2016": {
      "weight": 2.5,
      "reason": "Comprehensive review - exemplary writing style"
    },
    "van Wezel_McDowall_NPR2011": {
      "weight": 2.5,
      "reason": "Classic Gilles style"
    },
    "vanderMeij_FEMSMicrobiolRev2017": {
      "weight": 2.0,
      "reason": "Major review paper"
    }
  }
}
```

**权重说明：**
- `1.0` = 正常权重
- `2.0` = 双倍重要性（训练时出现 2 次）
- `2.5` = 最高优先级
- `0.5` = 降低重要性

### 第二步：准备训练数据

```bash
# 生成加权训练数据
make prepare-training
```

这会创建 `data/training/alpaca_train.jsonl` 和 `alpaca_val.jsonl`。

### 第三步：安装训练依赖（Mac）

```bash
# 安装 MLX
pip install mlx mlx-lm
```

### 第四步：运行微调

```bash
# Mac 用户
make finetune-mlx

# 或者手动运行
python scripts/finetune_mlx_mac.py --model mistral --iters 1000
```

### 第五步：使用微调后的模型

```bash
# 1. 模型会自动创建 Ollama Modelfile
# 2. 创建 Ollama 模型
ollama create gswa-gilles -f models/gswa-mlx-*/Modelfile

# 3. 更新 .env
VLLM_MODEL_NAME=gswa-gilles

# 4. 重启 GSWA
make run
```

---

## 详细教程

### 1. 语料库优先级设置

Gilles 的论文库中，有些文章更能代表他的写作风格。将这些放入高权重：

**放置位置：** `data/corpus/priority_weights.json`

```json
{
  "priority_docs": {
    "最能代表风格的文章ID": {
      "weight": 2.5,
      "reason": "说明为什么这篇最能代表 Gilles 风格"
    }
  },
  "exclude_docs": {
    "不代表风格的文章ID": {
      "reason": "为什么排除"
    }
  }
}
```

**如何确定文章 ID？**
```bash
# 查看所有文章 ID
python3 << 'EOF'
import json
with open('data/corpus/parsed/corpus.jsonl') as f:
    docs = set()
    for line in f:
        d = json.loads(line)
        docs.add(d['doc_id'])
    for doc in sorted(docs):
        print(doc)
EOF
```

### 2. 训练数据格式

脚本支持多种格式：

| 格式 | 用途 | 命令 |
|------|------|------|
| Alpaca | 通用微调 | `--format alpaca` |
| ShareGPT | Axolotl 等 | `--format sharegpt` |
| Completion | 继续预训练 | `--format completion` |
| DPO | 偏好学习 | `--format dpo --from-feedback` |

```bash
# 生成所有格式
python scripts/prepare_training_data.py --format all --weighted --split
```

### 3. Linux GPU 微调 (LoRA/QLoRA)

```bash
# 安装依赖
make install-train

# 准备数据
make prepare-training

# QLoRA 微调（推荐，节省显存）
python scripts/finetune_lora.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --quantize 4bit \
    --epochs 3 \
    --batch-size 4

# 全精度 LoRA（更高质量）
python scripts/finetune_lora.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --quantize none \
    --epochs 3
```

### 4. Mac MLX 微调

```bash
# 安装 MLX
pip install mlx mlx-lm

# 运行微调
python scripts/finetune_mlx_mac.py \
    --model mistral \
    --batch-size 4 \
    --lora-layers 16 \
    --iters 1000
```

---

## 高级：DPO 训练（偏好对齐）

DPO (Direct Preference Optimization) 使用用户反馈来进一步优化模型。

### 收集反馈

1. 使用 GSWA 生成变体
2. 在 UI 中为变体评分（Best/Good/Bad）
3. 提交反馈

### 导出 DPO 数据

```bash
make export-dpo
```

### 运行 DPO 训练

```bash
python scripts/prepare_training_data.py --format dpo --from-feedback
python scripts/finetune_lora.py --training-data ./data/training/dpo.jsonl
```

---

## 如何减少 AI 检测？

微调后的模型会更好地模仿人类写作风格，但还可以采取以下措施：

### 1. 调整生成参数

在 `.env` 中设置：
```bash
TEMPERATURE_BASE=0.4      # 略高的温度增加变化
TEMPERATURE_VARIANCE=0.2  # 变体间更大差异
```

### 2. 使用更多样的 Prompt

系统会自动使用不同的重写策略（A/B/C/D）。

### 3. 启用回退重写

确保相似度检测开启，避免与语料库过于相似。

### 4. 后处理

- 轻微编辑生成的文本
- 添加个人表达
- 调整句子结构

---

## 故障排除

### Q: MLX 训练太慢？

A: 减少迭代次数或使用更小的模型：
```bash
python scripts/finetune_mlx_mac.py --model phi --iters 500
```

### Q: 显存不足 (CUDA OOM)？

A: 使用 4-bit 量化：
```bash
python scripts/finetune_lora.py --quantize 4bit --batch-size 2
```

### Q: 生成质量下降？

A: 可能是过拟合，尝试：
- 减少训练轮数
- 增加 dropout
- 使用更多验证数据

### Q: 如何回滚到原模型？

A: 修改 `.env`：
```bash
VLLM_MODEL_NAME=mistral  # 使用原始模型
```

---

## 推荐工作流程

```
1. 配置优先文档权重
   └── 编辑 data/corpus/priority_weights.json

2. 准备训练数据
   └── make prepare-training

3. 微调模型
   ├── Mac: make finetune-mlx
   └── Linux: make finetune-lora

4. 部署微调模型
   └── ollama create gswa-gilles -f Modelfile

5. 使用并收集反馈
   └── 在 UI 中评分变体

6. DPO 迭代优化
   └── make export-dpo && 重新训练
```

---

## 参考资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [MLX 文档](https://ml-explore.github.io/mlx/)
- [Ollama 模型创建](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
