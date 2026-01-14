# GSWA Fine-tuning Guide / 微调指南

## TL;DR 傻瓜式操作

```
只需 3 步：

1. 放文章到文件夹
   data/corpus/raw/           <- 普通文章放这里
   data/corpus/raw/important_examples/  <- 重要文章放这里（权重 2.5x）

2. 运行一条命令
   make finetune-all

3. 重启 GSWA
   make run
```

---

## 文件夹结构说明

```
data/corpus/raw/                      <- 普通 Gilles 文章
├── paper1.pdf
├── paper2.docx
├── paper3.txt
│
└── important_examples/               <- 重要/代表性文章 (自动 2.5x 权重)
    ├── best_review.pdf
    └── classic_paper.pdf
```

**权重说明：**
| 位置 | 自动权重 | 说明 |
|------|----------|------|
| `raw/` | 1.0x | 普通文章 |
| `raw/important_examples/` | 2.5x | 重要文章，训练时出现更多次 |

**支持的文件格式：** `.pdf`, `.docx`, `.txt`

---

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
| **MLX (Mac)** | M1/M2/M3 16GB+ | 1-2小时 | ⭐⭐⭐⭐ | 低 | **Mac 用户首选** |
| **LoRA** | GPU 16GB+ | 2-4小时 | ⭐⭐⭐⭐ | 中 | Linux 服务器 |
| **QLoRA** | GPU 8GB+ | 3-6小时 | ⭐⭐⭐ | 中 | 显存有限 |
| **Full Fine-tuning** | GPU 48GB+ | 8-24小时 | ⭐⭐⭐⭐⭐ | 高 | 最佳质量 |

---

## Mac 用户傻瓜式教程

### 第一步：放入文章

1. 打开 Finder，进入项目目录
2. 打开 `data/corpus/raw/` 文件夹
3. 把 Gilles 的 PDF 文章拖进去
4. 如果有最能代表 Gilles 风格的文章，放入 `raw/important_examples/`

```bash
# 或者用命令行
cp ~/Downloads/*.pdf data/corpus/raw/

# 重要文章放这里
cp ~/Downloads/important_paper.pdf data/corpus/raw/important_examples/
```

### 第二步：安装依赖（首次运行）

```bash
# 安装 MLX（Apple Silicon 专用机器学习库）
pip install mlx mlx-lm
```

### 第三步：一键微调

```bash
# 这一条命令完成所有工作：解析文章 → 生成训练数据 → 微调模型
make finetune-all
```

**看到的输出：**
```
============================================================
GSWA Corpus Parser
============================================================
Input (regular):  ./data/corpus/raw
Input (priority): ./data/corpus/raw/important_examples

Found 15 documents:
  - Regular articles:  12
  - Priority articles: 3 (in important_examples/)

Processing: paper1.pdf...
  Extracted 45 paragraphs
Processing: best_review.pdf ⭐...
  Extracted 120 paragraphs
...

============================================================
Starting MLX Fine-tuning
============================================================
Epoch 1/3: loss=2.45
Epoch 2/3: loss=1.89
Epoch 3/3: loss=1.23

Model saved to: models/gswa-mlx-mistral/
```

### 第四步：创建 Ollama 模型

```bash
# 根据输出的模型路径创建 Ollama 模型
ollama create gswa-gilles -f models/gswa-mlx-mistral/Modelfile
```

### 第五步：更新配置并运行

```bash
# 更新 .env 使用新模型
echo "VLLM_MODEL_NAME=gswa-gilles" >> .env

# 重启 GSWA
make run
```

**恭喜！现在 GSWA 使用的是微调后的模型！**

---

## Linux 用户傻瓜式教程

### 第一步：放入文章

同 Mac 用户，放入 `data/corpus/raw/` 和 `raw/important_examples/`

### 第二步：安装依赖

```bash
# 安装训练依赖
make install-train
```

### 第三步：一键微调

```bash
# QLoRA 微调（推荐，节省显存）
make finetune-lora
```

### 第四步：部署模型

微调完成后，模型保存在 `models/gswa-lora/`。

```bash
# 使用 PEFT 合并模型（可选）
python scripts/merge_lora.py

# 或者直接配置 .env 使用 LoRA adapter
LORA_ADAPTER_PATH=./models/gswa-lora
```

---

## 硬件配置文件（自动检测）

系统会自动检测你的硬件并选择最佳训练参数：

| 硬件 | 内存 | batch_size | num_layers | iters |
|------|------|------------|------------|-------|
| M1/M2/M3 8GB | 8GB | 1 | 4 | 300 |
| M1/M2/M3 16GB | 16GB | 2 | 8 | 500 |
| M1/M2/M3 Max 32GB+ | 32GB+ | 4 | 16 | 1000 |
| M1/M2/M3 Ultra 64GB+ | 64GB+ | 8 | 32 | 1500 |

**自定义配置：** 编辑 `config/training_profiles.json` 文件。

**查看系统检测结果：**
```bash
python scripts/finetune_mlx_mac.py --auto --check-only
```

---

## 手动配置权重（高级）

如果你想精确控制每篇文章的权重，可以编辑 `data/corpus/priority_weights.json`：

```json
{
  "default_weight": 1.0,
  "priority_folder_weight": 2.5,

  "priority_docs": {
    "Barka_MicrobiolMolBiolRev2016": {
      "weight": 3.0,
      "reason": "最能代表 Gilles 风格的综述"
    }
  },

  "exclude_docs": {
    "some_bad_paper": {
      "reason": "太短，不能代表风格"
    }
  }
}
```

**查看所有文章 ID：**
```bash
make list-docs
```

---

## 完整 Makefile 命令

```bash
make parse-corpus      # 解析 raw/ 中的文章
make prepare-training  # 生成训练数据
make finetune-mlx      # Mac MLX 微调
make finetune-lora     # Linux LoRA 微调
make finetune-all      # 一键完成所有步骤

make list-docs         # 列出所有文章 ID
make training-stats    # 查看训练数据统计
```

---

## 如何减少 AI 检测？

微调后的模型会更好地模仿人类写作风格，但还可以采取以下措施：

### 1. 使用高质量语料

- 放入更多 Gilles 的文章（越多越好）
- 把最能代表风格的放入 `important_examples/`
- 排除不典型的文章

### 2. 调整生成参数

在 `.env` 中设置：
```bash
TEMPERATURE_BASE=0.4      # 略高的温度增加变化
TEMPERATURE_VARIANCE=0.2  # 变体间更大差异
```

### 3. 后处理

- 轻微编辑生成的文本
- 添加个人表达
- 调整句子结构

---

## DPO 进阶训练（偏好对齐）

使用后，你可以通过反馈进一步优化：

1. 使用 GSWA 生成变体
2. 在 UI 中为变体评分（Best/Good/Bad）
3. 提交反馈
4. 导出并训练：

```bash
make export-dpo
python scripts/prepare_training_data.py --format dpo --from-feedback
make finetune-lora
```

---

## 故障排除

### Q: 没有检测到文章？

A: 检查文件位置和格式：
```bash
ls data/corpus/raw/
ls data/corpus/raw/important_examples/
```
确保是 `.pdf`, `.docx`, 或 `.txt` 文件。

### Q: MLX 训练太慢？

A: 减少迭代次数或使用更小的模型：
```bash
python scripts/finetune_mlx_mac.py --model phi --iters 500
```

### Q: 内存不足 (Mac)?

A: 系统会自动检测你的硬件并选择合适的配置。如需手动调整：
```bash
# 查看可用配置
python scripts/finetune_mlx_mac.py --list-profiles

# 使用保守配置（最低内存）
python scripts/finetune_mlx_mac.py --profile conservative

# 或者手动设置参数
python scripts/finetune_mlx_mac.py --batch-size 1 --num-layers 4 --max-seq-length 512
```

### Q: 显存不足 (CUDA OOM)？

A: 使用 4-bit 量化：
```bash
python scripts/finetune_lora.py --quantize 4bit --batch-size 2
```

### Q: 生成质量下降？

A: 可能是过拟合，尝试：
- 减少训练轮数
- 增加更多文章
- 使用验证集

### Q: 如何回滚到原模型？

A: 修改 `.env`：
```bash
VLLM_MODEL_NAME=mistral  # 使用原始模型
```

---

## 参考资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [MLX 文档](https://ml-explore.github.io/mlx/)
- [Ollama 模型创建](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
