# GSWA 用户操作指南

## 目录
1. [快速开始](#快速开始)
2. [准备语料库](#准备语料库)
3. [启动服务](#启动服务)
4. [使用 Web UI](#使用-web-ui)
5. [提供反馈](#提供反馈)
6. [导出训练数据](#导出训练数据)
7. [微调模型](#微调模型)

---

## 快速开始

### 环境要求
- Python 3.10+
- CUDA GPU (推荐 24GB+ VRAM)
- 本地 vLLM 服务器

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd Gilles-Style-Writing-Assistant

# 2. 安装依赖
pip install -e .
pip install -e ".[similarity]"  # 安装相似度检测依赖

# 3. 复制配置文件
cp .env.example .env
```

---

## 准备语料库

语料库用于两个目的：
1. **相似度检测** - 防止生成与已发表论文太相似的文本
2. **风格参考** - 提供 Gilles 的写作风格样本

### 步骤 1: 收集文档

将 Gilles 的论文（PDF 或 DOCX 格式）放入：
```
data/corpus/raw/
├── paper1.pdf
├── paper2.pdf
├── paper3.docx
└── ...
```

### 步骤 2: 解析语料库

```bash
python scripts/parse_corpus.py
```

输出示例：
```
GSWA Corpus Parser
============================================================
Input: ./data/corpus/raw
Output: ./data/corpus/parsed

Found 15 documents:
  - PDF: 12
  - DOCX: 3

Processing: paper1.pdf...
  Extracted 45 paragraphs
Processing: paper2.pdf...
  Extracted 38 paragraphs
...

Total paragraphs extracted: 523
Output file: ./data/corpus/parsed/corpus.jsonl
```

### 步骤 3: 构建索引

```bash
python scripts/build_index.py
```

这会生成：
- `data/index/ngrams.json` - N-gram 索引
- `data/index/corpus.faiss` - 向量索引

---

## 启动服务

### 1. 启动 vLLM 服务器

```bash
# 使用默认 Mistral-7B-Instruct
./scripts/start_vllm.sh

# 或指定其他模型
VLLM_MODEL=meta-llama/Llama-3.2-7B-Instruct ./scripts/start_vllm.sh
```

vLLM 将在 `http://localhost:8000` 运行。

### 2. 启动 GSWA 服务器

```bash
make run
# 或
uvicorn gswa.main:app --reload --host 0.0.0.0 --port 8080
```

GSWA 将在 `http://localhost:8080` 运行。

---

## 使用 Web UI

1. 打开浏览器访问 `http://localhost:8080`

2. **粘贴段落**: 将要重写的段落粘贴到文本框中

3. **选择 Section**: 选择段落所属的论文部分（可选）
   - Abstract / Introduction / Methods / Results / Discussion / Conclusion

4. **选择变体数量**: 选择生成 3-5 个变体

5. **点击 "Generate Variants"**

6. **查看结果**:
   - 每个变体显示使用的策略 (A/B/C/D)
   - 显示相似度分数
   - 如果触发了 Fallback，会显示原因

---

## 提供反馈

反馈是训练微调模型的关键！

### 评分选项

对每个变体，选择一个评分：

| 评分 | 含义 | 何时使用 |
|------|------|----------|
| **Best** | 最佳 | 这是最好的变体，可以直接使用 |
| **Good** | 可接受 | 质量不错，但不是最佳 |
| **Bad** | 不可接受 | 有问题，不能使用 |
| **Edit** | 需要编辑 | 接近正确，但需要手动修改 |

### 评分建议

1. **至少评一个 "Best"** - 选出最好的变体
2. **标记 "Bad"** - 这对训练很重要！模型需要知道什么是不好的
3. **使用 "Edit"** - 如果变体接近正确但需要微调，编辑后保存

### 提交反馈

1. 为所有变体评分（至少一个）
2. 可选：添加备注
3. 点击 "Submit Feedback for Training"
4. 反馈会保存到 `logs/feedback/` 目录

---

## 导出训练数据

收集足够的反馈后（建议 100+ 个 session），导出 DPO 训练数据：

```bash
python scripts/export_dpo_data.py
```

输出示例：
```
GSWA DPO Data Export
============================================================
Feedback directory: ./logs/feedback
Output file: ./data/training/dpo_pairs.jsonl

Loading feedback records...
  Loading feedback_20240115.jsonl...
  Loaded 45 feedback sessions

Extracting DPO training pairs...
  Extracted 127 training pairs

Summary
============================================================
Total feedback sessions: 45
Total DPO pairs: 127
Output file: ./data/training/dpo_pairs.jsonl
```

### DPO 数据格式

```json
{
  "prompt": "We observed a 2.5-fold increase...",
  "chosen": "The rewritten text that was rated as best...",
  "rejected": "The rewritten text that was rated as bad...",
  "metadata": {
    "section": "Results",
    "chosen_strategy": "A",
    "rejected_strategy": "C"
  }
}
```

---

## 微调模型

### 推荐工具

1. **HuggingFace TRL** (推荐)
   ```bash
   pip install trl transformers
   ```

2. **OpenRLHF**

3. **axolotl**

### 使用 TRL 进行 DPO 训练

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# 加载数据
dataset = load_dataset("json", data_files="data/training/dpo_pairs.jsonl")

# 加载模型
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# DPO 配置
config = DPOConfig(
    output_dir="./gswa-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    num_train_epochs=3,
    beta=0.1,  # DPO 温度参数
)

# 训练
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()

# 保存模型
trainer.save_model("./gswa-finetuned")
```

### 训练建议

1. **数据量**: 建议 500+ 个 DPO pairs
2. **学习率**: 从 5e-7 开始，谨慎调整
3. **Beta**: 0.1 是默认值，可尝试 0.05-0.2
4. **验证**: 保留 10% 数据用于验证

---

## 完整工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        GSWA 工作流程                             │
└─────────────────────────────────────────────────────────────────┘

1. 准备阶段
   ├── 收集 Gilles 的论文 (PDF/DOCX)
   ├── 运行 parse_corpus.py 解析语料
   └── 运行 build_index.py 构建索引

2. 使用阶段
   ├── 启动 vLLM 服务器
   ├── 启动 GSWA 服务器
   └── 使用 Web UI 生成和评估变体

3. 反馈收集
   ├── 为每个变体评分 (Best/Good/Bad/Edit)
   ├── 提交反馈
   └── 重复直到收集足够数据 (建议 100+ sessions)

4. 模型微调
   ├── 运行 export_dpo_data.py 导出训练数据
   ├── 使用 TRL/DPOTrainer 进行微调
   └── 将微调后的模型部署到 vLLM

5. 迭代改进
   └── 使用微调后的模型继续收集反馈...
```

---

## 常见问题

### Q: 相似度检测报警，但我觉得文本是可以的？

相似度阈值是保守的。如果你确信重写是安全的：
1. 检查 `n-gram match` 是否确实是抄袭还是常见表达
2. 可以调整 `.env` 中的阈值（谨慎操作）

### Q: 生成的变体质量不高？

1. 确保语料库包含足够多的 Gilles 论文
2. 选择正确的 Section 类型
3. 输入段落应该完整、有意义
4. 考虑使用更大的模型

### Q: 如何知道收集了多少反馈？

访问 `http://localhost:8080/v1/feedback/stats` 查看统计信息。

---

## 文件结构参考

```
Gilles-Style-Writing-Assistant/
├── data/
│   ├── corpus/
│   │   ├── raw/           # 原始论文 (PDF/DOCX)
│   │   └── parsed/        # 解析后的 JSONL
│   ├── index/             # 相似度索引
│   └── training/          # 导出的训练数据
├── logs/
│   └── feedback/          # 用户反馈数据
├── scripts/
│   ├── parse_corpus.py    # 解析语料库
│   ├── build_index.py     # 构建索引
│   ├── export_dpo_data.py # 导出 DPO 数据
│   └── smoke_test.py      # 系统测试
├── src/gswa/              # 核心代码
└── web/                   # Web UI
```
