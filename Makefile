.PHONY: install dev test lint smoke-test run clean build-index parse-corpus export-dpo help setup setup-auto setup-mac setup-ollama prepare-training finetune-lora finetune-mlx finetune-mlx-safe finetune-all finetune-all-safe finetune-smart list-docs training-stats check-deps check-mlx check-lora corpus corpus-add corpus-guide corpus-validate train train-auto train-safe train-model status models analyze-style style-show ai-check humanize analyze-data preprocess-data train-linux train-linux-safe train-linux-full train-info

# Default target
help:
	@echo "GSWA - Gilles-Style Writing Assistant"
	@echo ""
	@echo "=== Quick Start (傻瓜式一键操作) ==="
	@echo ""
	@echo "  1. Setup:    make setup-cuda-auto  (Linux GPU) or make setup-auto (Mac)"
	@echo "  2. Train:    make finetune-smart   (auto-detect platform & hardware)"
	@echo "  3. Run:      make run              (start server at localhost:8080)"
	@echo ""
	@echo "=== All Commands ==="
	@echo ""
	@echo "  Setup:"
	@echo "    setup-auto       Automatic setup (no prompts)"
	@echo "    setup-cuda-auto  Automatic setup with CUDA (NVIDIA GPU)"
	@echo "    train-info       Show hardware info and recommendations"
	@echo ""
	@echo "  Training:"
	@echo "    finetune-smart   One-click training (auto-detect platform) - RECOMMENDED"
	@echo "    parse-corpus     Parse PDF/DOCX files to training data"
	@echo "    status           Show corpus and training status"
	@echo ""
	@echo "  Server:"
	@echo "    run              Start development server"
	@echo "    test             Run unit tests"
	@echo ""
	@echo "  AI Detection:"
	@echo "    ai-check         Check text for AI traces"
	@echo "    humanize         Auto-humanize text"
	@echo ""
	@echo "  Advanced (see docs/TRAINING_GUIDE.md):"
	@echo "    train-safe       Memory-safe training (Mac)"
	@echo "    train-linux-safe Linux CUDA training with OOM fallback"

# ==================
# One-Click Setup
# ==================

# One-click environment setup (auto-install Python 3.10+ if needed)
# 傻瓜式一键环境配置
setup:
	@bash scripts/setup.sh --dev

# Automatic setup (no prompts)
setup-auto:
	@bash scripts/setup.sh --auto --dev

# Setup with CUDA support (for NVIDIA GPUs)
# 傻瓜式一键CUDA环境配置
setup-cuda:
	@bash scripts/setup.sh --dev --cuda

# Automatic CUDA setup (no prompts)
setup-cuda-auto:
	@bash scripts/setup.sh --auto --dev --cuda

# ==================
# Installation
# ==================

# Install core dependencies
install:
	pip install -e .

# Install dev dependencies
dev:
	pip install -e ".[dev,similarity]"

# Install training dependencies
install-train:
	pip install torch transformers peft datasets accelerate bitsandbytes

# Install MLX training dependencies (Mac only)
install-mlx:
	pip install mlx mlx-lm

# Run tests
test:
	python -m pytest tests/ -v

# Lint code
lint:
	ruff check src/

# Run smoke test (requires server to be running)
smoke-test:
	python scripts/smoke_test.py

# Run smoke test (security only, no server needed)
smoke-test-security:
	python scripts/smoke_test.py --skip-llm

# Start GSWA server (development mode)
run:
	uvicorn gswa.main:app --reload --host 0.0.0.0 --port 8080

# Start GSWA server (production mode)
run-prod:
	uvicorn gswa.main:app --host 0.0.0.0 --port 8080 --workers 4

# ==================
# Mac Setup (Ollama)
# ==================

# Complete Mac setup
setup-mac: setup-ollama dev
	@echo ""
	@echo "Mac setup complete! Run 'make run' to start the server."

# Setup Ollama for Mac
setup-ollama:
	bash scripts/setup_ollama.sh

# ==================
# Linux Setup (vLLM)
# ==================

# Start vLLM server
start-vllm:
	bash scripts/start_vllm.sh

# ==================
# Corpus Management
# ==================

# Build similarity index from corpus
build-index:
	python scripts/build_index.py

# Parse corpus documents (PDF/DOCX -> JSONL)
parse-corpus:
	python scripts/parse_corpus.py

# ==================
# Fine-tuning
# ==================

# Prepare training data from corpus (with priority weights)
prepare-training:
	python scripts/prepare_training_data.py --format alpaca --weighted --split

# Fine-tune with LoRA (Linux/Windows with NVIDIA GPU)
# Uses auto-detection to select optimal settings for your hardware
finetune-lora: prepare-training
	python scripts/finetune_lora.py --auto

# Fine-tune with MLX (Mac Apple Silicon)
# Uses auto-detection to select optimal settings for your hardware
# See config/training_profiles.json to customize
finetune-mlx: prepare-training
	python scripts/finetune_mlx_mac.py --auto

# Fine-tune with MLX in memory-safe mode (RECOMMENDED for OOM errors)
# Automatically preprocesses long sequences and uses conservative settings
finetune-mlx-safe: prepare-training
	@echo ""
	@echo "============================================"
	@echo "Starting Memory-Safe MLX Training"
	@echo "============================================"
	@echo ""
	@echo "This mode will:"
	@echo "  1. Automatically split long sequences"
	@echo "  2. Use conservative memory settings"
	@echo "  3. Retry with reduced settings if OOM occurs"
	@echo ""
	python scripts/finetune_mlx_mac.py --auto --memory-safe

# One-click fine-tuning for Mac (parse + prepare + finetune)
finetune-all: parse-corpus prepare-training finetune-mlx
	@echo ""
	@echo "============================================"
	@echo "Fine-tuning complete!"
	@echo "============================================"
	@echo ""
	@echo "Next steps:"
	@echo "  1. ollama create gswa-gilles -f models/gswa-mlx-*/Modelfile"
	@echo "  2. Add to .env: VLLM_MODEL_NAME=gswa-gilles"
	@echo "  3. make run"

# One-click fine-tuning for Mac with memory-safe mode (RECOMMENDED)
# 一键傻瓜式训练（推荐用于内存不足问题）
finetune-all-safe: parse-corpus prepare-training finetune-mlx-safe
	@echo ""
	@echo "============================================"
	@echo "Memory-Safe Fine-tuning Complete!"
	@echo "============================================"
	@echo ""
	@echo "Next steps:"
	@echo "  1. ollama create gswa-gilles -f models/gswa-mlx-*/Modelfile"
	@echo "  2. Add to .env: VLLM_MODEL_NAME=gswa-gilles"
	@echo "  3. make run"

# Export feedback data for DPO training
export-dpo:
	python scripts/export_dpo_data.py

# List all document IDs in the corpus
list-docs:
	@python3 -c "import json; docs=set(); f=open('data/corpus/parsed/corpus.jsonl'); [docs.add(json.loads(l).get('doc_id','')) for l in f]; print('\\n'.join(sorted(docs)))"

# Show training data statistics
training-stats:
	@echo "=== Corpus Statistics ==="
	@python3 -c "import json; lines=list(open('data/corpus/parsed/corpus.jsonl')); print(f'Total paragraphs: {len(lines)}'); priority=sum(1 for l in lines if json.loads(l).get('is_priority')); print(f'Priority paragraphs: {priority}'); print(f'Regular paragraphs: {len(lines)-priority}')" 2>/dev/null || echo "No corpus found. Run: make parse-corpus"
	@echo ""
	@echo "=== Training Data ==="
	@if [ -f data/training/alpaca_train.jsonl ]; then wc -l data/training/alpaca_*.jsonl; else echo "No training data found. Run: make prepare-training"; fi

# Check MLX dependencies (Mac)
check-mlx:
	python scripts/finetune_mlx_mac.py --check-only

# Check LoRA dependencies (Linux)
check-lora:
	python scripts/finetune_lora.py --check-only

# ==================
# Smart Training
# ==================

# Smart fine-tuning: auto-detect platform and hardware
# Works on Mac, Linux, and Windows
finetune-smart: parse-corpus prepare-training
	python scripts/smart_finetune.py
	@echo ""
	@echo "============================================"
	@echo "🎉 Fine-tuning complete!"
	@echo "============================================"

# ==================
# Corpus Management
# ==================

# Show corpus status
corpus:
	python scripts/corpus_manager.py

# Show quick guide for adding corpus files
corpus-guide:
	python scripts/corpus_manager.py --guide

# Validate all corpus files
corpus-validate:
	python scripts/corpus_manager.py --validate

# ==================
# Training Wizard (One-Click Training)
# ==================

# One-click training wizard (recommended)
train:
	python scripts/training_wizard.py

# Fully automatic training (no confirmation)
train-auto:
	python scripts/training_wizard.py --auto

# Memory-safe training (RECOMMENDED for OOM errors)
# 傻瓜式一键训练（推荐）
train-safe:
	@echo ""
	@echo "============================================"
	@echo "GSWA Memory-Safe Training Mode"
	@echo "傻瓜式一键训练模式"
	@echo "============================================"
	@echo ""
	@echo "This will:"
	@echo "  1. Parse corpus files (解析语料)"
	@echo "  2. Prepare training data (准备训练数据)"
	@echo "  3. Analyze and preprocess long sequences (预处理长序列)"
	@echo "  4. Train with memory-safe settings (内存安全训练)"
	@echo ""
	$(MAKE) finetune-all-safe

# Train with specific model
# Usage: make train-model MODEL=qwen-7b
train-model:
	@echo "Usage: make train-model MODEL=qwen-7b"
	@echo "Available models: qwen-7b, qwen-14b, qwen-1.5b, llama3-8b, mistral-7b, phi-3.5"
ifdef MODEL
	python scripts/training_wizard.py --model $(MODEL) --auto
endif

# Analyze training data (check for long sequences that may cause OOM)
analyze-data:
	@echo ""
	@echo "============================================"
	@echo "Analyzing Training Data"
	@echo "============================================"
	python scripts/preprocess_training_data.py --analyze

# Preprocess training data to split long sequences
# Usage: make preprocess-data MAX_TOKENS=1024
preprocess-data:
	@echo ""
	@echo "============================================"
	@echo "Preprocessing Training Data"
	@echo "============================================"
ifdef MAX_TOKENS
	python scripts/preprocess_training_data.py --max-tokens $(MAX_TOKENS) --in-place
else
	python scripts/preprocess_training_data.py --auto --in-place
endif

# Show status (combines corpus + training stats)
status:
	python scripts/training_wizard.py --status

# List available models
models:
	python scripts/training_wizard.py --models

# ==================
# Utilities
# ==================

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# ==================
# Style Analysis & AI Detection
# ==================

# Analyze author writing style from corpus
analyze-style:
	python scripts/analyze_style.py

# Show current style fingerprint
style-show:
	@if [ -f data/style/author_fingerprint.json ]; then \
		python scripts/analyze_style.py --verbose; \
	else \
		echo "No style fingerprint found. Run: make analyze-style"; \
	fi

# Check text for AI traces (interactive) - Scientific analysis
ai-check:
	@echo "=== Scientific AI Detection ==="
	@echo "Based on: Perplexity, Burstiness, Vocabulary Diversity, Style Consistency"
	@echo ""
	@echo "Paste your text and press Ctrl+D (or Ctrl+Z on Windows) when done:"
	@echo ""
	@python3 -c "\
import sys; sys.path.insert(0,'src'); \
from gswa.utils.ai_detector import analyze_text_detailed; \
text=sys.stdin.read().strip(); \
r=analyze_text_detailed(text); \
print(f\"\n{'='*50}\"); \
print(f\"OVERALL: AI Score = {r['overall']['ai_score']:.2f} | Risk: {r['overall']['risk_level'].upper()}\"); \
print(f\"Confidence: {r['overall']['confidence']:.0%}\"); \
print(f\"{'='*50}\"); \
print(f\"\nMETRICS (lower score = more human-like):\"); \
m=r['metrics']; \
print(f\"  Perplexity:     {m['perplexity']['value']:>6.1f}  (score: {m['perplexity']['score']:.2f})\"); \
print(f\"  Burstiness:     {m['burstiness']['value']:>6.3f}  (score: {m['burstiness']['score']:.2f})\"); \
print(f\"  Vocab Diversity:{m['vocabulary_diversity']['value']:>6.3f}  (score: {m['vocabulary_diversity']['score']:.2f})\"); \
print(f\"  Style Match:    {m['style_consistency']['value']:>6.3f}  (score: {m['style_consistency']['score']:.2f})\"); \
print(f\"\nSentence lengths: {r['sentence_lengths']}\"); \
print(f\"\nSUGGESTIONS:\"); \
[print(f'  - {s}') for s in r['suggestions'][:5]]"

# Humanize text (interactive) - Reduce AI detection score
humanize:
	@echo "=== Text Humanizer ==="
	@echo "Automatically reduces AI detection score"
	@echo ""
	@echo "Paste your text and press Ctrl+D (or Ctrl+Z on Windows) when done:"
	@echo ""
	@python3 -c "\
import sys; sys.path.insert(0,'src'); \
from gswa.utils.ai_detector import humanize_text, detect_ai_traces; \
text=sys.stdin.read().strip(); \
before=detect_ai_traces(text); \
result=humanize_text(text); \
after=detect_ai_traces(result); \
print(f\"\n{'='*50}\"); \
print(f\"AI Score: {before.ai_score:.2f} -> {after.ai_score:.2f} ({(before.ai_score-after.ai_score)*100:+.0f}%)\"); \
print(f\"Burstiness: {before.burstiness:.3f} -> {after.burstiness:.3f}\"); \
print(f\"{'='*50}\"); \
print(f\"\nHUMANIZED TEXT:\n\"); \
print(result)"

# ==================
# Linux One-Click Training
# ==================

# Show hardware info and training recommendations
train-info:
	@echo ""
	@echo "============================================"
	@echo "GSWA Hardware Detection & Recommendations"
	@echo "============================================"
	python -m gswa.train info

# One-click Linux/CUDA training with auto-configuration
# 傻瓜式一键Linux训练
train-linux: parse-corpus prepare-training
	@echo ""
	@echo "============================================"
	@echo "GSWA One-Click Linux/CUDA Training"
	@echo "Linux傻瓜式一键训练"
	@echo "============================================"
	@echo ""
	@echo "This will automatically:"
	@echo "  1. Detect your NVIDIA GPU and VRAM"
	@echo "  2. Select optimal training parameters"
	@echo "  3. Preprocess long sequences"
	@echo "  4. Train with progress visualization"
	@echo "  5. Generate training reports"
	@echo ""
	python -m gswa.train train --preprocess -y

# Memory-safe Linux training with OOM fallback (RECOMMENDED)
# 内存安全Linux训练（推荐）
train-linux-safe: parse-corpus prepare-training
	@echo ""
	@echo "============================================"
	@echo "GSWA Memory-Safe Linux Training"
	@echo "Linux内存安全训练模式"
	@echo "============================================"
	@echo ""
	@echo "This mode will:"
	@echo "  1. Detect hardware and use conservative settings"
	@echo "  2. Preprocess long sequences (no truncation)"
	@echo "  3. Automatically retry with reduced settings on OOM"
	@echo "  4. Generate visualizations and reports"
	@echo ""
	python -m gswa.train train --preprocess -y

# Full Linux training pipeline with planner
train-linux-full: parse-corpus prepare-training
	@echo ""
	@echo "============================================"
	@echo "GSWA Full Linux Training Pipeline"
	@echo "============================================"
	python -m gswa.train train --preprocess --auto-plan -y
	@echo ""
	@echo "============================================"
	@echo "Training Complete!"
	@echo "============================================"
	@echo ""
	@echo "Check runs/ directory for outputs."
	@echo "View report: open runs/<run-id>/reports/report.html"
