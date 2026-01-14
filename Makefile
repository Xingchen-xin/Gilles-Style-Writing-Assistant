.PHONY: install dev test lint smoke-test run clean build-index parse-corpus export-dpo help setup-mac setup-ollama prepare-training finetune-lora finetune-mlx finetune-all list-docs training-stats check-deps check-mlx check-lora

# Default target
help:
	@echo "GSWA - Gilles-Style Writing Assistant"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  === Installation ==="
	@echo "  install        - Install core dependencies"
	@echo "  dev            - Install development dependencies"
	@echo "  install-train  - Install training dependencies (LoRA)"
	@echo ""
	@echo "  === Mac Setup (Apple Silicon) ==="
	@echo "  setup-mac      - Complete Mac setup with Ollama"
	@echo "  setup-ollama   - Install and configure Ollama"
	@echo ""
	@echo "  === Linux Setup (NVIDIA GPU) ==="
	@echo "  start-vllm     - Start vLLM server"
	@echo ""
	@echo "  === Running ==="
	@echo "  run            - Start GSWA server (development)"
	@echo "  run-prod       - Start GSWA server (production)"
	@echo ""
	@echo "  === Testing ==="
	@echo "  test           - Run unit tests"
	@echo "  lint           - Run linter"
	@echo "  smoke-test     - Run end-to-end smoke test"
	@echo ""
	@echo "  === Corpus Management ==="
	@echo "  parse-corpus   - Parse PDF/DOCX files to JSONL"
	@echo "  build-index    - Build similarity index from corpus"
	@echo ""
	@echo "  === Fine-tuning (Gilles Style) ==="
	@echo "  finetune-all      - One-click: parse + train + finetune (Mac)"
	@echo "  prepare-training  - Prepare training data from corpus"
	@echo "  finetune-lora     - Fine-tune with LoRA (Linux/GPU)"
	@echo "  finetune-mlx      - Fine-tune with MLX (Mac Apple Silicon)"
	@echo "  export-dpo        - Export feedback for DPO training"
	@echo "  list-docs         - List all document IDs in corpus"
	@echo "  training-stats    - Show training data statistics"
	@echo "  check-mlx         - Check MLX dependencies (Mac)"
	@echo "  check-lora        - Check LoRA dependencies (Linux)"
	@echo ""
	@echo "  === Utilities ==="
	@echo "  clean          - Clean build artifacts"

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

# Fine-tune with LoRA (Linux with NVIDIA GPU)
finetune-lora: prepare-training
	python scripts/finetune_lora.py --quantize 4bit

# Fine-tune with MLX (Mac Apple Silicon)
finetune-mlx: prepare-training
	python scripts/finetune_mlx_mac.py --model mistral --batch-size 2 --num-layers 8 --iters 500

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
# Utilities
# ==================

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
