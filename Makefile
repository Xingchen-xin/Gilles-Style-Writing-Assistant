.PHONY: install dev test lint smoke-test run clean build-index parse-corpus export-dpo help setup-mac setup-ollama

# Default target
help:
	@echo "GSWA - Gilles-Style Writing Assistant"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  === Installation ==="
	@echo "  install        - Install core dependencies"
	@echo "  dev            - Install development dependencies"
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
	@echo "  === Training Data ==="
	@echo "  export-dpo     - Export feedback data for DPO training"
	@echo ""
	@echo "  === Utilities ==="
	@echo "  clean          - Clean build artifacts"

# Install core dependencies
install:
	pip install -e .

# Install dev dependencies
dev:
	pip install -e ".[dev,similarity]"

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
# Training Data
# ==================

# Export feedback data for DPO training
export-dpo:
	python scripts/export_dpo_data.py

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
