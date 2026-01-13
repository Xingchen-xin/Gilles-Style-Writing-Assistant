.PHONY: install dev test lint smoke-test run clean build-index help

# Default target
help:
	@echo "GSWA - Gilles-Style Writing Assistant"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install core dependencies"
	@echo "  dev          - Install development dependencies"
	@echo "  test         - Run unit tests"
	@echo "  lint         - Run linter"
	@echo "  smoke-test   - Run end-to-end smoke test"
	@echo "  run          - Start GSWA server"
	@echo "  build-index  - Build similarity index from corpus"
	@echo "  clean        - Clean build artifacts"

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

# Start GSWA server
run:
	uvicorn gswa.main:app --reload --host 0.0.0.0 --port 8080

# Start GSWA server (production mode)
run-prod:
	uvicorn gswa.main:app --host 0.0.0.0 --port 8080 --workers 4

# Build similarity index from corpus
build-index:
	python scripts/build_index.py

# Start vLLM server
start-vllm:
	bash scripts/start_vllm.sh

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
