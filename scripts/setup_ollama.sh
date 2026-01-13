#!/bin/bash
# GSWA Ollama Setup Script for Mac (Apple Silicon)
#
# This script helps Mac users set up Ollama as the LLM backend.
# Ollama runs efficiently on Apple Silicon (M1/M2/M3) chips.
#
# Usage:
#   ./scripts/setup_ollama.sh
#   ./scripts/setup_ollama.sh --model mistral
#   ./scripts/setup_ollama.sh --model llama2

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default model
MODEL=${1:-"mistral"}
if [[ "$1" == "--model" ]]; then
    MODEL=${2:-"mistral"}
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GSWA Ollama Setup for Mac${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on Mac
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${YELLOW}Warning: This script is designed for macOS.${NC}"
    echo -e "${YELLOW}For Linux, you can still use Ollama but installation differs.${NC}"
    echo ""
fi

# Check for Apple Silicon
if [[ "$(uname -m)" == "arm64" ]]; then
    echo -e "${GREEN}Detected: Apple Silicon (M1/M2/M3)${NC}"
else
    echo -e "${YELLOW}Detected: Intel Mac - Ollama will work but may be slower${NC}"
fi
echo ""

# Step 1: Check if Ollama is installed
echo -e "${BLUE}Step 1: Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}Ollama is installed: ${OLLAMA_VERSION}${NC}"
else
    echo -e "${YELLOW}Ollama is not installed.${NC}"
    echo ""
    echo "To install Ollama on Mac:"
    echo ""
    echo "  Option 1: Download from https://ollama.ai/download"
    echo ""
    echo "  Option 2: Using Homebrew:"
    echo "    brew install ollama"
    echo ""
    read -p "Would you like to install via Homebrew now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v brew &> /dev/null; then
            echo "Installing Ollama via Homebrew..."
            brew install ollama
        else
            echo -e "${RED}Homebrew not found. Please install from https://ollama.ai/download${NC}"
            exit 1
        fi
    else
        echo "Please install Ollama manually and run this script again."
        exit 1
    fi
fi
echo ""

# Step 2: Start Ollama service
echo -e "${BLUE}Step 2: Starting Ollama service...${NC}"
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}Ollama service is already running.${NC}"
else
    echo "Starting Ollama service in background..."
    ollama serve &> /dev/null &
    sleep 2

    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}Ollama service started successfully.${NC}"
    else
        echo -e "${YELLOW}Could not start Ollama service automatically.${NC}"
        echo "Please start it manually: ollama serve"
    fi
fi
echo ""

# Step 3: Pull the model
echo -e "${BLUE}Step 3: Pulling model '${MODEL}'...${NC}"
echo "This may take a while for the first download..."
echo ""

if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    echo -e "${GREEN}Model '${MODEL}' is already downloaded.${NC}"
else
    echo "Downloading ${MODEL}..."
    ollama pull ${MODEL}
    echo -e "${GREEN}Model '${MODEL}' downloaded successfully.${NC}"
fi
echo ""

# Step 4: Create .env file
echo -e "${BLUE}Step 4: Configuring GSWA for Ollama...${NC}"

ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
    echo -e "${YELLOW}Found existing .env file.${NC}"
    read -p "Update it for Ollama? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping .env update."
    else
        # Update existing .env
        if grep -q "LLM_BACKEND" "$ENV_FILE"; then
            sed -i '' "s/LLM_BACKEND=.*/LLM_BACKEND=ollama/" "$ENV_FILE"
        else
            echo "LLM_BACKEND=ollama" >> "$ENV_FILE"
        fi

        if grep -q "VLLM_MODEL_NAME" "$ENV_FILE"; then
            sed -i '' "s/VLLM_MODEL_NAME=.*/VLLM_MODEL_NAME=${MODEL}/" "$ENV_FILE"
        else
            echo "VLLM_MODEL_NAME=${MODEL}" >> "$ENV_FILE"
        fi
        echo -e "${GREEN}.env updated for Ollama.${NC}"
    fi
else
    echo "Creating .env file..."
    cat > "$ENV_FILE" << EOF
# GSWA Configuration for Ollama (Mac)
ALLOW_EXTERNAL_API=false
LLM_BACKEND=ollama
VLLM_MODEL_NAME=${MODEL}
EOF
    echo -e "${GREEN}.env file created.${NC}"
fi
echo ""

# Step 5: Test the connection
echo -e "${BLUE}Step 5: Testing Ollama connection...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}Ollama API is responding at http://localhost:11434${NC}"

    # List available models
    echo ""
    echo "Available models:"
    ollama list 2>/dev/null || echo "  (Could not list models)"
else
    echo -e "${YELLOW}Ollama API not responding. Make sure 'ollama serve' is running.${NC}"
fi
echo ""

# Done
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Start GSWA server:"
echo "     make run"
echo ""
echo "  2. Open browser:"
echo "     http://localhost:8080"
echo ""
echo "Recommended models for scientific writing:"
echo "  - mistral (7B, good balance)"
echo "  - llama2 (7B, general purpose)"
echo "  - mixtral (47B, higher quality, needs more RAM)"
echo ""
echo "To switch models:"
echo "  ollama pull <model-name>"
echo "  # Then update VLLM_MODEL_NAME in .env"
echo ""
