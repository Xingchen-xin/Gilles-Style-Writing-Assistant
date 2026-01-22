#!/bin/bash
# GSWA One-Click Setup Script
# 傻瓜式一键环境配置脚本
#
# Automatically detects and installs Python 3.10+ if needed,
# creates a virtual environment, and installs all dependencies.
#
# Supports environments WITHOUT sudo (servers, shared machines)
# by using micromamba/conda for bundled Python installation.
#
# Usage:
#   ./scripts/setup.sh              # Interactive setup
#   ./scripts/setup.sh --auto       # Automatic (no prompts)
#   ./scripts/setup.sh --dev        # Install dev dependencies too
#   ./scripts/setup.sh --all        # Install all optional dependencies
#   ./scripts/setup.sh --cuda       # Install PyTorch with CUDA support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=10
PREFERRED_PYTHON_VERSION="3.11"
CONDA_ENV_NAME="gswa"
VENV_DIR="venv"
AUTO_MODE=false
INSTALL_DEV=false
INSTALL_ALL=false
INSTALL_CUDA=false
USE_CONDA=false

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --auto|-y)
            AUTO_MODE=true
            ;;
        --dev)
            INSTALL_DEV=true
            ;;
        --all)
            INSTALL_ALL=true
            INSTALL_DEV=true
            ;;
        --cuda)
            INSTALL_CUDA=true
            ;;
        --help|-h)
            echo "GSWA Setup Script"
            echo ""
            echo "Usage: ./scripts/setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --auto, -y    Automatic mode (no prompts)"
            echo "  --dev         Install development dependencies"
            echo "  --all         Install all optional dependencies"
            echo "  --cuda        Install PyTorch with CUDA support (for NVIDIA GPUs)"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "This script works WITHOUT sudo by using micromamba/conda for"
            echo "Python installation if Python 3.10+ is not found."
            exit 0
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GSWA One-Click Setup${NC}"
echo -e "${BLUE}  傻瓜式一键环境配置${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect OS
OS="unknown"
if [[ "$(uname)" == "Darwin" ]]; then
    OS="mac"
    echo -e "${GREEN}Detected: macOS${NC}"
elif [[ "$(uname)" == "Linux" ]]; then
    OS="linux"
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        echo -e "${GREEN}Detected: Linux ($NAME)${NC}"
    else
        DISTRO="unknown"
        echo -e "${GREEN}Detected: Linux${NC}"
    fi
else
    echo -e "${YELLOW}Detected: $(uname) (may not be fully supported)${NC}"
fi

# Detect GPU
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo -e "${GREEN}Detected: NVIDIA GPU ($GPU_NAME)${NC}"
        if ! $INSTALL_CUDA; then
            echo -e "${YELLOW}Tip: Use --cuda flag to install PyTorch with CUDA support${NC}"
        fi
    fi
fi
echo ""

# Function to find Python 3.10+
find_python() {
    local python_cmd=""
    local python_version=""

    # Also check pyenv shims if pyenv is installed
    if command -v pyenv &> /dev/null; then
        eval "$(pyenv init -)" 2>/dev/null || true
    fi

    # Try different Python commands in order of preference
    for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            version=$("$cmd" --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+' || echo "0.0")
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)

            if [[ "$major" -ge $REQUIRED_PYTHON_MAJOR ]] && [[ "$minor" -ge $REQUIRED_PYTHON_MINOR ]]; then
                # Check if ctypes module works (important for PyTorch)
                if "$cmd" -c "import ctypes" 2>/dev/null; then
                    python_cmd="$cmd"
                    python_version="$version"
                    break
                fi
            fi
        fi
    done

    echo "$python_cmd|$python_version"
}

# Function to check for conda/micromamba
find_conda() {
    if command -v micromamba &> /dev/null; then
        echo "micromamba"
    elif command -v mamba &> /dev/null; then
        echo "mamba"
    elif command -v conda &> /dev/null; then
        echo "conda"
    else
        echo ""
    fi
}

# Function to install micromamba
install_micromamba() {
    echo -e "${BLUE}Installing micromamba (lightweight conda)...${NC}"
    echo ""

    # Install micromamba using the official installer (no sudo needed)
    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

    # Initialize for current shell
    export MAMBA_ROOT_PREFIX="$HOME/micromamba"
    eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"

    echo ""
    echo -e "${GREEN}micromamba installed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}NOTE: Add to your ~/.bashrc or ~/.zshrc:${NC}"
    echo ""
    echo '  export MAMBA_ROOT_PREFIX="$HOME/micromamba"'
    echo '  eval "$(micromamba shell hook -s bash)"'
    echo ""
}

# Function to create conda environment
create_conda_env() {
    local conda_cmd=$1

    echo -e "${BLUE}Creating conda environment '$CONDA_ENV_NAME' with Python $PREFERRED_PYTHON_VERSION...${NC}"
    echo ""

    $conda_cmd create -n "$CONDA_ENV_NAME" python="$PREFERRED_PYTHON_VERSION" -y

    echo ""
    echo -e "${GREEN}Conda environment '$CONDA_ENV_NAME' created!${NC}"
}

# Step 1: Find or install Python 3.10+
echo -e "${BLUE}Step 1: Checking Python version...${NC}"

result=$(find_python)
PYTHON_CMD=$(echo "$result" | cut -d'|' -f1)
PYTHON_VERSION=$(echo "$result" | cut -d'|' -f2)

# Check for existing conda environment
CONDA_CMD=$(find_conda)
if [[ -n "$CONDA_CMD" ]]; then
    # Check if gswa environment exists (multiple methods)
    ENV_EXISTS=false

    # Method 1: Check env list
    if $CONDA_CMD env list 2>/dev/null | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
        ENV_EXISTS=true
    fi

    # Method 2: Check direct path (for micromamba)
    if [[ -d "$HOME/micromamba/envs/$CONDA_ENV_NAME" ]]; then
        ENV_EXISTS=true
    fi

    # Method 3: Check conda envs path
    if [[ -d "$HOME/.conda/envs/$CONDA_ENV_NAME" ]]; then
        ENV_EXISTS=true
    fi

    # Method 4: Check public_data conda envs (shared server)
    if [[ -d "/home/public_data/conda_envs/$CONDA_ENV_NAME" ]]; then
        ENV_EXISTS=true
    fi

    if $ENV_EXISTS; then
        echo -e "${GREEN}Found existing conda environment: $CONDA_ENV_NAME${NC}"
        USE_CONDA=true
        PYTHON_CMD="$CONDA_CMD run -n $CONDA_ENV_NAME python"
        PYTHON_VERSION=$($CONDA_CMD run -n $CONDA_ENV_NAME python --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+' || echo "3.11")
    fi
fi

if [[ -n "$PYTHON_CMD" ]] && ! $USE_CONDA; then
    echo -e "${GREEN}Found: $PYTHON_CMD (Python $PYTHON_VERSION)${NC}"
elif ! $USE_CONDA; then
    echo -e "${YELLOW}Python 3.10+ not found on this system.${NC}"
    echo ""

    # Check for conda/micromamba first
    CONDA_CMD=$(find_conda)

    if [[ -n "$CONDA_CMD" ]]; then
        echo -e "${GREEN}Found: $CONDA_CMD${NC}"
        echo ""
        if $AUTO_MODE; then
            REPLY="1"
        else
            echo "How would you like to proceed?"
            echo "  1) Create conda environment with Python $PREFERRED_PYTHON_VERSION (recommended)"
            echo "  2) Exit and install manually"
            echo ""
            read -p "Choice [1/2]: " REPLY
        fi

        if [[ "$REPLY" == "1" ]]; then
            create_conda_env "$CONDA_CMD"
            USE_CONDA=true
            PYTHON_CMD="$CONDA_CMD run -n $CONDA_ENV_NAME python"
            PYTHON_VERSION="$PREFERRED_PYTHON_VERSION"
        else
            echo ""
            echo "Please install Python 3.10+ and run this script again."
            exit 1
        fi
    else
        echo "Options for installing Python 3.10+ (no sudo needed):"
        echo ""
        echo "  1. micromamba (recommended for Linux) - Lightweight conda"
        echo "     This script can install it automatically."
        echo ""
        echo "  2. pyenv (for Mac) - Local Python version manager"
        echo "     Note: May require system libraries (libffi-devel) on Linux"
        echo ""
        echo "  3. Exit and install manually"
        echo ""

        if $AUTO_MODE; then
            if [[ "$OS" == "linux" ]]; then
                REPLY="1"
            else
                REPLY="1"
            fi
        else
            read -p "Choice [1/2/3]: " REPLY
        fi

        if [[ "$REPLY" == "1" ]]; then
            install_micromamba
            CONDA_CMD="micromamba"
            create_conda_env "$CONDA_CMD"
            USE_CONDA=true
            PYTHON_CMD="$CONDA_CMD run -n $CONDA_ENV_NAME python"
            PYTHON_VERSION="$PREFERRED_PYTHON_VERSION"
        elif [[ "$REPLY" == "2" ]]; then
            echo ""
            echo "For pyenv installation:"
            echo "  curl https://pyenv.run | bash"
            echo "  pyenv install 3.11.9"
            echo ""
            echo "Note: On Linux servers, pyenv may fail if libffi-devel is not installed."
            echo "In that case, use micromamba instead (option 1)."
            exit 1
        else
            echo ""
            echo "Please install Python 3.10+ and run this script again."
            echo ""
            echo "Recommended options:"
            echo "  - micromamba: curl -L micro.mamba.pm/install.sh | bash"
            echo "  - conda: Download from https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
    fi
fi
echo ""

# Step 2: Setup environment
echo -e "${BLUE}Step 2: Setting up environment...${NC}"

if $USE_CONDA; then
    echo -e "${GREEN}Using conda environment: $CONDA_ENV_NAME${NC}"

    # Install dependencies using conda run
    echo ""
    echo -e "${BLUE}Step 3: Upgrading pip...${NC}"
    $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install --upgrade pip --quiet
    echo -e "${GREEN}pip upgraded${NC}"
    echo ""

    # Step 4: Install PyTorch (with CUDA if requested)
    if $INSTALL_CUDA && $HAS_NVIDIA_GPU; then
        echo -e "${BLUE}Step 4: Installing PyTorch with CUDA support...${NC}"
        $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        echo -e "${GREEN}PyTorch with CUDA installed!${NC}"

        # Install DeepSpeed and Accelerate for multi-GPU training
        echo -e "${BLUE}Installing DeepSpeed for multi-GPU training...${NC}"
        $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install deepspeed accelerate pyyaml
        echo -e "${GREEN}DeepSpeed installed!${NC}"
    elif $HAS_NVIDIA_GPU; then
        echo -e "${BLUE}Step 4: Installing PyTorch (CPU version)...${NC}"
        echo -e "${YELLOW}Use --cuda flag to install with GPU support${NC}"
        $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install torch torchvision
        echo -e "${GREEN}PyTorch installed!${NC}"
    fi
    echo ""

    # Step 5: Install project dependencies
    echo -e "${BLUE}Step 5: Installing project dependencies...${NC}"

    if $INSTALL_ALL; then
        echo "Installing all dependencies (core + dev + similarity + train)..."
        $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install -e ".[dev,similarity,train]" pymupdf
    elif $INSTALL_CUDA; then
        echo "Installing core + dev + similarity + train dependencies..."
        $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install -e ".[dev,similarity,train]" pymupdf
    elif $INSTALL_DEV; then
        echo "Installing core + dev + similarity dependencies..."
        $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install -e ".[dev,similarity]" pymupdf
    else
        echo "Installing core dependencies..."
        $CONDA_CMD run -n "$CONDA_ENV_NAME" pip install -e "." pymupdf
    fi

    echo -e "${GREEN}Dependencies installed successfully!${NC}"
    echo ""

    # Step 6: Verify installation
    echo -e "${BLUE}Step 6: Verifying installation...${NC}"

    venv_python_version=$($CONDA_CMD run -n "$CONDA_ENV_NAME" python --version)
    echo -e "  Python: ${GREEN}$venv_python_version${NC}"

    if $CONDA_CMD run -n "$CONDA_ENV_NAME" pip show gswa &> /dev/null; then
        echo -e "  GSWA: ${GREEN}installed${NC}"
    else
        echo -e "  GSWA: ${RED}not installed${NC}"
    fi

    if $CONDA_CMD run -n "$CONDA_ENV_NAME" pip show torch &> /dev/null; then
        cuda_check=$($CONDA_CMD run -n "$CONDA_ENV_NAME" python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "CPU")
        torch_version=$($CONDA_CMD run -n "$CONDA_ENV_NAME" pip show torch | grep Version | cut -d' ' -f2)
        echo -e "  PyTorch: ${GREEN}$torch_version ($cuda_check)${NC}"
    else
        echo -e "  PyTorch: ${YELLOW}not installed${NC}"
    fi

    if $CONDA_CMD run -n "$CONDA_ENV_NAME" pip show deepspeed &> /dev/null; then
        ds_version=$($CONDA_CMD run -n "$CONDA_ENV_NAME" pip show deepspeed | grep Version | cut -d' ' -f2)
        echo -e "  DeepSpeed: ${GREEN}$ds_version${NC}"
    else
        echo -e "  DeepSpeed: ${YELLOW}not installed (for multi-GPU 70B+ training)${NC}"
    fi

    echo ""

    # Done!
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Setup Complete! 设置完成!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the conda environment:"
    echo "     ${CYAN}$CONDA_CMD activate $CONDA_ENV_NAME${NC}"
    echo ""
    echo "  2. Or run commands directly:"
    echo "     ${CYAN}$CONDA_CMD run -n $CONDA_ENV_NAME make test${NC}"
    echo ""
    echo "  3. Start the server:"
    echo "     ${CYAN}$CONDA_CMD run -n $CONDA_ENV_NAME make run${NC}"
    echo ""
    echo "  4. Open in browser:"
    echo "     ${CYAN}http://localhost:8080${NC}"
    echo ""

    if $HAS_NVIDIA_GPU; then
        echo "For training with GPU:"
        echo "  ${CYAN}$CONDA_CMD run -n $CONDA_ENV_NAME make finetune-smart${NC}"
        echo ""
    fi

else
    # Original venv-based setup
    if [[ -d "$VENV_DIR" ]]; then
        if [[ -f "$VENV_DIR/bin/python" ]]; then
            existing_version=$("$VENV_DIR/bin/python" --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+' || echo "0.0")
            existing_major=$(echo "$existing_version" | cut -d. -f1)
            existing_minor=$(echo "$existing_version" | cut -d. -f2)

            if [[ "$existing_major" -ge $REQUIRED_PYTHON_MAJOR ]] && [[ "$existing_minor" -ge $REQUIRED_PYTHON_MINOR ]]; then
                echo -e "${GREEN}Existing venv found with Python $existing_version${NC}"
            else
                echo -e "${YELLOW}Existing venv has Python $existing_version (need 3.10+)${NC}"

                if $AUTO_MODE; then
                    REPLY="y"
                else
                    read -p "Replace with Python $PYTHON_VERSION? (y/n) " -n 1 -r
                    echo ""
                fi

                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "Removing old venv..."
                    rm -rf "$VENV_DIR"
                    echo "Creating new venv with Python $PYTHON_VERSION..."
                    "$PYTHON_CMD" -m venv "$VENV_DIR"
                else
                    echo "Keeping existing venv. Some features may not work correctly."
                fi
            fi
        else
            echo -e "${YELLOW}Corrupted venv found, recreating...${NC}"
            rm -rf "$VENV_DIR"
            "$PYTHON_CMD" -m venv "$VENV_DIR"
        fi
    else
        echo "Creating virtual environment with Python $PYTHON_VERSION..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi

    # Activate venv for subsequent commands
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}Virtual environment ready: $VENV_DIR${NC}"
    echo ""

    # Step 3: Upgrade pip
    echo -e "${BLUE}Step 3: Upgrading pip...${NC}"
    pip install --upgrade pip --quiet
    echo -e "${GREEN}pip upgraded to $(pip --version | cut -d' ' -f2)${NC}"
    echo ""

    # Step 4: Install dependencies
    echo -e "${BLUE}Step 4: Installing dependencies...${NC}"

    if $INSTALL_ALL; then
        echo "Installing all dependencies (core + dev + similarity)..."
        pip install -e ".[dev,similarity]"
    elif $INSTALL_DEV; then
        echo "Installing core + dev dependencies..."
        pip install -e ".[dev,similarity]"
    else
        echo "Installing core dependencies..."
        pip install -e "."
    fi

    echo -e "${GREEN}Dependencies installed successfully!${NC}"
    echo ""

    # Step 5: Verify installation
    echo -e "${BLUE}Step 5: Verifying installation...${NC}"

    venv_python_version=$("$VENV_DIR/bin/python" --version)
    echo -e "  Python: ${GREEN}$venv_python_version${NC}"

    if pip show gswa &> /dev/null; then
        echo -e "  GSWA: ${GREEN}installed${NC}"
    else
        echo -e "  GSWA: ${RED}not installed${NC}"
    fi

    if pip show fastapi &> /dev/null; then
        echo -e "  FastAPI: ${GREEN}installed${NC}"
    else
        echo -e "  FastAPI: ${RED}not installed${NC}"
    fi

    if pip show torch &> /dev/null; then
        torch_version=$(pip show torch | grep Version | cut -d' ' -f2)
        echo -e "  PyTorch: ${GREEN}$torch_version${NC}"
    else
        echo -e "  PyTorch: ${YELLOW}not installed (install with: pip install torch)${NC}"
    fi

    echo ""

    # Done!
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Setup Complete! 设置完成!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo "     ${CYAN}source venv/bin/activate${NC}"
    echo ""
    echo "  2. Run tests to verify:"
    echo "     ${CYAN}make test${NC}"
    echo ""
    echo "  3. Start the server:"
    echo "     ${CYAN}make run${NC}"
    echo ""
    echo "  4. Open in browser:"
    echo "     ${CYAN}http://localhost:8080${NC}"
    echo ""

    # Show additional setup options
    if [[ "$OS" == "mac" ]]; then
        echo "For Mac users with Ollama:"
        echo "  ${CYAN}make setup-ollama${NC}"
        echo ""
    elif [[ "$OS" == "linux" ]]; then
        echo "For Linux users with NVIDIA GPU:"
        echo "  ${CYAN}make start-vllm${NC}"
        echo ""
    fi

    echo "For training your own model:"
    echo "  ${CYAN}make train${NC}"
    echo ""
fi
