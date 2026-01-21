#!/usr/bin/env python3
"""
GSWA Smart Fine-tuning System
=============================

Unified entry point for fine-tuning on any platform:
- Mac (Apple Silicon): MLX
- Linux (NVIDIA GPU): LoRA/QLoRA with PyTorch
- Windows (NVIDIA GPU): LoRA/QLoRA with PyTorch

Auto-detects:
- Operating system
- Available hardware (GPU type, memory)
- Optimal base model for your hardware
- Best training parameters

Usage:
    python scripts/smart_finetune.py              # Full auto mode
    python scripts/smart_finetune.py --info       # Show system info only
    python scripts/smart_finetune.py --model llama3  # Force specific model
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ==============================================================================
# Constants
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
PROFILES_PATH = CONFIG_DIR / "training_profiles.json"

# Base model recommendations based on task and hardware
# For English scientific writing, Llama 3.3 is best but requires HuggingFace login
# Mistral models are UNGATED fallback options
MODEL_RECOMMENDATIONS = {
    "scientific_writing": {
        # Format: (min_vram_gb, model_id, description)
        "tier_0": {
            "min_vram": 60,
            "models": {
                "mlx": "mlx-community/Llama-3.3-70B-Instruct-4bit",
                "cuda": "meta-llama/Llama-3.3-70B-Instruct",
            },
            "description": "Llama 3.3 70B - Best for English academic writing (requires HF login)",
        },
        "tier_1": {
            "min_vram": 48,
            "models": {
                "mlx": "mlx-community/Mistral-Large-Instruct-2407-4bit",
                "cuda": "mistralai/Mistral-Large-Instruct-2407",
            },
            "description": "Mistral Large - Excellent for English scientific writing",
        },
        "tier_2": {
            "min_vram": 24,
            "models": {
                "mlx": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
                "cuda": "mistralai/Mistral-Nemo-Instruct-2407",
            },
            "description": "Mistral Nemo 12B - Strong English, good for academic style",
        },
        "tier_3": {
            "min_vram": 16,
            "models": {
                "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "cuda": "mistralai/Mistral-7B-Instruct-v0.3",
            },
            "description": "Mistral 7B - Good balance for English writing",
        },
        "tier_4": {
            "min_vram": 8,
            "models": {
                "mlx": "mlx-community/Phi-3.5-mini-instruct-4bit",
                "cuda": "microsoft/Phi-3.5-mini-instruct",
            },
            "description": "Phi-3.5 Mini - Fast training, decent English quality",
        },
        "tier_5": {
            "min_vram": 4,
            "models": {
                "mlx": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "cuda": "Qwen/Qwen2.5-1.5B-Instruct",
            },
            "description": "Qwen2.5 1.5B - Minimum viable, limited quality",
        },
    }
}


@dataclass
class SystemInfo:
    """System hardware information."""
    os_name: str
    os_version: str
    cpu: str
    cpu_cores: int
    ram_gb: float
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_type: Optional[str] = None  # "apple", "nvidia", "amd", "intel", None
    cuda_available: bool = False
    mlx_available: bool = False
    recommended_backend: Optional[str] = None
    recommended_model_tier: Optional[str] = None


# ==============================================================================
# System Detection
# ==============================================================================

def get_system_info() -> SystemInfo:
    """Detect system hardware and capabilities."""
    info = SystemInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        cpu=platform.processor() or "Unknown",
        cpu_cores=os.cpu_count() or 1,
        ram_gb=get_system_ram_gb(),
    )

    # Get detailed CPU info
    if info.os_name == "Darwin":
        info.cpu = get_mac_cpu_info()
    elif info.os_name == "Linux":
        info.cpu = get_linux_cpu_info()
    elif info.os_name == "Windows":
        info.cpu = get_windows_cpu_info()

    # Detect GPU
    detect_gpu(info)

    # Determine recommended backend
    determine_backend(info)

    return info


def get_system_ram_gb() -> float:
    """Get system RAM in GB."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            return int(result.stdout.strip()) / (1024 ** 3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong),
                ]
            memoryStatus = MEMORYSTATUS()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
            return memoryStatus.dwTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 16.0  # Default assumption


def get_mac_cpu_info() -> str:
    """Get Mac CPU info."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "Apple Silicon"


def get_linux_cpu_info() -> str:
    """Get Linux CPU info."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def get_windows_cpu_info() -> str:
    """Get Windows CPU info."""
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True, text=True, shell=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            return lines[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def detect_gpu(info: SystemInfo):
    """Detect GPU type and memory."""
    # Check for Apple Silicon GPU (unified memory)
    if info.os_name == "Darwin" and "Apple" in info.cpu:
        info.gpu_type = "apple"
        info.gpu_name = info.cpu
        # Apple Silicon uses unified memory, so GPU memory = RAM
        info.gpu_vram_gb = info.ram_gb

        # Check if MLX is available
        try:
            import mlx
            info.mlx_available = True
        except ImportError:
            info.mlx_available = False
        return

    # Check for NVIDIA GPU
    nvidia_info = detect_nvidia_gpu()
    if nvidia_info:
        info.gpu_type = "nvidia"
        info.gpu_name = nvidia_info["name"]
        info.gpu_vram_gb = nvidia_info["vram_gb"]
        info.cuda_available = nvidia_info["cuda_available"]
        return

    # Check for AMD GPU (ROCm on Linux)
    if info.os_name == "Linux":
        amd_info = detect_amd_gpu()
        if amd_info:
            info.gpu_type = "amd"
            info.gpu_name = amd_info["name"]
            info.gpu_vram_gb = amd_info["vram_gb"]
            return

    # No GPU detected
    info.gpu_type = None
    info.gpu_name = None
    info.gpu_vram_gb = None


def detect_nvidia_gpu() -> Optional[dict]:
    """Detect NVIDIA GPU using nvidia-smi. Supports multi-GPU setups."""
    try:
        # Get GPU name and memory for ALL GPUs
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_count = len(lines)
            total_vram_mb = 0
            gpu_names = []

            for line in lines:
                parts = line.split(",")
                name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                gpu_names.append(name)
                total_vram_mb += vram_mb

            # Use first GPU name, but indicate multi-GPU if applicable
            if gpu_count > 1:
                display_name = f"{gpu_names[0]} x{gpu_count}"
            else:
                display_name = gpu_names[0]

            # Check CUDA availability
            cuda_available = False
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                pass

            return {
                "name": display_name,
                "vram_gb": total_vram_mb / 1024,  # Total VRAM across all GPUs
                "gpu_count": gpu_count,
                "cuda_available": cuda_available,
            }
    except Exception:
        pass
    return None


def detect_amd_gpu() -> Optional[dict]:
    """Detect AMD GPU using rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Parse output (format varies)
            lines = result.stdout.strip().split("\n")
            name = "AMD GPU"
            vram_gb = 8.0  # Default

            for line in lines:
                if "GPU" in line and ":" in line:
                    name = line.split(":")[-1].strip()
                if "Total" in line and "vram" in line.lower():
                    # Extract memory value
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.isdigit() or p.replace(".", "").isdigit():
                            vram_gb = float(p) / 1024  # Usually in MB
                            break

            return {"name": name, "vram_gb": vram_gb}
    except Exception:
        pass
    return None


def determine_backend(info: SystemInfo):
    """Determine the recommended training backend."""
    if info.gpu_type == "apple" and info.mlx_available:
        info.recommended_backend = "mlx"
    elif info.gpu_type == "nvidia" and info.cuda_available:
        info.recommended_backend = "cuda"
    elif info.gpu_type == "amd":
        info.recommended_backend = "rocm"
    else:
        # CPU-only fallback
        info.recommended_backend = "cpu"

    # Determine model tier based on available VRAM (total across all GPUs)
    vram = info.gpu_vram_gb or info.ram_gb * 0.5  # Use half RAM for CPU

    for tier_name in ["tier_0", "tier_1", "tier_2", "tier_3", "tier_4", "tier_5"]:
        tier = MODEL_RECOMMENDATIONS["scientific_writing"][tier_name]
        if vram >= tier["min_vram"]:
            info.recommended_model_tier = tier_name
            break
    else:
        info.recommended_model_tier = "tier_5"


# ==============================================================================
# Training Configuration
# ==============================================================================

def get_training_params(info: SystemInfo, model_tier: str) -> dict:
    """Get training parameters based on system and model."""
    vram = info.gpu_vram_gb or info.ram_gb * 0.5

    # Base parameters
    params = {
        "batch_size": 1,
        "num_layers": 4,
        "iters": 300,
        "max_seq_length": 512,
        "learning_rate": 1e-5,
        "gradient_accumulation": 4,
    }

    # Adjust based on VRAM
    # Note: For 70B models, even with 64GB VRAM, batch_size must be small
    # The model itself uses ~35-40GB, leaving limited room for activations
    if vram >= 60:  # Use 60 instead of 64 (32760MB * 2 / 1024 = 63.98)
        params.update({
            "batch_size": 1,  # 70B model needs batch_size=1
            "num_layers": 16,
            "iters": 1000,
            "max_seq_length": 1024,  # Reduced for memory
            "gradient_accumulation": 8,  # Effective batch = 8
        })
    elif vram >= 32:
        params.update({
            "batch_size": 2,
            "num_layers": 16,
            "iters": 1000,
            "max_seq_length": 1536,
            "gradient_accumulation": 4,
        })
    elif vram >= 16:
        params.update({
            "batch_size": 2,
            "num_layers": 8,
            "iters": 500,
            "max_seq_length": 1024,
            "gradient_accumulation": 4,
        })
    elif vram >= 8:
        params.update({
            "batch_size": 1,
            "num_layers": 4,
            "iters": 400,
            "max_seq_length": 768,
            "gradient_accumulation": 8,
        })

    return params


def get_recommended_model(info: SystemInfo) -> dict:
    """Get the recommended model for the system."""
    tier_name = info.recommended_model_tier or "tier_4"
    tier = MODEL_RECOMMENDATIONS["scientific_writing"][tier_name]

    backend = info.recommended_backend
    if backend == "mlx":
        model_key = "mlx"
    else:
        model_key = "cuda"  # Works for CUDA, ROCm, and CPU

    return {
        "tier": tier_name,
        "model_id": tier["models"].get(model_key) or tier["models"].get("cuda"),
        "description": tier["description"],
        "alternatives": {
            k: v for k, v in tier["models"].items()
            if k != model_key
        }
    }


# ==============================================================================
# Display Functions
# ==============================================================================

def print_system_info(info: SystemInfo):
    """Print system information."""
    print("\n" + "=" * 70)
    print("GSWA Smart Fine-tuning System - System Information")
    print("=" * 70)

    print(f"\n{'Operating System:':<25} {info.os_name} ({info.os_version[:50]}...)")
    print(f"{'CPU:':<25} {info.cpu}")
    print(f"{'CPU Cores:':<25} {info.cpu_cores}")
    print(f"{'System RAM:':<25} {info.ram_gb:.1f} GB")

    print("\n" + "-" * 70)
    print("GPU Information")
    print("-" * 70)

    if info.gpu_type:
        print(f"{'GPU Type:':<25} {info.gpu_type.upper()}")
        print(f"{'GPU Name:':<25} {info.gpu_name}")
        print(f"{'GPU VRAM:':<25} {info.gpu_vram_gb:.1f} GB")

        if info.gpu_type == "apple":
            print(f"{'MLX Available:':<25} {'Yes' if info.mlx_available else 'No (pip install mlx mlx-lm)'}")
        elif info.gpu_type == "nvidia":
            print(f"{'CUDA Available:':<25} {'Yes' if info.cuda_available else 'No (install PyTorch with CUDA)'}")
    else:
        print("No GPU detected - will use CPU (slow)")

    print("\n" + "-" * 70)
    print("Recommended Configuration")
    print("-" * 70)

    print(f"{'Training Backend:':<25} {info.recommended_backend.upper()}")
    print(f"{'Model Tier:':<25} {info.recommended_model_tier}")

    model_info = get_recommended_model(info)
    print(f"{'Recommended Model:':<25} {model_info['description']}")
    print(f"{'Model ID:':<25} {model_info['model_id']}")

    params = get_training_params(info, info.recommended_model_tier)
    print(f"\n{'Training Parameters:'}")
    print(f"  {'batch_size:':<20} {params['batch_size']}")
    print(f"  {'num_layers:':<20} {params['num_layers']}")
    print(f"  {'iters:':<20} {params['iters']}")
    print(f"  {'max_seq_length:':<20} {params['max_seq_length']}")
    print(f"  {'learning_rate:':<20} {params['learning_rate']}")


def print_model_options():
    """Print available model options."""
    print("\n" + "=" * 70)
    print("Available Base Models for Scientific Writing")
    print("=" * 70)

    for tier_name, tier in MODEL_RECOMMENDATIONS["scientific_writing"].items():
        print(f"\n{tier_name.upper()} (Min VRAM: {tier['min_vram']} GB)")
        print(f"  {tier['description']}")
        print("  Models:")
        for backend, model_id in tier["models"].items():
            if not backend.startswith("alt_"):
                print(f"    {backend:<8}: {model_id}")


# ==============================================================================
# Corpus Management
# ==============================================================================

def check_corpus() -> dict:
    """Check corpus status and return statistics."""
    corpus_dir = PROJECT_ROOT / "data" / "corpus"
    raw_dir = corpus_dir / "raw"
    priority_dir = raw_dir / "important_examples"

    stats = {
        "raw_dir_exists": raw_dir.exists(),
        "priority_dir_exists": priority_dir.exists(),
        "regular_files": [],
        "priority_files": [],
        "supported_formats": [".pdf", ".docx", ".txt", ".md"],
        "warnings": [],
    }

    if raw_dir.exists():
        for f in raw_dir.iterdir():
            if f.is_file() and f.suffix.lower() in stats["supported_formats"]:
                stats["regular_files"].append({
                    "name": f.name,
                    "size_kb": f.stat().st_size / 1024,
                    "type": f.suffix.lower(),
                })

    if priority_dir.exists():
        for f in priority_dir.iterdir():
            if f.is_file() and f.suffix.lower() in stats["supported_formats"]:
                stats["priority_files"].append({
                    "name": f.name,
                    "size_kb": f.stat().st_size / 1024,
                    "type": f.suffix.lower(),
                })

    # Generate warnings
    total = len(stats["regular_files"]) + len(stats["priority_files"])
    if total == 0:
        stats["warnings"].append("No corpus files found! Add files to data/corpus/raw/")
    elif total < 5:
        stats["warnings"].append(f"Only {total} files found. More files = better results.")

    if len(stats["priority_files"]) == 0 and total > 0:
        stats["warnings"].append("Consider adding important examples to raw/important_examples/ for higher weight.")

    return stats


def print_corpus_status():
    """Print corpus status."""
    stats = check_corpus()

    print("\n" + "=" * 70)
    print("Corpus Status")
    print("=" * 70)

    print(f"\n{'Directory:':<25} data/corpus/raw/")
    print(f"{'Regular files:':<25} {len(stats['regular_files'])}")

    if stats["regular_files"]:
        for f in stats["regular_files"][:5]:
            print(f"  - {f['name']} ({f['size_kb']:.1f} KB)")
        if len(stats["regular_files"]) > 5:
            print(f"  ... and {len(stats['regular_files']) - 5} more")

    print(f"\n{'Priority directory:':<25} data/corpus/raw/important_examples/")
    print(f"{'Priority files:':<25} {len(stats['priority_files'])} (2.5x weight)")

    if stats["priority_files"]:
        for f in stats["priority_files"]:
            print(f"  - {f['name']} ({f['size_kb']:.1f} KB) ⭐")

    total = len(stats["regular_files"]) + len(stats["priority_files"])
    print(f"\n{'Total corpus files:':<25} {total}")

    if stats["warnings"]:
        print("\n" + "-" * 70)
        print("Warnings")
        print("-" * 70)
        for w in stats["warnings"]:
            print(f"  ⚠️  {w}")

    print("\n" + "-" * 70)
    print("Supported Formats")
    print("-" * 70)
    print("  .pdf  - PDF documents (recommended)")
    print("  .docx - Microsoft Word documents")
    print("  .txt  - Plain text files")
    print("  .md   - Markdown files")


# ==============================================================================
# Training Execution
# ==============================================================================

def run_training(info: SystemInfo, args):
    """Run the appropriate training script based on system."""
    backend = info.recommended_backend

    # Check corpus first
    corpus_stats = check_corpus()
    total_files = len(corpus_stats["regular_files"]) + len(corpus_stats["priority_files"])

    if total_files == 0:
        print("\n" + "=" * 70)
        print("ERROR: No Corpus Files Found!")
        print("=" * 70)
        print("\nPlease add files to train on:")
        print("  1. Put PDF/DOCX/TXT files in: data/corpus/raw/")
        print("  2. Put important examples in: data/corpus/raw/important_examples/")
        print("\nThen run this script again.")
        return False

    print_corpus_status()

    # Get model and parameters
    model_info = get_recommended_model(info)
    params = get_training_params(info, info.recommended_model_tier)

    # Override with user-specified model if provided
    if args.model:
        # Map short names to full model IDs
        # NOTE: Llama/Gemma models are GATED - only use if you have HF access
        model_aliases = {
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
            "mistral-large": "mistralai/Mistral-Large-Instruct-2407",
            "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
            "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
            "phi": "microsoft/Phi-3.5-mini-instruct",
            # Gated models (require HuggingFace login + approval):
            "llama3": "meta-llama/Llama-3.1-8B-Instruct",
            "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
            "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
            "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
            "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",  # older 3.1 version
            "gemma": "google/gemma-2-9b-it",
        }
        model_id = model_aliases.get(args.model.lower(), args.model)
    else:
        model_id = model_info["model_id"]

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"\nBackend: {backend.upper()}")
    print(f"Model: {model_id}")
    print(f"Parameters: batch_size={params['batch_size']}, layers={params['num_layers']}, iters={params['iters']}")

    # Run appropriate training script
    if backend == "mlx":
        return run_mlx_training(model_id, params, args)
    elif backend in ["cuda", "rocm", "cpu"]:
        return run_lora_training(model_id, params, args, backend)
    else:
        print(f"\nERROR: Unsupported backend: {backend}")
        return False


def run_mlx_training(model_id: str, params: dict, args) -> bool:
    """Run MLX training on Mac."""
    script = SCRIPT_DIR / "finetune_mlx_mac.py"

    cmd = [
        sys.executable, str(script),
        "--model", model_id,
        "--batch-size", str(params["batch_size"]),
        "--num-layers", str(params["num_layers"]),
        "--iters", str(params["iters"]),
        "--max-seq-length", str(params["max_seq_length"]),
        "--learning-rate", str(params["learning_rate"]),
    ]

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return True

    print(f"\nExecuting: {' '.join(cmd[:4])}...")

    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return False


def run_lora_training(model_id: str, params: dict, args, backend: str) -> bool:
    """Run LoRA/QLoRA training on Linux/Windows."""
    script = SCRIPT_DIR / "finetune_lora.py"

    # Determine quantization based on VRAM
    quantize = "4bit" if backend == "cuda" else "8bit"

    cmd = [
        sys.executable, str(script),
        "--model", model_id,
        "--batch-size", str(params["batch_size"]),
        "--epochs", str(max(1, params["iters"] // 200)),  # Convert iters to epochs
        "--max-length", str(params["max_seq_length"]),
        "--learning-rate", str(params["learning_rate"]),
        "--quantize", quantize,
        "--gradient-accumulation", str(params["gradient_accumulation"]),
    ]

    if backend == "cpu":
        cmd.append("--cpu")

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return True

    print(f"\nExecuting: {' '.join(cmd[:4])}...")

    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return False


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GSWA Smart Fine-tuning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect everything and train
  python scripts/smart_finetune.py

  # Show system info only (no training)
  python scripts/smart_finetune.py --info

  # Show corpus status
  python scripts/smart_finetune.py --corpus

  # Use specific model
  python scripts/smart_finetune.py --model qwen

  # Dry run (show what would be executed)
  python scripts/smart_finetune.py --dry-run

Available model shortcuts (ungated, no login needed):
  mistral, mistral-large, mistral-nemo, qwen, qwen-14b, phi

Gated models (require HuggingFace login + Meta approval):
  llama3.3      - Llama 3.3 70B (RECOMMENDED for 60GB+ VRAM)
  llama3-8b     - Llama 3.1 8B (fast testing)
  llama3-70b    - Llama 3.1 70B (older version)
  gemma         - Google Gemma 2 9B
        """
    )

    parser.add_argument("--info", action="store_true",
                        help="Show system info and exit")
    parser.add_argument("--corpus", action="store_true",
                        help="Show corpus status and exit")
    parser.add_argument("--models", action="store_true",
                        help="Show available models and exit")
    parser.add_argument("--model", type=str,
                        help="Force specific model (overrides auto-detection)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed without running")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GSWA Smart Fine-tuning System")
    print("=" * 70)
    print("Unified fine-tuning for Mac, Linux, and Windows")

    # Detect system
    print("\nDetecting system...")
    info = get_system_info()

    # Handle info-only modes
    if args.info:
        print_system_info(info)
        return 0

    if args.corpus:
        print_corpus_status()
        return 0

    if args.models:
        print_model_options()
        return 0

    # Print system info
    print_system_info(info)

    # Check if training is possible
    if info.recommended_backend == "cpu":
        print("\n" + "=" * 70)
        print("WARNING: No GPU Detected")
        print("=" * 70)
        print("Training on CPU is possible but VERY slow (10-100x slower than GPU).")
        print("Recommended: Use a machine with NVIDIA GPU or Apple Silicon Mac.")

        response = input("\nContinue with CPU training? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return 1

    if info.recommended_backend == "mlx" and not info.mlx_available:
        print("\n" + "=" * 70)
        print("ERROR: MLX Not Installed")
        print("=" * 70)
        print("\nInstall MLX with:")
        print("  pip install mlx mlx-lm")
        return 1

    if info.recommended_backend == "cuda" and not info.cuda_available:
        print("\n" + "=" * 70)
        print("ERROR: PyTorch CUDA Not Available")
        print("=" * 70)
        print("\nInstall PyTorch with CUDA:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return 1

    # Run training
    print("\n" + "=" * 70)
    print("Ready to Train")
    print("=" * 70)

    if not args.dry_run:
        response = input("\nStart training? (Y/n): ")
        if response.lower() == "n":
            print("Aborted.")
            return 1

    success = run_training(info, args)

    if success:
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Check models/ directory for the trained model")
        print("  2. Follow the instructions to create an Ollama model")
        print("  3. Update .env and run: make run")
        return 0
    else:
        print("\n" + "=" * 70)
        print("Training Failed")
        print("=" * 70)
        print("\nCheck the error messages above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
