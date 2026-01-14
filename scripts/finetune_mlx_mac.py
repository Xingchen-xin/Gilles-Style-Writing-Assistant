#!/usr/bin/env python3
"""
MLX Fine-tuning Script for Mac Apple Silicon.

This is optimized for Mac M1/M2/M3 chips using Apple's MLX framework.
MLX is designed for efficient training on Apple Silicon.

Requirements:
    pip install mlx mlx-lm

Usage:
    python scripts/finetune_mlx_mac.py --model mistral --iters 1000

After training:
    # Create Ollama model
    ollama create gswa-gilles -f models/gswa-mlx-<timestamp>/Modelfile

    # Update .env
    VLLM_MODEL_NAME=gswa-gilles
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ==============================================================================
# Dependency Checking
# ==============================================================================

def check_platform():
    """Check if running on Mac with Apple Silicon."""
    if platform.system() != "Darwin":
        print("ERROR: This script is designed for macOS only.")
        print(f"       Current platform: {platform.system()}")
        print("\nFor Linux, use: make finetune-lora")
        return False

    # Check for Apple Silicon
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True
        )
        cpu = result.stdout.strip()
        if "Apple" not in cpu:
            print(f"WARNING: This script is optimized for Apple Silicon.")
            print(f"         Detected CPU: {cpu}")
            print("         Performance may be suboptimal on Intel Macs.")
    except Exception:
        pass

    return True


def check_mlx():
    """Check if MLX is available and working."""
    print("\n" + "=" * 60)
    print("Checking MLX Installation")
    print("=" * 60)

    # Check mlx
    try:
        import mlx
        import mlx.core as mx

        # Try to get version - different methods for different versions
        mlx_version = getattr(mlx, '__version__', None)
        if mlx_version is None:
            # Try getting version from package metadata
            try:
                from importlib.metadata import version
                mlx_version = version('mlx')
            except Exception:
                mlx_version = "unknown"

        print(f"  MLX installed: Yes (version {mlx_version})")
        print(f"  Default device: {mx.default_device()}")

    except ImportError as e:
        print(f"  MLX installed: No")
        print(f"  Error: {e}")
        print("\n  Install with: pip install mlx")
        return False

    # Check mlx-lm
    try:
        import mlx_lm
        mlx_lm_version = getattr(mlx_lm, '__version__', None)
        if mlx_lm_version is None:
            try:
                from importlib.metadata import version
                mlx_lm_version = version('mlx-lm')
            except Exception:
                mlx_lm_version = "unknown"

        print(f"  mlx-lm installed: Yes (version {mlx_lm_version})")

    except ImportError as e:
        print(f"  mlx-lm installed: No")
        print(f"  Error: {e}")
        print("\n  Install with: pip install mlx-lm")
        return False

    print("  Status: Ready for training")
    return True


def check_training_data(path: str) -> bool:
    """Check if training data exists and is valid."""
    data_path = Path(path)

    if not data_path.exists():
        print(f"\nERROR: Training data not found: {path}")
        print("\nGenerate training data first:")
        print("  1. Put PDF files in data/corpus/raw/")
        print("  2. Run: make parse-corpus")
        print("  3. Run: make prepare-training")
        return False

    # Count entries
    count = 0
    try:
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"\nERROR: Cannot read training data: {e}")
        return False

    if count < 10:
        print(f"\nWARNING: Training data has only {count} entries.")
        print("         Consider adding more corpus documents for better results.")

    print(f"\nTraining data: {path}")
    print(f"  Entries: {count}")
    return True


# ==============================================================================
# Data Preparation
# ==============================================================================

def prepare_mlx_data(input_file: str, output_dir: str) -> str:
    """Prepare data in MLX format with progress display."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 60)
    print("Preparing Training Data")
    print("-" * 60)

    # Load data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"  Loaded {len(data)} entries from {input_file}")

    # Split into train/valid/test
    import random
    random.seed(42)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * 0.9)
    valid_end = int(n * 0.95)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    # Format for MLX (expects "text" field)
    def format_for_mlx(item):
        if "instruction" in item:
            text = f"### Instruction: {item['instruction']}\n"
            if item.get("input"):
                text += f"### Input: {item['input']}\n"
            text += f"### Response: {item['output']}"
        elif "conversations" in item:
            conv = item["conversations"]
            text = f"### Human: {conv[0]['value']}\n### Assistant: {conv[1]['value']}"
        elif "text" in item:
            text = item["text"]
        else:
            text = str(item)
        return {"text": text}

    # Save files
    for name, subset in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        filepath = output_dir / f"{name}.jsonl"
        with open(filepath, 'w') as f:
            for item in subset:
                formatted = format_for_mlx(item)
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
        print(f"  {name}.jsonl: {len(subset)} entries")

    return str(output_dir)


# ==============================================================================
# Model Version Management
# ==============================================================================

def get_model_version_dir(base_dir: str, model_name: str) -> Path:
    """Create versioned model directory with metadata."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    version_name = f"gswa-mlx-{model_name}-{timestamp}"
    version_dir = base_path / version_name

    # Check for existing models and get next version number
    existing = list(base_path.glob(f"gswa-mlx-{model_name}-*"))
    version_num = len(existing) + 1

    version_dir.mkdir(parents=True, exist_ok=True)

    # Create version metadata
    metadata = {
        "version": version_num,
        "version_name": version_name,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "status": "training",
    }

    with open(version_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return version_dir


def update_model_metadata(version_dir: Path, updates: dict):
    """Update model metadata file."""
    metadata_file = version_dir / "metadata.json"

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata.update(updates)
    metadata["updated_at"] = datetime.now().isoformat()

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def create_model_registry(base_dir: str):
    """Create/update model registry file."""
    base_path = Path(base_dir)
    registry_file = base_path / "registry.json"

    models = []
    for model_dir in sorted(base_path.glob("gswa-mlx-*")):
        if model_dir.is_dir():
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    metadata["path"] = str(model_dir)
                    models.append(metadata)

    registry = {
        "updated_at": datetime.now().isoformat(),
        "models": models,
        "latest": models[-1]["version_name"] if models else None,
    }

    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)

    return registry


# ==============================================================================
# Training
# ==============================================================================

def run_mlx_training(args) -> Path | None:
    """Run MLX LoRA training with progress display."""
    print("\n" + "=" * 60)
    print("GSWA MLX Fine-tuning (Apple Silicon)")
    print("=" * 60)

    # Model mapping
    model_map = {
        "mistral": "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        "llama2": "mlx-community/Llama-2-7b-chat-mlx-4bit",
        "llama3": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "phi": "mlx-community/phi-2-4bit",
        "phi3": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "gemma": "mlx-community/gemma-7b-it-4bit",
        "gemma2": "mlx-community/gemma-2-9b-it-4bit",
        "qwen": "mlx-community/Qwen1.5-7B-Chat-4bit",
    }

    model_id = model_map.get(args.model, args.model)
    model_short = args.model.split("/")[-1].split("-")[0] if "/" in args.model else args.model

    print(f"\nConfiguration:")
    print(f"  Base model: {model_id}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Iterations: {args.iters}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max seq length: {args.max_seq_length}")

    # Prepare training data
    data_dir = prepare_mlx_data(args.training_data, "./data/training/mlx")

    # Create versioned output directory
    output_dir = get_model_version_dir(args.output_dir, model_short)
    print(f"\n  Output directory: {output_dir}")

    # Build mlx_lm command (updated for newer mlx-lm versions)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_id,
        "--train",
        "--data", data_dir,
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.num_layers),
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--max-seq-length", str(args.max_seq_length),
        "--adapter-path", str(output_dir / "adapters"),
    ]

    # Save training config
    training_config = {
        "base_model": model_id,
        "model_short": model_short,
        "training_data": args.training_data,
        "data_dir": data_dir,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "command": " ".join(cmd),
        "started_at": datetime.now().isoformat(),
    }

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2)

    print("\n" + "-" * 60)
    print("Starting Training")
    print("-" * 60)
    print(f"\nCommand: {' '.join(cmd[:6])}...")
    print("\nTraining output:")
    print("-" * 40)

    try:
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output
        training_log = []
        for line in process.stdout:
            print(line, end='')
            training_log.append(line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        print("-" * 40)
        print("\nTraining complete!")

        # Save training log
        with open(output_dir / "training.log", 'w') as f:
            f.writelines(training_log)

        # Update metadata
        update_model_metadata(output_dir, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        })

        # Create Ollama Modelfile
        create_ollama_modelfile(output_dir, model_id, model_short)

        # Update registry
        registry = create_model_registry(args.output_dir)

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"  Model saved to: {output_dir}")
        print(f"  Adapters: {output_dir}/adapters")
        print(f"  Modelfile: {output_dir}/Modelfile")
        print(f"  Total models in registry: {len(registry['models'])}")

        print("\n" + "-" * 60)
        print("Next Steps")
        print("-" * 60)
        print(f"\n  1. Create Ollama model:")
        print(f"     ollama create gswa-gilles -f {output_dir}/Modelfile")
        print(f"\n  2. Update .env:")
        print(f"     echo 'VLLM_MODEL_NAME=gswa-gilles' >> .env")
        print(f"\n  3. Restart GSWA:")
        print(f"     make run")

        return output_dir

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")

        # Update metadata
        update_model_metadata(output_dir, {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat(),
        })

        return None

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

        # Update metadata
        update_model_metadata(output_dir, {
            "status": "interrupted",
            "interrupted_at": datetime.now().isoformat(),
        })

        return None


def create_ollama_modelfile(output_dir: Path, base_model: str, model_short: str):
    """Create Ollama Modelfile for the fine-tuned adapter."""
    # Get the base model name for Ollama
    ollama_base = {
        "mistral": "mistral",
        "llama2": "llama2",
        "llama3": "llama3",
        "phi": "phi",
        "phi3": "phi3",
        "gemma": "gemma",
        "gemma2": "gemma2",
        "qwen": "qwen",
    }.get(model_short, model_short)

    modelfile_content = f'''# GSWA Gilles-Style Model
# Fine-tuned on Gilles van Wezel's writing corpus
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

FROM {ollama_base}

# Apply the LoRA adapter
ADAPTER {output_dir.absolute()}/adapters

# System prompt for Gilles-style writing
SYSTEM """You are a scientific writing assistant trained on the papers of Prof. Gilles van Wezel.
Your writing style is characterized by:
- Clear, precise scientific prose with logical flow
- Careful qualification of claims and conclusions
- Rich vocabulary while maintaining accessibility
- Smooth transitions between concepts
- Background context before presenting new findings

When rewriting scientific text:
1. Preserve all numerical values, statistics, and experimental details exactly
2. Maintain the original meaning and conclusions
3. Improve clarity, flow, and readability
4. Use Gilles's characteristic academic style"""

# Parameters optimized for scientific writing
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
'''

    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"\nOllama Modelfile created: {modelfile_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLX fine-tuning for Mac Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with Mistral
  python scripts/finetune_mlx_mac.py --model mistral --iters 1000

  # Training with more iterations for better quality
  python scripts/finetune_mlx_mac.py --model mistral --iters 2000 --batch-size 2

  # Use a smaller model for faster training
  python scripts/finetune_mlx_mac.py --model phi --iters 500

Available models:
  mistral  - Mistral 7B Instruct (recommended)
  llama3   - Llama 3 8B Instruct
  phi3     - Phi-3 Mini (fastest)
  gemma2   - Gemma 2 9B
  qwen     - Qwen 1.5 7B
        """
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Model to fine-tune (mistral, llama3, phi3, gemma2, qwen, or HuggingFace path)"
    )
    parser.add_argument(
        "--training-data",
        default="./data/training/alpaca_train.jsonl",
        help="Training data file (default: ./data/training/alpaca_train.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory for models (default: ./models)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4, reduce if OOM)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=16,
        help="Number of layers to apply LoRA to (default: 16)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Training iterations (default: 1000)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024, reduce if OOM)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies, don't train"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GSWA MLX Fine-tuning Script")
    print("=" * 60)

    # Check platform
    if not check_platform():
        sys.exit(1)

    # Check MLX
    if not check_mlx():
        print("\n" + "=" * 60)
        print("Installation Instructions")
        print("=" * 60)
        print("\n  pip install mlx mlx-lm")
        print("\nNote: MLX requires Mac with Apple Silicon (M1/M2/M3/M4)")
        sys.exit(1)

    if args.check_only:
        print("\nDependency check passed!")
        sys.exit(0)

    # Check training data
    if not check_training_data(args.training_data):
        sys.exit(1)

    # Run training
    result = run_mlx_training(args)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
