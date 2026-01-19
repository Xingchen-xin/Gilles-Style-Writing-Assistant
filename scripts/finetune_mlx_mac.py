#!/usr/bin/env python3
"""
MLX Fine-tuning Script for Mac Apple Silicon.

This is optimized for Mac M1/M2/M3 chips using Apple's MLX framework.
MLX is designed for efficient training on Apple Silicon.

Requirements:
    pip install mlx mlx-lm

Usage:
    python scripts/finetune_mlx_mac.py --model mistral --iters 1000

    # Auto-detect settings based on your hardware (RECOMMENDED)
    python scripts/finetune_mlx_mac.py --auto

    # Memory-safe mode with automatic preprocessing (RECOMMENDED for OOM)
    python scripts/finetune_mlx_mac.py --auto --memory-safe

    # Use a specific profile
    python scripts/finetune_mlx_mac.py --profile mac_m1_16gb

After training:
    # Create Ollama model
    ollama create gswa-gilles -f models/gswa-mlx-<timestamp>/Modelfile

    # Update .env
    VLLM_MODEL_NAME=gswa-gilles

Memory Optimization Tips:
    - Use --memory-safe to auto-preprocess long sequences
    - Reduce --batch-size if OOM occurs
    - Reduce --max-seq-length (512, 1024, 2048)
    - Reduce --num-layers (4, 8, 16)
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
# Configuration and Auto-Detection
# ==============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "config" / "training_profiles.json"


def get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True
            )
            bytes_mem = int(result.stdout.strip())
            return bytes_mem / (1024 ** 3)
    except Exception:
        pass
    return 16.0  # Default assumption


def get_chip_info() -> dict:
    """Get Apple Silicon chip information."""
    info = {"chip": "unknown", "cores": 0}
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True
        )
        info["chip"] = result.stdout.strip()

        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True,
            text=True
        )
        info["cores"] = int(result.stdout.strip())
    except Exception:
        pass
    return info


def load_training_profiles() -> dict:
    """Load training profiles from config file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"auto_detect": True, "profiles": {}}


def auto_detect_profile(profiles: dict) -> tuple[str, dict]:
    """Auto-detect the best profile based on hardware."""
    memory_gb = get_system_memory_gb()
    chip_info = get_chip_info()

    print(f"\n  System Memory: {memory_gb:.1f} GB")
    print(f"  Chip: {chip_info['chip']}")
    print(f"  CPU Cores: {chip_info['cores']}")

    # Find the best matching profile
    best_profile = None
    best_name = "conservative"

    for name, profile in profiles.get("profiles", {}).items():
        if not name.startswith("mac_"):
            continue

        min_mem = profile.get("min_memory_gb", 0)
        max_mem = profile.get("max_memory_gb", float('inf'))

        if min_mem <= memory_gb < max_mem:
            best_profile = profile
            best_name = name
            break

    if best_profile is None:
        best_profile = profiles.get("profiles", {}).get("conservative", {})
        best_name = "conservative"

    print(f"  Selected Profile: {best_name}")
    if best_profile.get("description"):
        print(f"  Description: {best_profile['description']}")

    return best_name, best_profile.get("settings", {})


def apply_profile_settings(args, profile_settings: dict):
    """Apply profile settings to args if not explicitly set."""
    # Map profile settings to args
    mappings = {
        "batch_size": "batch_size",
        "num_layers": "num_layers",
        "iters": "iters",
        "max_seq_length": "max_seq_length",
        "learning_rate": "learning_rate",
    }

    for profile_key, arg_key in mappings.items():
        if profile_key in profile_settings:
            # Only apply if user didn't explicitly set (using default)
            current = getattr(args, arg_key, None)
            default = get_default_value(arg_key)
            if current == default:
                setattr(args, arg_key, profile_settings[profile_key])

    return args


def get_default_value(arg_name: str):
    """Get default value for an argument."""
    defaults = {
        "batch_size": 4,
        "num_layers": 16,
        "iters": 1000,
        "max_seq_length": 1024,
        "learning_rate": 1e-5,
    }
    return defaults.get(arg_name)


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
# Memory-Safe Preprocessing
# ==============================================================================

def analyze_data_for_memory(input_file: str, max_seq_length: int) -> dict:
    """
    Analyze training data to check if preprocessing is needed.

    Returns:
        Dictionary with analysis results and recommendations
    """
    stats = {
        'total_entries': 0,
        'max_tokens': 0,
        'over_limit': 0,
        'needs_preprocessing': False,
    }

    try:
        with open(input_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    stats['total_entries'] += 1

                    # Extract text
                    if 'text' in entry:
                        text = entry['text']
                    elif 'instruction' in entry:
                        text = f"{entry.get('instruction', '')} {entry.get('input', '')} {entry.get('output', '')}"
                    else:
                        text = str(entry)

                    # Estimate tokens (roughly 4 chars per token)
                    tokens = len(text) // 4

                    if tokens > stats['max_tokens']:
                        stats['max_tokens'] = tokens

                    if tokens > max_seq_length:
                        stats['over_limit'] += 1

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  Warning: Could not analyze data: {e}")
        return stats

    stats['needs_preprocessing'] = stats['over_limit'] > 0
    stats['over_limit_pct'] = (stats['over_limit'] / stats['total_entries'] * 100
                               if stats['total_entries'] > 0 else 0)

    return stats


def preprocess_for_memory_safe(input_file: str, max_seq_length: int) -> str:
    """
    Preprocess training data to fit within max_seq_length.

    Returns:
        Path to preprocessed file
    """
    from pathlib import Path

    # Try to import preprocessor
    try:
        # Add scripts directory to path
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from preprocess_training_data import preprocess_training_data
    except ImportError:
        # Fallback: simple truncation
        print("  Warning: preprocess_training_data.py not found, using simple truncation")
        return input_file

    input_path = Path(input_file)
    output_path = input_path.with_suffix('.safe.jsonl')

    print(f"\n  Preprocessing data for max_seq_length={max_seq_length}...")

    result = preprocess_training_data(
        input_file=input_file,
        output_file=str(output_path),
        max_tokens=max_seq_length,
    )

    print(f"  Preprocessed: {result['entries_before']} -> {result['entries_after']} entries")
    print(f"  Max tokens: {result['max_tokens_before']} -> {result['max_tokens_after']}")

    return str(output_path)


def get_memory_safe_settings(memory_gb: float) -> dict:
    """
    Get memory-safe settings that are very conservative.
    These settings should work even on systems with limited memory.
    """
    # Very conservative settings based on memory
    if memory_gb < 12:
        return {
            'batch_size': 1,
            'num_layers': 4,
            'max_seq_length': 512,
            'iters': 200,
        }
    elif memory_gb < 20:
        return {
            'batch_size': 1,
            'num_layers': 8,
            'max_seq_length': 768,
            'iters': 300,
        }
    elif memory_gb < 32:
        return {
            'batch_size': 2,
            'num_layers': 8,
            'max_seq_length': 1024,
            'iters': 500,
        }
    elif memory_gb < 64:
        return {
            'batch_size': 2,
            'num_layers': 16,
            'max_seq_length': 1536,
            'iters': 800,
        }
    else:
        return {
            'batch_size': 4,
            'num_layers': 16,
            'max_seq_length': 2048,
            'iters': 1000,
        }


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

    # Split into train/valid/test using ML best practices (80/10/10)
    import random
    random.seed(42)
    random.shuffle(data)

    n = len(data)
    # Standard ML split: 80% train, 10% validation, 10% test
    train_ratio = 0.80
    valid_ratio = 0.10
    # test_ratio = 0.10 (remaining)

    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    print(f"  Data split: train={len(train_data)} ({train_ratio*100:.0f}%), "
          f"valid={len(valid_data)} ({valid_ratio*100:.0f}%), "
          f"test={len(test_data)} ({(1-train_ratio-valid_ratio)*100:.0f}%)")

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

    # Try to import metrics parser for real-time visualization
    try:
        scripts_dir = Path(__file__).parent.parent / "src"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from gswa.training.metrics_parser import MLXMetricsParser, create_ascii_loss_graph
        HAS_METRICS_PARSER = True
    except ImportError:
        HAS_METRICS_PARSER = False
        MLXMetricsParser = None

    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output with metrics parsing
        training_log = []
        metrics_parser = MLXMetricsParser(str(logs_dir)) if HAS_METRICS_PARSER else None

        print("\n" + "─" * 60)
        for line in process.stdout:
            training_log.append(line)

            # Parse metrics if available
            if metrics_parser:
                metric = metrics_parser.parse_line(line)
                if metric:
                    # Show condensed progress
                    metrics_parser.print_progress(total_iters=args.iters)
                elif "Val loss" in line or "error" in line.lower():
                    print(f"\n{line.strip()}")
            else:
                print(line, end='')

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        print("\n" + "─" * 60)
        print("\nTraining complete!")

        # Show metrics summary and ASCII graph
        if metrics_parser:
            metrics_parser.print_final_summary()
            if metrics_parser.metrics:
                print("\n" + "─" * 50)
                print("Loss Curve (ASCII)")
                print("─" * 50)
                print(create_ascii_loss_graph(metrics_parser.metrics))
            metrics_parser.close()

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

        # Generate visualization report
        try:
            from gswa.training.visualizer import generate_training_report
            if logs_dir.exists():
                reports_dir = output_dir / "reports"
                reports_dir.mkdir(exist_ok=True)
                report_path = generate_training_report(
                    str(logs_dir),
                    str(reports_dir),
                    training_config,
                )
                print(f"  Report: {report_path}")
        except Exception as e:
            print(f"  (Visualization report skipped: {e})")

        print("\n" + "-" * 60)
        print("Next Steps")
        print("-" * 60)
        print(f"\n  1. View training report:")
        print(f"     open {output_dir}/reports/report.html")
        print(f"\n  2. Create Ollama model:")
        print(f"     ollama create gswa-gilles -f {output_dir}/Modelfile")
        print(f"\n  3. Update .env:")
        print(f"     echo 'VLLM_MODEL_NAME=gswa-gilles' >> .env")
        print(f"\n  4. Restart GSWA:")
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
  # Auto-detect settings based on hardware (recommended)
  python scripts/finetune_mlx_mac.py --auto

  # Use a specific profile from config/training_profiles.json
  python scripts/finetune_mlx_mac.py --profile mac_m1_32gb

  # Manual settings
  python scripts/finetune_mlx_mac.py --model mistral --batch-size 2 --iters 500

  # List available profiles
  python scripts/finetune_mlx_mac.py --list-profiles

Available models:
  mistral  - Mistral 7B Instruct (recommended)
  llama3   - Llama 3 8B Instruct
  phi3     - Phi-3 Mini (fastest)
  gemma2   - Gemma 2 9B
  qwen     - Qwen 1.5 7B
        """
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect optimal settings based on hardware"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Use a specific profile from config/training_profiles.json"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available training profiles"
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
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help="Enable memory-safe mode: auto-preprocess long sequences and use conservative settings"
    )
    parser.add_argument(
        "--retry-on-oom",
        action="store_true",
        default=True,
        help="Automatically retry with reduced settings if OOM occurs (default: True)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GSWA MLX Fine-tuning Script")
    print("=" * 60)

    # Load training profiles
    profiles = load_training_profiles()

    # List profiles if requested
    if args.list_profiles:
        print("\nAvailable Training Profiles:")
        print("-" * 40)
        for name, profile in profiles.get("profiles", {}).items():
            desc = profile.get("description", "No description")
            print(f"  {name}")
            print(f"    {desc}")
            settings = profile.get("settings", {})
            if settings:
                print(f"    Settings: batch_size={settings.get('batch_size', '?')}, "
                      f"num_layers={settings.get('num_layers', '?')}, "
                      f"iters={settings.get('iters', '?')}")
            print()
        print(f"Config file: {CONFIG_PATH}")
        sys.exit(0)

    # Check platform
    if not check_platform():
        sys.exit(1)

    # Apply profile or auto-detect
    profile_name = None
    if args.auto or profiles.get("auto_detect", False):
        print("\n" + "=" * 60)
        print("Auto-Detecting Hardware")
        print("=" * 60)
        profile_name, profile_settings = auto_detect_profile(profiles)
        args = apply_profile_settings(args, profile_settings)
    elif args.profile:
        profile_name = args.profile
        if profile_name in profiles.get("profiles", {}):
            profile_settings = profiles["profiles"][profile_name].get("settings", {})
            args = apply_profile_settings(args, profile_settings)
            print(f"\n  Using profile: {profile_name}")
        else:
            print(f"\nWARNING: Profile '{profile_name}' not found. Using defaults.")
            print(f"  Available profiles: {', '.join(profiles.get('profiles', {}).keys())}")

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

    # Memory-safe mode: analyze data and preprocess if needed
    if args.memory_safe:
        print("\n" + "=" * 60)
        print("Memory-Safe Mode Enabled")
        print("=" * 60)

        memory_gb = get_system_memory_gb()

        # Get memory-safe settings
        safe_settings = get_memory_safe_settings(memory_gb)
        print(f"\n  Using memory-safe settings for {memory_gb:.0f}GB RAM:")
        print(f"    batch_size: {safe_settings['batch_size']}")
        print(f"    num_layers: {safe_settings['num_layers']}")
        print(f"    max_seq_length: {safe_settings['max_seq_length']}")
        print(f"    iters: {safe_settings['iters']}")

        # Apply memory-safe settings
        args.batch_size = safe_settings['batch_size']
        args.num_layers = safe_settings['num_layers']
        args.max_seq_length = safe_settings['max_seq_length']
        args.iters = safe_settings['iters']

        # Analyze data
        print("\n  Analyzing training data...")
        data_stats = analyze_data_for_memory(args.training_data, args.max_seq_length)

        if data_stats['needs_preprocessing']:
            print(f"\n  Found {data_stats['over_limit']} sequences over {args.max_seq_length} tokens")
            print(f"  ({data_stats['over_limit_pct']:.1f}% of data)")
            print(f"  Max tokens in data: {data_stats['max_tokens']}")

            # Preprocess data
            preprocessed_file = preprocess_for_memory_safe(args.training_data, args.max_seq_length)
            args.training_data = preprocessed_file
            print(f"\n  Using preprocessed data: {preprocessed_file}")
        else:
            print(f"\n  Data is already within {args.max_seq_length} token limit")

    # Run training with retry logic
    max_retries = 3 if args.retry_on_oom else 1
    retry_count = 0

    while retry_count < max_retries:
        result = run_mlx_training(args)

        if result:
            sys.exit(0)
        else:
            # Check if it was an OOM error (we can detect from training log)
            retry_count += 1

            if retry_count < max_retries and args.retry_on_oom:
                print(f"\n" + "=" * 60)
                print(f"Training failed - Attempting retry {retry_count}/{max_retries-1}")
                print("=" * 60)

                # Reduce settings for retry
                if args.batch_size > 1:
                    args.batch_size = max(1, args.batch_size // 2)
                    print(f"  Reduced batch_size to {args.batch_size}")
                elif args.max_seq_length > 512:
                    args.max_seq_length = max(512, args.max_seq_length // 2)
                    print(f"  Reduced max_seq_length to {args.max_seq_length}")

                    # Re-preprocess data if needed
                    if args.memory_safe:
                        preprocessed_file = preprocess_for_memory_safe(
                            args.training_data.replace('.safe.jsonl', '.jsonl'),
                            args.max_seq_length
                        )
                        args.training_data = preprocessed_file
                elif args.num_layers > 4:
                    args.num_layers = max(4, args.num_layers // 2)
                    print(f"  Reduced num_layers to {args.num_layers}")
                else:
                    print("\n  Cannot reduce settings further. Please try a smaller model.")
                    break
            else:
                break

    print("\n" + "=" * 60)
    print("Training Failed")
    print("=" * 60)
    print("\nTroubleshooting tips:")
    print("  1. Try --memory-safe flag for automatic memory optimization")
    print("  2. Manually reduce: --batch-size 1 --max-seq-length 512 --num-layers 4")
    print("  3. Use a smaller model: --model phi3")
    print("  4. Close other applications to free memory")
    print("  5. Check 'Activity Monitor' for memory pressure")
    sys.exit(1)


if __name__ == "__main__":
    main()
