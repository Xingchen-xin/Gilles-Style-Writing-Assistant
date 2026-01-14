#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for GSWA (Linux/Windows with NVIDIA GPU).

This script fine-tunes a base model on Gilles's writing style using LoRA.
Supports Linux and Windows with NVIDIA GPUs, with auto-detection of hardware.

Requirements:
    pip install torch transformers peft datasets accelerate bitsandbytes

Usage:
    # Auto-detect settings
    python scripts/finetune_lora.py --auto

    # Basic usage with QLoRA (recommended)
    python scripts/finetune_lora.py --quantize 4bit

    # Use a different base model
    python scripts/finetune_lora.py --model Qwen/Qwen2.5-7B-Instruct

    # CPU-only mode (slow but works without GPU)
    python scripts/finetune_lora.py --cpu
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "config" / "training_profiles.json"

# Model aliases for easier usage
MODEL_ALIASES = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "gemma": "google/gemma-2-9b-it",
    "gemma-2b": "google/gemma-2-2b-it",
}


# ==============================================================================
# Auto-Detection
# ==============================================================================

def get_gpu_info() -> dict:
    """Detect NVIDIA GPU information."""
    info = {
        "available": False,
        "name": None,
        "vram_gb": 0,
        "cuda_available": False,
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            info["name"] = parts[0].strip()
            info["vram_gb"] = float(parts[1].strip()) / 1024
            info["available"] = True
    except Exception:
        pass

    # Check CUDA via PyTorch
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"] and not info["name"]:
            info["name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["available"] = True
    except Exception:
        pass

    return info


def get_system_memory_gb() -> float:
    """Get system RAM in GB."""
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 16.0


def load_training_profiles() -> dict:
    """Load training profiles from config file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"profiles": {}}


def auto_detect_settings() -> dict:
    """Auto-detect optimal settings based on hardware."""
    gpu_info = get_gpu_info()
    ram_gb = get_system_memory_gb()

    print("\n" + "=" * 60)
    print("Auto-Detecting Hardware")
    print("=" * 60)
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  System RAM: {ram_gb:.1f} GB")

    if gpu_info["available"]:
        print(f"  GPU: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['vram_gb']:.1f} GB")
        print(f"  CUDA: {'Available' if gpu_info['cuda_available'] else 'Not available'}")
        vram = gpu_info["vram_gb"]
    else:
        print("  GPU: Not detected (will use CPU)")
        vram = 0

    # Select settings based on VRAM
    settings = {
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_length": 512,
        "quantize": "4bit",
        "lora_r": 8,
        "lora_alpha": 16,
        "epochs": 3,
        "use_cpu": not gpu_info["cuda_available"],
    }

    if vram >= 48:
        settings.update({
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "max_length": 2048,
            "quantize": "none",
            "lora_r": 32,
            "lora_alpha": 64,
        })
        profile = "High-end GPU (48GB+)"
    elif vram >= 24:
        settings.update({
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_length": 2048,
            "quantize": "4bit",
            "lora_r": 16,
            "lora_alpha": 32,
        })
        profile = "Professional GPU (24GB)"
    elif vram >= 16:
        settings.update({
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_length": 1536,
            "quantize": "4bit",
            "lora_r": 16,
            "lora_alpha": 32,
        })
        profile = "Standard GPU (16GB)"
    elif vram >= 8:
        settings.update({
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 1024,
            "quantize": "4bit",
            "lora_r": 8,
            "lora_alpha": 16,
        })
        profile = "Entry GPU (8GB)"
    elif vram >= 4:
        settings.update({
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_length": 512,
            "quantize": "4bit",
            "lora_r": 4,
            "lora_alpha": 8,
        })
        profile = "Low VRAM (4GB)"
    else:
        profile = "CPU Only"

    print(f"\n  Selected Profile: {profile}")
    print(f"  Settings: batch={settings['batch_size']}, "
          f"max_len={settings['max_length']}, "
          f"quantize={settings['quantize']}")

    return settings


# ==============================================================================
# Dependency Checking
# ==============================================================================

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)

    deps = {
        "torch": None,
        "transformers": None,
        "peft": None,
        "datasets": None,
        "accelerate": None,
    }

    missing = []

    for dep in deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            deps[dep] = version
            print(f"  {dep}: {version}")
        except ImportError:
            missing.append(dep)
            print(f"  {dep}: NOT INSTALLED")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  CUDA available: Yes")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"\n  CUDA available: No (will use CPU, very slow)")
    except Exception:
        pass

    if missing:
        print(f"\n  Missing: {', '.join(missing)}")
        print("\n  Install with:")
        print("    pip install torch transformers peft datasets accelerate bitsandbytes")
        return False

    print("\n  Status: Ready for training")
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
# Model Version Management
# ==============================================================================

def get_model_version_dir(base_dir: str, model_name: str) -> Path:
    """Create versioned model directory with metadata."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    version_name = f"gswa-lora-{model_name}-{timestamp}"
    version_dir = base_path / version_name

    # Check for existing models and get next version number
    existing = list(base_path.glob(f"gswa-lora-{model_name}-*"))
    version_num = len(existing) + 1

    version_dir.mkdir(parents=True, exist_ok=True)

    # Create version metadata
    metadata = {
        "version": version_num,
        "version_name": version_name,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "status": "training",
        "framework": "transformers+peft",
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
    for model_dir in sorted(base_path.glob("gswa-lora-*")):
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


def train_with_transformers(args):
    """Train using Hugging Face Transformers + PEFT."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("GSWA LoRA Fine-tuning")
    print("=" * 60)

    # Setup quantization if requested
    bnb_config = None
    if args.quantize == "4bit":
        print("\nUsing 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantize == "8bit":
        print("\nUsing 8-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print(f"\nConfiguring LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print(f"\nLoading training data: {args.training_data}")

    def load_jsonl(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    raw_data = load_jsonl(args.training_data)
    print(f"  Loaded {len(raw_data)} training examples")

    # Format for training
    def format_example(example):
        if "instruction" in example:
            # Alpaca format
            prompt = f"### Instruction:\n{example['instruction']}\n\n"
            if example.get("input"):
                prompt += f"### Input:\n{example['input']}\n\n"
            prompt += f"### Response:\n{example['output']}"
        elif "conversations" in example:
            # ShareGPT format
            conv = example["conversations"]
            prompt = f"### Human:\n{conv[0]['value']}\n\n### Assistant:\n{conv[1]['value']}"
        elif "text" in example:
            # Completion format
            prompt = example["text"]
        else:
            prompt = str(example)

        return {"text": prompt}

    formatted_data = [format_example(ex) for ex in raw_data]

    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Split into train/eval
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Evaluation samples: {len(eval_dataset)}")

    # Create versioned output directory
    model_short = args.base_model.split("/")[-1].split("-")[0]
    output_dir = get_model_version_dir(args.output_dir, model_short)
    print(f"\n  Output directory: {output_dir}")

    # Save training config
    training_config = {
        "base_model": args.base_model,
        "model_short": model_short,
        "training_data": args.training_data,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "quantization": args.quantize,
        "max_length": args.max_length,
        "started_at": datetime.now().isoformat(),
    }

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print("\nThis may take a while depending on your hardware...")
    print("Training progress will be shown below:")
    print("-" * 40)

    try:
        trainer.train()

        print("-" * 40)
        print("\nTraining complete!")

        # Save model
        print(f"\nSaving model to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Update metadata
        update_model_metadata(output_dir, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        })

        # Update registry
        registry = create_model_registry(args.output_dir)

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"  Model saved to: {output_dir}")
        print(f"  Total models in registry: {len(registry['models'])}")

        print("\n" + "-" * 60)
        print("Next Steps")
        print("-" * 60)
        print("\n  Option 1: Load directly in Python:")
        print(f"    from peft import PeftModel")
        print(f"    model = PeftModel.from_pretrained(base_model, '{output_dir}')")
        print("\n  Option 2: Merge with base model:")
        print(f"    python scripts/merge_lora.py --adapter {output_dir}")
        print("\n  Option 3: Use with vLLM:")
        print(f"    Configure LORA_ADAPTER_PATH={output_dir} in .env")

        return str(output_dir)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        update_model_metadata(output_dir, {
            "status": "interrupted",
            "interrupted_at": datetime.now().isoformat(),
        })
        return None

    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        update_model_metadata(output_dir, {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat(),
        })
        return None


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for GSWA (Linux/Windows with NVIDIA GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect optimal settings (recommended)
  python scripts/finetune_lora.py --auto

  # Use a specific model with alias
  python scripts/finetune_lora.py --model qwen

  # Use a specific model with full path
  python scripts/finetune_lora.py --model Qwen/Qwen2.5-7B-Instruct

  # CPU-only training (slow)
  python scripts/finetune_lora.py --cpu

Available model aliases:
  mistral, mistral-large, llama3, llama3-70b, qwen, qwen-14b, qwen-1.5b,
  phi, gemma, gemma-2b

Note: For Mac users, use scripts/finetune_mlx_mac.py instead.
        """
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect optimal settings based on hardware"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to fine-tune (alias like 'qwen' or full HuggingFace path)"
    )
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="[Deprecated: use --model] Base model to fine-tune"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (very slow)"
    )
    parser.add_argument(
        "--training-data",
        default="./data/training/alpaca_train.jsonl",
        help="Training data file (default: ./data/training/alpaca_train.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory for fine-tuned model (default: ./models)"
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="4bit",
        help="Quantization level (default: 4bit for QLoRA)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies, don't train"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GSWA LoRA Fine-tuning Script")
    print("=" * 60)

    # Handle model selection
    if args.model:
        # Resolve alias if provided
        args.base_model = MODEL_ALIASES.get(args.model.lower(), args.model)
        print(f"\nModel: {args.base_model}")

    # Auto-detect settings if requested
    if args.auto:
        auto_settings = auto_detect_settings()

        # Apply auto-detected settings (only if not explicitly set by user)
        if args.batch_size == 4:  # default
            args.batch_size = auto_settings["batch_size"]
        if args.gradient_accumulation_steps == 4:  # default
            args.gradient_accumulation_steps = auto_settings["gradient_accumulation_steps"]
        if args.max_length == 2048:  # default
            args.max_length = auto_settings["max_length"]
        if args.quantize == "4bit":  # default
            args.quantize = auto_settings["quantize"]
        if args.lora_r == 16:  # default
            args.lora_r = auto_settings["lora_r"]
        if args.lora_alpha == 32:  # default
            args.lora_alpha = auto_settings["lora_alpha"]

        # Set CPU mode if auto-detected
        if auto_settings.get("use_cpu"):
            args.cpu = True

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    if args.check_only:
        print("\nDependency check passed!")
        sys.exit(0)

    # CPU mode warning
    if args.cpu:
        print("\n" + "=" * 60)
        print("WARNING: CPU Training Mode")
        print("=" * 60)
        print("Training on CPU is 10-100x slower than GPU.")
        print("Consider using a machine with NVIDIA GPU for better performance.")

        import torch
        if torch.cuda.is_available():
            print("\nNote: CUDA GPU detected but --cpu flag is set.")

    # Check training data
    if not check_training_data(args.training_data):
        sys.exit(1)

    # Run training
    result = train_with_transformers(args)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
