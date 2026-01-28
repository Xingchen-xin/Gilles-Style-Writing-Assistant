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
# Ungated models (no login required):
#   mistral, mistral-large, mistral-nemo, qwen, qwen-14b, phi
# Gated models (require HuggingFace login + approval):
#   llama3.3, llama3-8b, llama3-70b, gemma
MODEL_ALIASES = {
    # Ungated - Mistral
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    # Ungated - Qwen
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # Ungated - Microsoft
    "phi": "microsoft/Phi-3.5-mini-instruct",
    # Gated - Meta Llama (require HF login + Meta approval)
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    # Gated - Google (require HF login + Google approval)
    "gemma": "google/gemma-2-9b-it",
    "gemma-2b": "google/gemma-2-2b-it",
}


# ==============================================================================
# Auto-Detection
# ==============================================================================

def get_gpu_info() -> dict:
    """Detect NVIDIA GPU information. Supports multi-GPU setups by summing VRAM."""
    info = {
        "available": False,
        "name": None,
        "vram_gb": 0,
        "gpu_count": 0,
        "cuda_available": False,
    }

    try:
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

            # Display name: show count if multiple GPUs
            if gpu_count > 1:
                info["name"] = f"{gpu_names[0]} x{gpu_count}"
            else:
                info["name"] = gpu_names[0]

            info["vram_gb"] = total_vram_mb / 1024  # Total VRAM across all GPUs
            info["gpu_count"] = gpu_count
            info["available"] = True
    except Exception:
        pass

    # Check CUDA via PyTorch
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"] and not info["name"]:
            gpu_count = torch.cuda.device_count()
            total_vram = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(gpu_count)
            )
            info["name"] = torch.cuda.get_device_name(0)
            if gpu_count > 1:
                info["name"] = f"{info['name']} x{gpu_count}"
            info["vram_gb"] = total_vram / (1024**3)
            info["gpu_count"] = gpu_count
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

    if vram >= 60:
        settings.update({
            "batch_size": 1,  # 70B model with 4bit still needs batch_size=1
            "gradient_accumulation_steps": 8,  # Effective batch = 8
            "max_length": 1024,  # Reduced to save memory
            "quantize": "4bit",  # 4bit for 70B model
            "lora_r": 16,  # Smaller rank to save memory
            "lora_alpha": 32,
        })
        profile = "Multi-GPU (60GB+) - Optimal for 70B models"
    elif vram >= 48:
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
            gpu_count = torch.cuda.device_count()
            total_memory = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(gpu_count)
            )
            gpu_name = torch.cuda.get_device_name(0)
            if gpu_count > 1:
                gpu_name = f"{gpu_name} x{gpu_count}"
            print(f"\n  CUDA available: Yes")
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Memory: {total_memory / 1e9:.1f} GB (total across {gpu_count} GPU{'s' if gpu_count > 1 else ''})")
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
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    from datasets import load_dataset

    # Determine DDP rank early to suppress duplicate output from workers
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    gpu_count = torch.cuda.device_count()
    is_main = local_rank <= 0  # True for rank 0 (DDP main) or -1 (non-DDP)

    if is_main:
        print("\n" + "=" * 60)
        print("GSWA LoRA Fine-tuning")
        print("=" * 60)

    # Setup quantization if requested
    bnb_config = None
    if args.quantize == "4bit":
        if is_main:
            print("\nUsing 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantize == "8bit":
        if is_main:
            print("\nUsing 8-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer
    if is_main:
        print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer_kwargs = {"trust_remote_code": True}

    # Suppress Mistral tokenizer warning and apply fix
    if "mistral" in args.base_model.lower() or "nemo" in args.base_model.lower():
        tokenizer_kwargs["fix_mistral_regex"] = True
        if is_main:
            print("  Applying Mistral tokenizer regex fix")
        # Suppress the warning about incorrect regex pattern
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)

    # Configure padding for training (right padding for causal LM)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        # Use a dedicated pad token if available, otherwise use eos
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_main:
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    # Load model
    if is_main:
        print(f"Loading model: {args.base_model}")

    # Determine device map for GPU assignment
    # DDP mode: accelerate launch sets LOCAL_RANK, each process uses its own GPU
    # Single mode: use GPU 0
    if local_rank >= 0:
        # Running under accelerate/torchrun DDP - each process uses its assigned GPU
        device_map = {"": local_rank}
        if local_rank == 0:
            print(f"  DDP mode: {gpu_count} GPUs, NCCL P2P disabled for compatibility")
    else:
        device_map = {"": 0}
        if gpu_count > 1:
            print(f"  Single GPU mode (use 'accelerate launch' for multi-GPU DDP)")
        else:
            print(f"  Using GPU 0")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16 if bnb_config else None,
    )

    # DDP fix: remove hf_device_map so Trainer wraps model with DDP
    # Without this, Trainer thinks model is in pipeline-parallel mode and skips DDP
    if local_rank >= 0 and hasattr(model, "hf_device_map"):
        del model.hf_device_map

    # Prepare model for k-bit training with gradient checkpointing
    # Gradient checkpointing saves memory by recomputing activations during backward pass
    # For Mistral models, use_reentrant=True is required to avoid hanging
    model_name_lower = args.base_model.lower()
    use_reentrant = "mistral" in model_name_lower or "nemo" in model_name_lower
    gc_kwargs = {"use_reentrant": use_reentrant}

    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs=gc_kwargs
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()

    if is_main:
        print(f"  Gradient checkpointing enabled (reentrant={use_reentrant})")

    # Verify model vocab size matches tokenizer
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    if is_main:
        print(f"\n  Model vocab size: {model_vocab_size}")
        print(f"  Tokenizer vocab size: {tokenizer_vocab_size}")

    # Note: Don't resize embeddings for quantized models - it can cause issues
    # The model vocab size should be >= tokenizer vocab size
    if model_vocab_size < tokenizer_vocab_size and is_main:
        print(f"  WARNING: Model vocab ({model_vocab_size}) < tokenizer vocab ({tokenizer_vocab_size})")
        print(f"           Some tokens may cause errors. Consider using a different model.")

    # Configure LoRA
    if is_main:
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
    if is_main:
        model.print_trainable_parameters()

    # Load and prepare dataset
    if is_main:
        print(f"\nLoading training data: {args.training_data}")

    def load_jsonl(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    raw_data = load_jsonl(args.training_data)
    if is_main:
        print(f"  Loaded {len(raw_data)} training examples")

    # Format for training using the model's native chat template
    # This ensures training format matches inference format exactly
    def format_example(example):
        if "instruction" in example:
            # Alpaca format → chat template
            user_content = example['instruction']
            if example.get("input"):
                user_content += f"\n\n{example['input']}"
            response = example['output']
        elif "conversations" in example:
            # ShareGPT format → chat template
            conv = example["conversations"]
            user_content = conv[0]['value']
            response = conv[1]['value']
        elif "text" in example:
            # Completion format - no chat template, train on everything
            return {"text": example["text"], "prompt_prefix": ""}
        else:
            return {"text": str(example), "prompt_prefix": ""}

        # Use the model's native chat template for consistent formatting
        # Prompt prefix = everything up to the response (masked during training)
        messages_prompt = [{"role": "user", "content": user_content}]
        prompt_prefix = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

        # Full text = prompt + response + EOS
        messages_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        return {"text": full_text, "prompt_prefix": prompt_prefix}

    formatted_data = [format_example(ex) for ex in raw_data]

    if is_main:
        # Show what the training format looks like
        sample = formatted_data[0]
        print(f"\n  Training format (native chat template):")
        print(f"    Prompt prefix: {repr(sample['prompt_prefix'][:100])}...")
        print(f"    Full text: {repr(sample['text'][:150])}...")

    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data)

    # Get pad token id for label masking
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    def tokenize_function(examples):
        # Chat template text already includes <s> and </s> tokens,
        # so we use add_special_tokens=False to avoid doubling them
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            add_special_tokens=False,
            return_tensors=None,  # Return lists, not tensors
        )

        # Tokenize prompt prefixes to determine masking boundaries
        # We mask instruction/input tokens so the model only trains on response tokens
        prefix_lengths = []
        for prefix in examples["prompt_prefix"]:
            if prefix:
                prefix_tokens = tokenizer(
                    prefix,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,
                    return_tensors=None,
                )
                prefix_lengths.append(len(prefix_tokens["input_ids"]))
            else:
                prefix_lengths.append(0)  # Completion format: no masking

        # Create labels: mask prompt prefix and padding tokens to -100
        labels = []
        for idx, input_ids in enumerate(tokenized["input_ids"]):
            prefix_len = prefix_lengths[idx]
            label = []
            for pos, token_id in enumerate(input_ids):
                if pos < prefix_len:
                    label.append(-100)  # Mask instruction/input tokens
                elif token_id == pad_token_id:
                    label.append(-100)  # Mask padding
                elif token_id >= vocab_size:
                    label.append(-100)  # Safety: out-of-vocab
                else:
                    label.append(token_id)
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    # Disable caching to ensure fresh tokenization
    if is_main:
        print("\n  Tokenizing dataset with label masking (cache disabled)...")
        print(f"  Only response tokens will contribute to loss (instruction/input masked)")
    else:
        # Suppress progress bars on worker processes
        import datasets
        datasets.disable_progress_bar()

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "prompt_prefix"],
        load_from_cache_file=False,  # Force re-tokenization
        desc="Tokenizing",
    )

    # Verify labels are correctly created
    if is_main:
        print(f"  Dataset columns: {tokenized_dataset.column_names}")
        sample = tokenized_dataset[0]
        num_masked = sum(1 for l in sample["labels"] if l == -100)
        num_train = sum(1 for l in sample["labels"] if l != -100)
        print(f"  Sample: {len(sample['labels'])} total tokens")
        print(f"    Masked (instruction+input+padding): {num_masked}")
        print(f"    Trainable (response): {num_train}")

    # Split into train/eval
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if is_main:
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Evaluation samples: {len(eval_dataset)}")

    # Create versioned output directory
    # In DDP mode, only the main process (local_rank 0) creates the directory
    # All processes must use the same output_dir path
    model_short = args.base_model.split("/")[-1].split("-")[0]

    marker = Path(args.output_dir) / ".current_ddp_run"

    if local_rank <= 0:
        # Main process or non-DDP: create the versioned directory
        # Delete stale marker from previous crashed runs BEFORE creating new dir
        if local_rank == 0 and marker.exists():
            marker.unlink()

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
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "quantization": args.quantize,
            "max_length": args.max_length,
            "started_at": datetime.now().isoformat(),
            "num_gpus": gpu_count if local_rank >= 0 else 1,
            "ddp": local_rank >= 0,
            "style_enhanced": getattr(args, 'style_enhanced', False),
        }

        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)

        # Atomic write: write to temp file then rename (rename is atomic on Linux)
        if local_rank == 0:
            tmp_marker = marker.with_suffix(".tmp")
            tmp_marker.write_text(str(output_dir))
            os.replace(str(tmp_marker), str(marker))  # atomic rename
    else:
        # DDP worker process: wait for main process to create directory
        import time
        for _ in range(60):
            if marker.exists():
                content = marker.read_text().strip()
                if content:  # Ensure file is non-empty (write completed)
                    output_dir = Path(content)
                    break
            time.sleep(0.5)
        else:
            raise RuntimeError("DDP worker: failed to get output directory from main process")
        print(f"  [rank {local_rank}] Using output directory: {output_dir}")

    # Synchronize all DDP processes before training starts
    # Must init process group explicitly here (TrainingArguments does it later, but we need it now)
    if local_rank >= 0:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier(device_ids=[local_rank])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=args.log_every,
        logging_strategy="steps",
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # Use bf16 instead of fp16 for better compatibility with 4-bit quantization
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        report_to="none",
        push_to_hub=False,
        # Gradient checkpointing already enabled by prepare_model_for_kbit_training
        gradient_checkpointing=False,
        # DDP settings for quantized LoRA models
        ddp_find_unused_parameters=False if local_rank >= 0 else None,
        # Disable dataloader multiprocessing to avoid potential issues
        dataloader_num_workers=0,
        disable_tqdm=args.disable_tqdm,
    )

    # Data collator - use default since we provide labels explicitly
    from transformers import default_data_collator
    data_collator = default_data_collator

    # Create trainer
    class ProgressCallback(TrainerCallback):
        def __init__(self, log_every: int, is_main_process: bool = True):
            self.log_every = max(1, log_every)
            self.is_main_process = is_main_process
            self._train_start = None
            self._step_times = []  # recent step durations for speed calc
            self._last_step_time = None
            self._eval_time_total = 0.0
            self._eval_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            import time
            self._train_start = time.time()
            self._last_step_time = time.time()

        def on_evaluate(self, args, state, control, **kwargs):
            import time
            self._eval_start = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            import time
            if self._eval_start and logs and "eval_loss" in logs:
                self._eval_time_total += time.time() - self._eval_start
                self._eval_start = None

        def on_step_end(self, args, state, control, **kwargs):
            import time
            now = time.time()
            if self._last_step_time:
                self._step_times.append(now - self._last_step_time)
                # Keep last 20 steps for moving average
                if len(self._step_times) > 20:
                    self._step_times = self._step_times[-20:]
            self._last_step_time = now

            if self.is_main_process and state.global_step % self.log_every == 0:
                max_steps = state.max_steps if state.max_steps else 0
                remaining = max_steps - state.global_step

                # Calculate speed and ETA
                eta_str = ""
                if self._step_times and remaining > 0:
                    avg_step = sum(self._step_times) / len(self._step_times)
                    speed = 1.0 / avg_step if avg_step > 0 else 0
                    eta_sec = remaining * avg_step
                    # Add estimated eval time (one eval every eval_steps)
                    eval_steps = args.eval_steps or 100
                    evals_remaining = remaining // eval_steps
                    if self._eval_time_total > 0 and state.global_step > 0:
                        avg_eval = self._eval_time_total  # total so far from single/few evals
                        evals_done = max(1, state.global_step // eval_steps)
                        avg_eval = self._eval_time_total / evals_done
                        eta_sec += evals_remaining * avg_eval

                    hours = int(eta_sec // 3600)
                    mins = int((eta_sec % 3600) // 60)
                    if hours > 0:
                        eta_str = f" | ETA: {hours}h{mins:02d}m | {avg_step:.1f}s/step"
                    else:
                        eta_str = f" | ETA: {mins}m | {avg_step:.1f}s/step"

                pct = f" ({100*state.global_step/max_steps:.0f}%)" if max_steps else ""
                print(f"[step {state.global_step}/{max_steps}]{pct}{eta_str}", flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback(args.log_every, is_main_process=(local_rank <= 0))],
    )

    # Train
    import time as _time
    _train_wall_start = _time.time()

    if local_rank <= 0:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        total_steps = trainer.state.max_steps if trainer.state.max_steps else len(train_dataset) * args.epochs // (args.batch_size * args.gradient_accumulation_steps * max(1, gpu_count))
        print(f"\n  Total steps: {total_steps}")
        print(f"  Epochs: {args.epochs}, Eval every: {training_args.eval_steps} steps")
        print(f"  Effective batch: {args.batch_size} × {gpu_count} GPU × {args.gradient_accumulation_steps} accum = {args.batch_size * gpu_count * args.gradient_accumulation_steps}")
        print("\nTraining progress will be shown below:")
        print("-" * 40)

    try:
        trainer.train()

        # Only main process handles saving and reporting
        if local_rank <= 0:
            _train_wall_end = _time.time()
            _duration = _train_wall_end - _train_wall_start
            _hours = int(_duration // 3600)
            _mins = int((_duration % 3600) // 60)
            _total_steps = trainer.state.global_step
            _avg_step = _duration / _total_steps if _total_steps > 0 else 0

            print("-" * 40)
            print(f"\nTraining complete!")
            print(f"  Duration: {_hours}h{_mins:02d}m ({_duration:.0f}s)")
            print(f"  Steps: {_total_steps}, Avg: {_avg_step:.1f}s/step")

            # Save model
            print(f"\nSaving model to: {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Save training metrics for visualization
            metrics_path = output_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(f"  Training metrics saved to: {metrics_path.name}")

            # Update metadata
            update_model_metadata(output_dir, {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": round(_duration),
                "duration_human": f"{_hours}h{_mins:02d}m",
                "total_steps": _total_steps,
                "avg_seconds_per_step": round(_avg_step, 1),
            })

            # Update registry
            registry = create_model_registry(args.output_dir)

            # Clean up DDP marker file
            if marker.exists():
                marker.unlink()

            # Auto-generate training plots
            if not args.no_visualize:
                try:
                    import subprocess
                    plot_script = Path(__file__).parent / "plot_training.py"
                    if plot_script.exists():
                        print(f"\n  Generating training plots...")
                        subprocess.run(
                            [sys.executable, str(plot_script), str(output_dir)],
                            check=False, capture_output=False,
                        )
                except Exception as e:
                    print(f"  Note: Visualization skipped ({e})")

            print("\n" + "=" * 60)
            print("Training Summary")
            print("=" * 60)
            print(f"  Model saved to: {output_dir}")
            print(f"  Total models in registry: {len(registry['models'])}")
            param_tuning_dir = output_dir / "Parameter_Tuning"
            if param_tuning_dir.exists():
                print(f"  Plots saved to: {param_tuning_dir}/")

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
        if local_rank <= 0:
            print("\n\nTraining interrupted by user.")
            update_model_metadata(output_dir, {
                "status": "interrupted",
                "interrupted_at": datetime.now().isoformat(),
            })
            if marker.exists():
                marker.unlink()
        return None

    except Exception as e:
        if local_rank <= 0:
            print(f"\nERROR: Training failed: {e}")
            update_model_metadata(output_dir, {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat(),
            })
            if marker.exists():
                marker.unlink()
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

  # Multi-GPU DDP (proper data-parallel training)
  accelerate launch --num_processes=2 scripts/finetune_lora.py --model mistral-nemo

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
        "--multi-gpu",
        action="store_true",
        help="[Deprecated] Use 'accelerate launch --num_processes=N' for multi-GPU DDP instead"
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N steps (default: 10)"
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bar (useful for log files)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip post-training plot generation"
    )
    parser.add_argument(
        "--style-enhanced",
        action="store_true",
        help="Style-enhanced mode: uses larger LoRA rank (32), alpha (64), "
             "longer context (4096 tokens), and expects context-window training data. "
             "Improves learning of transitions, argument structure, and cross-paragraph patterns."
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100). Set to steps-per-epoch "
             "for per-epoch checkpoints."
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep (default: 5)"
    )

    args = parser.parse_args()

    # Determine rank early to suppress duplicate output from DDP workers
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    _is_main = _local_rank <= 0

    if _is_main:
        print("\n" + "=" * 60)
        print("GSWA LoRA Fine-tuning Script")
        print("=" * 60)

    # Handle model selection
    if args.model:
        # Resolve alias if provided
        args.base_model = MODEL_ALIASES.get(args.model.lower(), args.model)
        if _is_main:
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

    # Apply style-enhanced mode overrides
    if args.style_enhanced:
        args.lora_r = 32
        args.lora_alpha = 64
        args.max_length = 4096
        args.epochs = max(args.epochs, 4)
        # Use context-window training data if default training data path
        if args.training_data == "./data/training/alpaca_train.jsonl":
            context_window_path = "./data/training/context-window_train.jsonl"
            if Path(context_window_path).exists():
                args.training_data = context_window_path
            else:
                if _is_main:
                    print(f"\n  Note: context-window data not found at {context_window_path}")
                    print(f"        Generate it with: python scripts/prepare_training_data.py "
                          f"--format context-window --section-aware --split")
                    print(f"        Falling back to: {args.training_data}")
        if _is_main:
            print(f"\n  Style-Enhanced Mode:")
            print(f"    LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
            print(f"    Max sequence length: {args.max_length}")
            print(f"    Epochs: {args.epochs}")
            print(f"    Training data: {args.training_data}")

    # Check dependencies (only on main process; workers share the same environment)
    if _is_main:
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

    # Check training data (only main process prints; workers share same filesystem)
    if _is_main:
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
