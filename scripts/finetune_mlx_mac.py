#!/usr/bin/env python3
"""
MLX Fine-tuning Script for Mac Apple Silicon.

This is optimized for Mac M1/M2/M3 chips using Apple's MLX framework.
MLX is designed for efficient training on Apple Silicon.

Requirements:
    pip install mlx mlx-lm

Usage:
    python scripts/finetune_mlx_mac.py --model mistral --epochs 3

After training:
    # Create Ollama modelfile
    python scripts/create_ollama_modelfile.py

    # Import to Ollama
    ollama create gswa-gilles -f Modelfile
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def check_mlx():
    """Check if MLX is available."""
    try:
        import mlx
        import mlx.core as mx
        print(f"MLX version: {mlx.__version__}")
        print(f"Device: {mx.default_device()}")
        return True
    except ImportError:
        return False


def prepare_mlx_data(input_file: str, output_dir: str):
    """Prepare data in MLX format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

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
        with open(output_dir / f"{name}.jsonl", 'w') as f:
            for item in subset:
                formatted = format_for_mlx(item)
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')

    print(f"Prepared {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test examples")
    return str(output_dir)


def run_mlx_training(args):
    """Run MLX LoRA training."""
    print("\n" + "=" * 60)
    print("GSWA MLX Fine-tuning (Apple Silicon)")
    print("=" * 60)

    # Model mapping
    model_map = {
        "mistral": "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        "llama2": "mlx-community/Llama-2-7b-chat-mlx-4bit",
        "phi": "mlx-community/phi-2-4bit",
        "gemma": "mlx-community/gemma-7b-it-4bit",
    }

    model_id = model_map.get(args.model, args.model)
    print(f"\nUsing model: {model_id}")

    # Prepare training data
    print("\nPreparing training data...")
    data_dir = prepare_mlx_data(args.training_data, "./data/training/mlx")

    # Create output directory
    output_dir = Path(args.output_dir) / f"gswa-mlx-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mlx_lm command
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_id,
        "--train",
        "--data", data_dir,
        "--batch-size", str(args.batch_size),
        "--lora-layers", str(args.lora_layers),
        "--iters", str(args.iters),
        "--adapter-path", str(output_dir / "adapters"),
    ]

    print("\nRunning MLX training...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        subprocess.run(cmd, check=True)
        print("-" * 60)
        print("\nTraining complete!")

        # Save config
        config = {
            "base_model": model_id,
            "training_data": args.training_data,
            "lora_layers": args.lora_layers,
            "batch_size": args.batch_size,
            "iters": args.iters,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nOutput saved to: {output_dir}")

        # Create Ollama Modelfile
        create_ollama_modelfile(output_dir, model_id)

        return str(output_dir)

    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")
        return None


def create_ollama_modelfile(output_dir: Path, base_model: str):
    """Create Ollama Modelfile for the fine-tuned adapter."""
    modelfile_content = f'''# GSWA Gilles-Style Model
# Created from fine-tuning on Gilles's corpus

FROM {base_model.split("/")[-1]}

# Apply the LoRA adapter
ADAPTER {output_dir}/adapters

# System prompt for Gilles-style writing
SYSTEM """You are a scientific writing assistant trained on the papers of Prof. Gilles van Wezel.
Your writing style is characterized by:
- Clear, precise scientific prose
- Logical flow from background to conclusions
- Careful qualification of claims
- Rich vocabulary while maintaining accessibility
- Smooth transitions between concepts

When rewriting text, maintain the original scientific meaning while improving clarity and style."""

# Parameters optimized for scientific writing
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
'''

    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"\nOllama Modelfile created: {modelfile_path}")
    print("\nTo create the Ollama model:")
    print(f"  ollama create gswa-gilles -f {modelfile_path}")
    print("\nThen update your .env:")
    print("  VLLM_MODEL_NAME=gswa-gilles")


def main():
    parser = argparse.ArgumentParser(description="MLX fine-tuning for Mac")
    parser.add_argument(
        "--model",
        default="mistral",
        help="Model to fine-tune (mistral, llama2, phi, gemma, or full HF path)"
    )
    parser.add_argument(
        "--training-data",
        default="./data/training/alpaca.jsonl",
        help="Training data file"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of LoRA layers"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Training iterations"
    )

    args = parser.parse_args()

    # Check MLX
    if not check_mlx():
        print("\nMLX not installed. Install with:")
        print("  pip install mlx mlx-lm")
        print("\nNote: MLX only works on Mac with Apple Silicon (M1/M2/M3)")
        sys.exit(1)

    # Check if training data exists
    if not Path(args.training_data).exists():
        print(f"\nTraining data not found: {args.training_data}")
        print("\nGenerate training data first:")
        print("  python scripts/prepare_training_data.py --format alpaca --weighted")
        sys.exit(1)

    # Run training
    run_mlx_training(args)


if __name__ == "__main__":
    main()
