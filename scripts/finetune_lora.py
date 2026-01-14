#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for GSWA.

This script fine-tunes a base model on Gilles's writing style using LoRA.
Supports multiple backends and configurations.

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes

Usage:
    # Basic usage (Linux with NVIDIA GPU)
    python scripts/finetune_lora.py --base-model mistralai/Mistral-7B-Instruct-v0.2

    # With specific training data
    python scripts/finetune_lora.py --training-data ./data/training/alpaca.jsonl

    # Mac with MLX (Apple Silicon)
    python scripts/finetune_lora.py --backend mlx --base-model mlx-community/Mistral-7B-Instruct-v0.2

    # 4-bit quantized (lower memory)
    python scripts/finetune_lora.py --quantize 4bit
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import peft
    except ImportError:
        missing.append("peft")

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install torch transformers peft datasets accelerate bitsandbytes")
        sys.exit(1)


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

    # Training arguments
    output_dir = Path(args.output_dir) / f"gswa-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

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
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    print(f"\nSaving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config
    config = {
        "base_model": args.base_model,
        "training_data": args.training_data,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "quantization": args.quantize,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nTo use the fine-tuned model:")
    print(f"  1. Copy adapter to Ollama (Mac):")
    print(f"     python scripts/convert_lora_to_ollama.py --input {output_dir}")
    print(f"  2. Or load directly in Python:")
    print(f"     from peft import PeftModel")
    print(f"     model = PeftModel.from_pretrained(base_model, '{output_dir}')")

    return str(output_dir)


def train_with_mlx(args):
    """Train using MLX (for Mac Apple Silicon)."""
    print("\n" + "=" * 60)
    print("GSWA LoRA Fine-tuning with MLX (Apple Silicon)")
    print("=" * 60)

    try:
        import mlx
        import mlx.core as mx
        from mlx_lm import load, generate
    except ImportError:
        print("\nMLX not installed. Install with:")
        print("  pip install mlx mlx-lm")
        sys.exit(1)

    print("\nMLX fine-tuning is best done with mlx-lm CLI:")
    print(f"""
    # Install mlx-lm
    pip install mlx-lm

    # Prepare data in MLX format
    python scripts/prepare_training_data.py --format completion --output ./data/training/

    # Run fine-tuning
    mlx_lm.lora \\
        --model {args.base_model} \\
        --train \\
        --data ./data/training/ \\
        --batch-size 4 \\
        --lora-layers 16 \\
        --iters 1000

    # The adapter will be saved to ./adapters/
    """)

    return None


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for GSWA")
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--training-data",
        default="./data/training/alpaca.jsonl",
        help="Training data file"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "mlx"],
        default="transformers",
        help="Training backend"
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization level"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    if args.backend == "mlx":
        train_with_mlx(args)
    else:
        train_with_transformers(args)


if __name__ == "__main__":
    main()
