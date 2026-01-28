#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 Fine-tuning Script for GSWA.

This script fine-tunes large models (70B+) using DeepSpeed ZeRO-3 for proper
multi-GPU training. Unlike device_map sharding, DeepSpeed properly handles
gradient synchronization and optimizer states across GPUs.

IMPORTANT: This script uses bf16 precision with CPU offloading, NOT 4-bit
quantization. bitsandbytes 4-bit quantization is incompatible with DeepSpeed
ZeRO-3. Memory efficiency is achieved through ZeRO-3 partitioning + CPU
offloading of optimizer states and parameters.

Requirements:
    - 2+ NVIDIA GPUs (for multi-GPU training)
    - Large CPU RAM (recommended: 256GB+ for 70B models)
    - DeepSpeed: pip install deepspeed accelerate

Usage:
    # With accelerate (recommended)
    accelerate launch --config_file config/accelerate_deepspeed.yaml \\
        scripts/finetune_deepspeed.py --model meta-llama/Llama-3.3-70B-Instruct

    # Direct with DeepSpeed
    deepspeed --num_gpus=2 scripts/finetune_deepspeed.py \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --deepspeed config/deepspeed_zero3.json
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from transformers.integrations import HfDeepSpeedConfig
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset

# Model aliases
MODEL_ALIASES = {
    "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
}


def load_jsonl(path):
    """Load JSONL training data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_example(example):
    """Format training example to text."""
    if "instruction" in example:
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example.get("input"):
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n{example['output']}"
    elif "conversations" in example:
        conv = example["conversations"]
        prompt = f"### Human:\n{conv[0]['value']}\n\n### Assistant:\n{conv[1]['value']}"
    elif "text" in example:
        prompt = example["text"]
    else:
        prompt = str(example)
    return {"text": prompt}


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 fine-tuning for large models")
    parser.add_argument("--model", type=str, required=True, help="Model name or alias")
    parser.add_argument("--training-data", type=str, default="./data/training/alpaca_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="./models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Note: 4-bit quantization is NOT compatible with DeepSpeed ZeRO-3
    # DeepSpeed handles memory via ZeRO partitioning + CPU offloading instead
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    # Resolve model alias
    model_name = MODEL_ALIASES.get(args.model.lower(), args.model)

    print("\n" + "=" * 60)
    print("GSWA DeepSpeed Fine-tuning")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print("Using bf16 with DeepSpeed ZeRO-3 + CPU offloading")
    print("(4-bit quantization is NOT compatible with DeepSpeed ZeRO-3)")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer_kwargs = {}
    if "mistral" in model_name.lower():
        tokenizer_kwargs["fix_mistral_regex"] = True
        print("  Applying Mistral tokenizer regex fix (fix_mistral_regex=True)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # Initialize HfDeepSpeedConfig BEFORE loading model for ZeRO-3 lazy init
    # This enables parameter partitioning during model loading, avoiding OOM
    ds_config_path = "config/deepspeed_zero3.json"
    dschf = HfDeepSpeedConfig(ds_config_path)  # Must keep this object alive!
    print(f"\nZeRO-3 lazy init enabled via HfDeepSpeedConfig")

    # Load model with ZeRO-3 lazy initialization
    # Parameters are partitioned across GPUs during loading, not after
    print("Loading model with ZeRO-3 parameter partitioning...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

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
    raw_data = load_jsonl(args.training_data)
    print(f"  Loaded {len(raw_data)} examples")

    formatted_data = [format_example(ex) for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors=None,
        )
        # Create labels with padding masked
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = [
                -100 if (tid == pad_token_id or tid >= vocab_size) else tid
                for tid in input_ids
            ]
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    print("  Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )

    # Split
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_short = model_name.split("/")[-1].split("-")[0]
    output_dir = Path(args.output_dir) / f"gswa-lora-{model_short}-ds-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output: {output_dir}")

    # Training arguments with native DeepSpeed integration
    # Pass deepspeed config to Trainer to handle ZeRO-3 initialization properly
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
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed="config/deepspeed_zero3.json",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    try:
        trainer.train()
        print("\nTraining complete!")

        # Save
        print(f"Saving model to: {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save config
        config = {
            "base_model": model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "training_method": "DeepSpeed ZeRO-3 + CPU offloading",
            "precision": "bf16",
            "completed_at": datetime.now().isoformat(),
        }
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Training Summary")
        print("=" * 60)
        print(f"  Model saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
