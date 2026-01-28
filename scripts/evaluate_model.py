#!/usr/bin/env python3
"""
Evaluate a trained GSWA LoRA model by generating text samples.

Loads the base model + LoRA adapter, runs inference on test prompts,
and saves results to Parameter_Tuning/ folder.

Usage:
    # Quick evaluation (5 samples)
    python scripts/evaluate_model.py models/gswa-lora-Mistral-20260123-012408/

    # More samples
    python scripts/evaluate_model.py models/gswa-lora-Mistral-20260123-012408/ --num-samples 10

    # Custom prompts file
    python scripts/evaluate_model.py models/gswa-lora-Mistral-20260123-012408/ --prompts-file my_prompts.jsonl

    # Custom max tokens
    python scripts/evaluate_model.py models/gswa-lora-Mistral-20260123-012408/ --max-new-tokens 512
"""
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime


def load_model(model_dir: Path, device: str = "cuda:0"):
    """Load base model + LoRA adapter with same quantization as training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Read training config
    config_file = model_dir / "training_config.json"
    if not config_file.exists():
        print(f"ERROR: training_config.json not found in {model_dir}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    base_model_name = config["base_model"]
    quantize = config.get("quantization", "4bit")

    print(f"\nLoading model for evaluation:")
    print(f"  Base: {base_model_name}")
    print(f"  Adapter: {model_dir.name}")
    print(f"  Quantization: {quantize}")
    print(f"  Device: {device}")

    # Setup quantization (same as training)
    bnb_config = None
    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),  # Use tokenizer saved with the adapter
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": device} if bnb_config else device,
        trust_remote_code=True,
        dtype=torch.bfloat16 if not bnb_config else None,
    )

    # Load LoRA adapter
    print(f"  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(model_dir))
    model.eval()

    print(f"  Model loaded successfully.")
    return model, tokenizer, config


def load_prompts(prompts_file: str, num_samples: int) -> list:
    """Load test prompts from a JSONL file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
                if len(prompts) >= num_samples:
                    break

    if not prompts:
        print(f"ERROR: No prompts found in {prompts_file}")
        sys.exit(1)

    return prompts


def format_prompt(example: dict, tokenizer=None) -> str:
    """Format an example using the model's native chat template.

    This ensures inference format matches training format exactly.
    """
    if "instruction" in example:
        user_content = example['instruction']
        if example.get("input"):
            user_content += f"\n\n{example['input']}"
    elif "conversations" in example:
        conv = example["conversations"]
        user_content = conv[0]['value']
    elif "text" in example:
        # Use first half as prompt
        text = example["text"]
        mid = len(text) // 2
        user_content = text[:mid]
    else:
        user_content = str(example)

    # Use tokenizer's chat template if available
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback to Alpaca format if no tokenizer
        prompt = f"### Instruction:\n{user_content}\n\n### Response:\n"

    return prompt


def generate_samples(model, tokenizer, prompts: list, max_new_tokens: int = 256,
                     temperature: float = 0.7) -> list:
    """Generate text for each prompt."""
    import torch

    results = []
    total = len(prompts)

    for i, prompt_data in enumerate(prompts, 1):
        instruction = prompt_data.get("instruction", "")
        input_text = prompt_data.get("input", "")
        reference = prompt_data.get("output", "")

        formatted_prompt = format_prompt(prompt_data, tokenizer=tokenizer)

        # Truncate input to avoid OOM (keep instruction + first 500 chars of input)
        if len(formatted_prompt) > 2000:
            truncated_content = instruction
            if input_text:
                truncated_content += f"\n\n{input_text[:500]}..."
            formatted_prompt = format_prompt(
                {"instruction": truncated_content}, tokenizer=tokenizer
            )

        print(f"  [{i}/{total}] Generating...", end="", flush=True)

        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True,
                          max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        print(f" ({len(generated)} chars)")

        results.append({
            "instruction": instruction[:200],
            "input": input_text[:200] if input_text else "",
            "reference": reference[:500] if reference else "",
            "generated": generated,
            "prompt_length": len(formatted_prompt),
            "generated_length": len(generated),
        })

    return results


def write_results(results: list, config: dict, output_dir: Path):
    """Write evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable text file
    txt_path = output_dir / "eval_results.txt"
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GSWA Model Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {config.get('base_model', 'unknown')}\n")
        f.write(f"LoRA: r={config.get('lora_r', '?')}, alpha={config.get('lora_alpha', '?')}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {len(results)}\n\n")

        for i, r in enumerate(results, 1):
            f.write("=" * 70 + "\n")
            f.write(f"  Sample {i}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Instruction:\n  {r['instruction']}\n\n")
            if r['input']:
                # Show first 300 chars of input
                input_display = r['input'][:300]
                if len(r['input']) > 300:
                    input_display += "..."
                f.write(f"Input (truncated):\n  {input_display}\n\n")
            f.write(f"Generated ({r['generated_length']} chars):\n")
            f.write("-" * 40 + "\n")
            f.write(r['generated'] + "\n")
            f.write("-" * 40 + "\n\n")
            if r.get("reference"):
                ref_display = r['reference'][:500]
                if len(r['reference']) > 500:
                    ref_display += "..."
                f.write(f"Reference (truncated):\n  {ref_display}\n\n")

    # Machine-readable JSON
    json_path = output_dir / "eval_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "model": config.get("base_model", "unknown"),
            "adapter": config.get("model_short", "unknown"),
            "lora_r": config.get("lora_r"),
            "lora_alpha": config.get("lora_alpha"),
            "evaluated_at": datetime.now().isoformat(),
            "num_samples": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved:")
    print(f"    {txt_path}")
    print(f"    {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GSWA LoRA model on test prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path,
                        help="Path to trained model directory")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of test samples to generate (default: 5)")
    parser.add_argument("--prompts-file", type=str,
                        default="./data/training/alpaca_val.jsonl",
                        help="Path to prompts file (JSONL format)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate per sample (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference (default: cuda:0)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")

    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)

    # Load prompts
    print(f"\nLoading {args.num_samples} test prompts from: {args.prompts_file}")
    prompts = load_prompts(args.prompts_file, args.num_samples)
    print(f"  Loaded {len(prompts)} prompts")

    # Load model
    model, tokenizer, config = load_model(args.model_dir, args.device)

    # Generate samples
    print(f"\nGenerating {len(prompts)} samples (max_tokens={args.max_new_tokens}, temp={args.temperature}):")
    results = generate_samples(model, tokenizer, prompts,
                              max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature)

    # Save results
    output_dir = args.output_dir or args.model_dir / "Parameter_Tuning"
    write_results(results, config, output_dir)

    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()
