"""
Unified Training CLI - Main entrypoint for GSWA fine-tuning.

Usage:
    python -m gswa.train --config configs/run.yaml --auto-plan --preprocess
    python -m gswa.train --auto  # Full auto mode
    python -m gswa.train --info  # Show hardware info
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gswa.training.hardware import HardwareDetector, HardwareInfo
from gswa.training.run_manager import RunManager, RunConfig
from gswa.training.preprocessor import DataPreprocessor, PreprocessStats
from gswa.training.planner import TrainingPlanner, select_best_plan
from gswa.training.logger import TrainingLogger
from gswa.training.visualizer import TrainingVisualizer, generate_training_report


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"


def print_banner():
    """Print GSWA banner."""
    print("\n" + "=" * 60)
    print("  GSWA Fine-tuning Pipeline")
    print("  Cross-Platform: Apple Silicon (MLX) + NVIDIA (CUDA)")
    print("=" * 60)


def cmd_info(args):
    """Show hardware information."""
    print_banner()

    detector = HardwareDetector()
    info = detector.detect()
    detector.print_summary()

    if args.json:
        print("\nJSON output:")
        print(info.to_json())


def cmd_preprocess(args):
    """Preprocess training data."""
    print_banner()

    input_file = args.input or str(DATA_DIR / "training" / "alpaca_train.jsonl")
    output_file = args.output or input_file.replace('.jsonl', '.preprocessed.jsonl')

    print(f"\nInput: {input_file}")
    print(f"Output: {output_file}")
    print(f"Max tokens: {args.max_tokens}")

    if not Path(input_file).exists():
        print(f"\nError: Input file not found: {input_file}")
        print("Generate training data first: make prepare-training")
        return 1

    preprocessor = DataPreprocessor(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        split_strategy=args.strategy,
    )

    if args.analyze_only:
        stats = preprocessor.analyze_data(input_file)
        print(preprocessor.generate_report(stats))
        return 0

    print("\nProcessing...")
    stats = preprocessor.preprocess(input_file, output_file)

    # Save reports
    if args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        stats.save(str(report_dir / "preprocess_stats.json"))
        preprocessor.save_report(str(report_dir / "preprocess_report.md"), stats)
        print(f"\nReports saved to: {report_dir}")

    print(preprocessor.generate_report(stats))
    return 0


def cmd_plan(args):
    """Run training plan selection."""
    print_banner()

    # Detect hardware
    detector = HardwareDetector()
    hw_info = detector.detect()

    print(f"\nDetected: {hw_info.chip_name}")
    print(f"Memory: {hw_info.total_memory_gb:.1f} GB")

    # Get memory for planning
    if hw_info.chip_type == "apple_silicon":
        available_mem = hw_info.total_memory_gb
    elif hw_info.cuda_available:
        available_mem = hw_info.cuda_device_memory_gb
    else:
        available_mem = hw_info.total_memory_gb * 0.5

    training_data = args.data or str(DATA_DIR / "training" / "alpaca_train.jsonl")

    if not Path(training_data).exists():
        print(f"\nError: Training data not found: {training_data}")
        return 1

    # Run planner
    best_plan, result = select_best_plan(
        model_id=args.model,
        training_data=training_data,
        available_memory_gb=available_mem,
        memory_margin=args.margin,
        dry_run_steps=args.dry_run_steps,
        max_candidates=args.max_candidates,
        skip_dry_run=args.skip_dry_run,
        seed=args.seed,
    )

    # Save results
    if args.output:
        result.save(args.output)
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_train(args):
    """Run full training pipeline."""
    print_banner()

    # Initialize run manager
    run_manager = RunManager(str(RUNS_DIR))

    # Load or create config
    if args.config:
        config = RunConfig.from_file(args.config)
        print(f"\nLoaded config from: {args.config}")
    else:
        config = RunConfig()

    # Detect hardware
    print("\n" + "-" * 60)
    print("Detecting Hardware")
    print("-" * 60)

    detector = HardwareDetector()
    hw_info = detector.detect()
    config.hardware_info = hw_info.to_dict()

    print(f"  Chip: {hw_info.chip_name}")
    print(f"  Memory: {hw_info.total_memory_gb:.1f} GB")
    print(f"  Backend: {hw_info.recommended_backend}")

    # Override with command line args
    if args.model:
        config.model_id = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.iters:
        config.iters = args.iters
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.seed:
        config.seed = args.seed

    # Apply auto-detected settings if no explicit config
    if not args.config and not args.no_auto:
        print("\n  Applying hardware-optimized settings...")
        config.batch_size = hw_info.recommended_batch_size
        config.max_seq_length = hw_info.recommended_max_seq_length
        config.num_layers = hw_info.recommended_num_layers
        config.eval_batch_size = hw_info.recommended_eval_batch_size

    # Training data
    training_data = args.data or str(DATA_DIR / "training" / "alpaca_train.jsonl")
    config.training_data = training_data

    if not Path(training_data).exists():
        print(f"\nError: Training data not found: {training_data}")
        print("Generate training data first: make prepare-training")
        return 1

    # Preprocessing
    if args.preprocess:
        print("\n" + "-" * 60)
        print("Preprocessing Data")
        print("-" * 60)

        preprocessor = DataPreprocessor(
            max_tokens=config.max_seq_length,
            overlap_tokens=config.chunk_overlap,
        )

        # Analyze first
        stats = preprocessor.analyze_data(training_data)
        print(f"  Total entries: {stats.total_entries_before}")
        print(f"  Max tokens: {stats.max_tokens_before}")
        print(f"  Over limit ({config.max_seq_length}): {stats.truncation_pct_before:.1f}%")

        if stats.truncation_pct_before > 1:
            output_file = training_data.replace('.jsonl', '.preprocessed.jsonl')
            stats = preprocessor.preprocess(training_data, output_file)
            config.preprocessed_data = output_file
            config.training_data = output_file
            print(f"  Preprocessed to: {output_file}")
            print(f"  New entries: {stats.total_entries_after}")
            print(f"  Max tokens after: {stats.max_tokens_after}")
        else:
            print("  Data is within limits, no preprocessing needed")

    # Auto-plan
    if args.auto_plan and hw_info.chip_type == "apple_silicon":
        print("\n" + "-" * 60)
        print("Running Training Planner")
        print("-" * 60)

        available_mem = hw_info.total_memory_gb
        best_plan, plan_result = select_best_plan(
            model_id=config.model_id,
            training_data=config.training_data,
            available_memory_gb=available_mem,
            memory_margin=0.8,
            dry_run_steps=args.plan_steps or 10,
            max_candidates=5,
            skip_dry_run=args.skip_plan_dry_run,
            seed=config.seed,
        )

        if best_plan:
            config.batch_size = best_plan.batch_size
            config.max_seq_length = best_plan.max_seq_length
            config.num_layers = best_plan.num_layers
            config.grad_accum_steps = best_plan.grad_accum_steps
            config.eval_batch_size = best_plan.eval_batch_size
            config.lora_rank = best_plan.lora_rank

    # Create run
    print("\n" + "-" * 60)
    print("Creating Run")
    print("-" * 60)

    run_dir = run_manager.create_run(
        config,
        run_name=args.name,
        description=args.description or "Training run",
    )
    print(f"  Run directory: {run_dir}")

    # Print config summary
    run_manager.print_config_summary()

    # Save preprocess stats if available
    if args.preprocess and 'stats' in dir():
        stats.save(str(run_dir / "stats" / "preprocess_stats.json"))

    # Confirm start
    if not args.yes:
        response = input("\nStart training? [Y/n]: ")
        if response.lower() == 'n':
            print("Cancelled.")
            return 0

    # Run training
    print("\n" + "-" * 60)
    print("Starting Training")
    print("-" * 60)

    run_manager.update_status("training", "Training started")

    # Build training command
    if hw_info.recommended_backend == "mlx":
        success = run_mlx_training(config, run_dir, run_manager, args)
    else:
        success = run_lora_training(config, run_dir, run_manager, args)

    if success:
        run_manager.update_status("completed", "Training completed successfully")

        # Generate visualizations
        if not args.skip_plots:
            print("\n" + "-" * 60)
            print("Generating Reports")
            print("-" * 60)

            try:
                report_path = generate_training_report(
                    str(run_dir / "logs"),
                    str(run_dir / "reports"),
                    config.to_dict(),
                )
                print(f"  Report: {report_path}")
            except Exception as e:
                print(f"  Warning: Could not generate plots: {e}")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nRun directory: {run_dir}")
        print(f"Adapters: {run_dir / 'adapters'}")
        print(f"Report: {run_dir / 'reports' / 'report.html'}")

        return 0
    else:
        run_manager.update_status("failed", "Training failed")
        print("\n" + "=" * 60)
        print("Training Failed")
        print("=" * 60)
        return 1


def run_mlx_training(
    config: RunConfig,
    run_dir: Path,
    run_manager: RunManager,
    args,
) -> bool:
    """Run MLX training with OOM handling."""
    max_retries = 3 if config.enable_oom_fallback else 1

    for attempt in range(max_retries):
        # Prepare data directory (MLX expects train.jsonl, valid.jsonl, test.jsonl)
        data_dir = run_dir / "data"
        prepare_mlx_data(config.training_data, data_dir)

        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", config.model_id,
            "--train",
            "--data", str(data_dir),
            "--batch-size", str(config.batch_size),
            "--num-layers", str(config.num_layers),
            "--iters", str(config.iters),
            "--learning-rate", str(config.learning_rate),
            "--max-seq-length", str(config.max_seq_length),
            "--adapter-path", str(run_dir / "adapters"),
        ]

        print(f"\n  Attempt {attempt + 1}/{max_retries}")
        print(f"  Command: {' '.join(cmd[:6])}...")
        print("-" * 40)

        # Run with output capture for logging
        log_file = run_dir / "logs" / "training_output.log"

        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Stream and log output
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)

                process.wait()

            if process.returncode == 0:
                return True

            # Check for OOM in output
            with open(log_file, 'r') as f:
                output = f.read().lower()

            if "memory" in output or "oom" in output:
                if attempt < max_retries - 1:
                    print(f"\n  OOM detected, attempting fallback...")
                    new_config = run_manager.apply_oom_fallback("Metal OOM error detected")
                    if new_config:
                        config = new_config
                        continue
            else:
                print(f"\n  Training failed with exit code {process.returncode}")
                break

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            run_manager.update_status("interrupted")
            return False

        except Exception as e:
            print(f"\n  Error: {e}")
            break

    return False


def run_lora_training(
    config: RunConfig,
    run_dir: Path,
    run_manager: RunManager,
    args,
) -> bool:
    """Run LoRA training for CUDA with OOM fallback and visualization."""
    from gswa.training.cuda_trainer import CUDATrainer, CUDATrainingConfig

    # Create CUDA training config from RunConfig
    cuda_config = CUDATrainingConfig(
        model_id=config.model_id,
        training_data=config.training_data,
        output_dir=str(run_dir.parent),
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        max_seq_length=config.max_seq_length,
        learning_rate=config.learning_rate,
        lora_r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        gradient_accumulation_steps=config.grad_accum_steps,
        preprocess_enabled=False,  # Already preprocessed in cmd_train
        enable_oom_fallback=config.enable_oom_fallback,
        run_name=run_dir.name,
    )

    # Create trainer and run
    trainer = CUDATrainer(cuda_config)

    print(f"\n  Model: {cuda_config.model_id}")
    print(f"  Batch size: {cuda_config.batch_size}")
    print(f"  Sequence length: {cuda_config.max_seq_length}")
    print(f"  Quantization: {cuda_config.quantize}")
    print(f"  OOM fallback: {'enabled' if cuda_config.enable_oom_fallback else 'disabled'}")
    print("-" * 40)

    # Run training subprocess directly for simplicity
    script = SCRIPTS_DIR / "finetune_lora.py"

    if not script.exists():
        print(f"Error: Training script not found: {script}")
        return False

    max_retries = 3 if config.enable_oom_fallback else 1
    current_config = cuda_config

    for attempt in range(max_retries):
        cmd = [
            sys.executable, str(script),
            "--base-model", current_config.model_id,
            "--training-data", current_config.training_data,
            "--output-dir", str(run_dir / "adapters"),
            "--batch-size", str(current_config.batch_size),
            "--max-length", str(current_config.max_seq_length),
            "--learning-rate", str(current_config.learning_rate),
            "--lora-r", str(current_config.lora_r),
            "--lora-alpha", str(current_config.lora_alpha),
            "--gradient-accumulation-steps", str(current_config.gradient_accumulation_steps),
            "--quantize", current_config.quantize,
        ]

        print(f"\n  Attempt {attempt + 1}/{max_retries}")
        print(f"  batch_size={current_config.batch_size}, "
              f"seq_len={current_config.max_seq_length}, "
              f"grad_accum={current_config.gradient_accumulation_steps}")

        # Run with output logging
        log_file = run_dir / "logs" / "training_output.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Stream output
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)

                process.wait()

            if process.returncode == 0:
                return True

            # Check for OOM
            with open(log_file, 'r') as f:
                output = f.read().lower()

            if "cuda out of memory" in output or "oom" in output:
                if attempt < max_retries - 1:
                    print(f"\n  CUDA OOM detected, applying fallback...")

                    # Apply fallback - halve batch size or reduce sequence length
                    if current_config.batch_size > 1:
                        current_config.batch_size = max(1, current_config.batch_size // 2)
                        current_config.gradient_accumulation_steps *= 2
                    elif current_config.max_seq_length > 512:
                        current_config.max_seq_length = int(current_config.max_seq_length * 0.75)

                    run_manager.log_event("oom_fallback", {
                        "attempt": attempt + 1,
                        "new_batch_size": current_config.batch_size,
                        "new_seq_length": current_config.max_seq_length,
                    })
                    continue
            else:
                print(f"\n  Training failed with exit code {process.returncode}")
                break

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            return False

        except Exception as e:
            print(f"\n  Error: {e}")
            break

    return False


# Standard ML data split ratios (configurable)
DEFAULT_TRAIN_RATIO = 0.80  # 80% for training
DEFAULT_VALID_RATIO = 0.10  # 10% for validation
DEFAULT_TEST_RATIO = 0.10   # 10% for testing


def prepare_mlx_data(
    input_file: str,
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
):
    """Prepare data in MLX format (train/valid/test splits).

    Standard ML best practice: 80/10/10 split for train/valid/test.
    - Training set: Used to train the model
    - Validation set: Used for hyperparameter tuning and early stopping
    - Test set: Held out for final evaluation (never seen during training)

    Args:
        input_file: Path to input JSONL file
        output_dir: Output directory for split files
        train_ratio: Fraction for training (default 0.80)
        valid_ratio: Fraction for validation (default 0.10)
        test_ratio: Fraction for testing (default 0.10)
        seed: Random seed for reproducibility
    """
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate ratios
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        print(f"  Warning: Split ratios sum to {total:.2f}, normalizing...")
        train_ratio /= total
        valid_ratio /= total
        test_ratio /= total

    # Load all data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(data)

    # Split according to ratios
    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    splits = {
        "train": data[:train_end],
        "valid": data[train_end:valid_end],
        "test": data[valid_end:],
    }

    # Format for MLX (expects "text" field)
    def format_entry(entry):
        if "instruction" in entry:
            text = f"### Instruction:\n{entry['instruction']}\n"
            if entry.get("input"):
                text += f"### Input:\n{entry['input']}\n"
            text += f"### Response:\n{entry['output']}"
            return {"text": text}
        return entry

    # Save splits
    for name, entries in splits.items():
        with open(output_dir / f"{name}.jsonl", 'w') as f:
            for entry in entries:
                formatted = format_entry(entry)
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')

    # Print split statistics
    print(f"\n  Data Split (seed={seed}):")
    print(f"    Train: {len(splits['train']):,} samples ({train_ratio*100:.0f}%)")
    print(f"    Valid: {len(splits['valid']):,} samples ({valid_ratio*100:.0f}%)")
    print(f"    Test:  {len(splits['test']):,} samples ({test_ratio*100:.0f}%)")
    print(f"    Total: {n:,} samples")

    # Save split info for reproducibility
    split_info = {
        "seed": seed,
        "total_samples": n,
        "train_samples": len(splits['train']),
        "valid_samples": len(splits['valid']),
        "test_samples": len(splits['test']),
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
    }
    with open(output_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    return str(output_dir), split_info


def cmd_visualize(args):
    """Generate visualizations from existing run."""
    print_banner()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    # Load config
    config = None
    config_file = run_dir / "config" / "run_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Generate report
    report_path = generate_training_report(
        str(run_dir / "logs"),
        str(run_dir / "reports"),
        config,
    )

    print(f"\nReport generated: {report_path}")
    return 0


def cmd_resume(args):
    """Resume a training run."""
    print_banner()

    run_manager = RunManager(str(RUNS_DIR))

    try:
        config = run_manager.resume_run(args.run_dir)
        print(f"\nResumed run: {args.run_dir}")
        run_manager.print_config_summary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Continue training logic would go here
    print("\nResume functionality: TODO")
    return 0


def cmd_list(args):
    """List recent training runs."""
    print_banner()

    run_manager = RunManager(str(RUNS_DIR))
    runs = run_manager.list_runs(limit=args.limit)

    if not runs:
        print("\nNo runs found.")
        return 0

    print(f"\nRecent runs ({len(runs)}):")
    print("-" * 80)
    print(f"{'ID':<30} {'Status':<12} {'Created':<20} {'Model':<20}")
    print("-" * 80)

    for run in runs:
        created = run.get("created_at", "")[:19]
        model = run.get("model_id", "")[:20]
        print(f"{run.get('run_id', ''):<30} {run.get('status', ''):<12} {created:<20} {model:<20}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="GSWA MLX Fine-tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show hardware info
  python -m gswa.train info

  # Auto-mode training
  python -m gswa.train train --auto

  # Training with preprocessing and auto-plan
  python -m gswa.train train --preprocess --auto-plan

  # Preprocess data only
  python -m gswa.train preprocess --max-tokens 2048

  # Run training planner
  python -m gswa.train plan --max-candidates 5

  # Generate visualizations from existing run
  python -m gswa.train visualize --run-dir runs/20240115-123456

  # List recent runs
  python -m gswa.train list
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show hardware information")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess training data")
    preprocess_parser.add_argument("--input", "-i", help="Input JSONL file")
    preprocess_parser.add_argument("--output", "-o", help="Output JSONL file")
    preprocess_parser.add_argument("--max-tokens", "-m", type=int, default=2048)
    preprocess_parser.add_argument("--overlap", type=int, default=50)
    preprocess_parser.add_argument("--strategy", choices=["auto", "paragraph", "sentence", "token"], default="auto")
    preprocess_parser.add_argument("--analyze-only", action="store_true")
    preprocess_parser.add_argument("--report-dir", help="Directory for reports")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Run training planner")
    plan_parser.add_argument("--model", default="mlx-community/Mistral-7B-Instruct-v0.2-4bit")
    plan_parser.add_argument("--data", help="Training data file")
    plan_parser.add_argument("--margin", type=float, default=0.8)
    plan_parser.add_argument("--dry-run-steps", type=int, default=10)
    plan_parser.add_argument("--max-candidates", type=int, default=5)
    plan_parser.add_argument("--skip-dry-run", action="store_true")
    plan_parser.add_argument("--seed", type=int, default=42)
    plan_parser.add_argument("--output", "-o", help="Save results to JSON")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", "-c", help="Config YAML/JSON file")
    train_parser.add_argument("--model", help="Model ID")
    train_parser.add_argument("--data", help="Training data file")
    train_parser.add_argument("--name", help="Run name")
    train_parser.add_argument("--description", help="Run description")
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--max-seq-length", type=int)
    train_parser.add_argument("--num-layers", type=int)
    train_parser.add_argument("--iters", type=int)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--preprocess", action="store_true", help="Preprocess data if needed")
    train_parser.add_argument("--auto-plan", action="store_true", help="Run planner to select best config")
    train_parser.add_argument("--plan-steps", type=int, default=10, help="Steps for planner dry-runs")
    train_parser.add_argument("--skip-plan-dry-run", action="store_true")
    train_parser.add_argument("--no-auto", action="store_true", help="Don't auto-detect settings")
    train_parser.add_argument("--skip-plots", action="store_true", help="Skip visualization generation")
    train_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--run-dir", "-r", required=True, help="Run directory")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a training run")
    resume_parser.add_argument("run_dir", help="Run directory to resume")

    # List command
    list_parser = subparsers.add_parser("list", help="List recent runs")
    list_parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command
    commands = {
        "info": cmd_info,
        "preprocess": cmd_preprocess,
        "plan": cmd_plan,
        "train": cmd_train,
        "visualize": cmd_visualize,
        "resume": cmd_resume,
        "list": cmd_list,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
