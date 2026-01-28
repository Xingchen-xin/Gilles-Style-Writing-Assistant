#!/usr/bin/env python3
"""
Plot training metrics from CUDA/Transformers LoRA training runs.

Generates annotated loss curves, learning rate schedules, and gradient norm
plots. Saves results to Parameter_Tuning/ folder in the model directory.

Usage:
    # Single run
    python scripts/plot_training.py models/gswa-lora-Mistral-20260123-012408/

    # Multiple runs for comparison
    python scripts/plot_training.py models/gswa-lora-Mistral-20260122-*/ models/gswa-lora-Mistral-20260123-*/

    # Custom output directory
    python scripts/plot_training.py models/gswa-lora-Mistral-20260123-012408/ --output-dir custom/
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib is required for visualization.")
    print("Install with: pip install matplotlib")
    sys.exit(1)


# Color palette
COLORS = {
    "train_raw": "#90CAF9",      # Light blue
    "train_smooth": "#1565C0",   # Dark blue
    "eval": "#4CAF50",           # Green
    "lr": "#FF9800",             # Orange
    "grad_norm": "#9C27B0",      # Purple
    "annotation_bg": "#FFFDE7",  # Light yellow
}


def find_latest_checkpoint(model_dir: Path) -> Path:
    """Find the checkpoint with the highest step number."""
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return checkpoints[-1]


def load_metrics(model_dir: Path) -> dict:
    """Load training metrics and config from a model directory."""
    model_dir = Path(model_dir)

    # Load log_history: prefer training_metrics.json, fallback to trainer_state.json
    root_metrics = model_dir / "training_metrics.json"
    if root_metrics.exists():
        with open(root_metrics) as f:
            log_history = json.load(f)
    else:
        checkpoint = find_latest_checkpoint(model_dir)
        state_file = checkpoint / "trainer_state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No metrics found in {model_dir}")
        with open(state_file) as f:
            state = json.load(f)
        log_history = state["log_history"]

    # Load training config
    config = {}
    config_file = model_dir / "training_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    # Load metadata for timing info
    metadata = {}
    metadata_file = model_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Separate train steps from eval steps
    train_steps = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_steps = [e for e in log_history if "eval_loss" in e]

    return {
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "config": config,
        "metadata": metadata,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
    }


def moving_average(values, window):
    """Compute moving average with given window size."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def build_annotation(config: dict, metadata: dict) -> str:
    """Build hyperparameter annotation text."""
    lines = []
    if config.get("base_model"):
        model_short = config["base_model"].split("/")[-1]
        if len(model_short) > 30:
            model_short = model_short[:27] + "..."
        lines.append(f"Model: {model_short}")
    if config.get("lora_r"):
        lines.append(f"LoRA: r={config['lora_r']}, alpha={config.get('lora_alpha', '?')}")
    if config.get("learning_rate"):
        lr = config["learning_rate"]
        lines.append(f"LR: {lr:.0e}" if lr < 0.001 else f"LR: {lr}")
    if config.get("batch_size") and config.get("num_gpus"):
        gpu_count = config.get("num_gpus", 1)
        bs = config["batch_size"]
        lines.append(f"Batch: {bs}/GPU x {gpu_count} GPUs")
    elif config.get("batch_size"):
        lines.append(f"Batch: {config['batch_size']}")
    if config.get("epochs"):
        lines.append(f"Epochs: {config['epochs']}")
    if config.get("max_length"):
        lines.append(f"Max length: {config['max_length']}")
    if config.get("quantization"):
        lines.append(f"Quant: {config['quantization']}")
    # Training duration from metadata
    if metadata.get("created_at") and metadata.get("completed_at"):
        try:
            start = datetime.fromisoformat(metadata["created_at"])
            end = datetime.fromisoformat(metadata["completed_at"])
            duration = end - start
            mins = int(duration.total_seconds() / 60)
            lines.append(f"Duration: {mins} min")
        except (ValueError, TypeError):
            pass
    return "\n".join(lines)


def plot_loss_curve(metrics: dict, output_path: Path):
    """Plot training and evaluation loss curves with annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    train_steps = metrics["train_steps"]
    steps = [e["step"] for e in train_steps]
    losses = [e["loss"] for e in train_steps]

    # Raw training loss
    ax.plot(steps, losses, color=COLORS["train_raw"], linewidth=1.0,
            alpha=0.6, label="Train Loss (raw)")

    # Smoothed training loss
    if len(losses) > 5:
        window = max(3, len(losses) // 8)
        smoothed = moving_average(losses, window)
        ax.plot(steps, smoothed, color=COLORS["train_smooth"],
                linewidth=2.5, label=f"Train Loss (smooth, w={window})")

    # Eval loss
    eval_steps = metrics["eval_steps"]
    if eval_steps:
        e_steps = [e["step"] for e in eval_steps]
        e_losses = [e["eval_loss"] for e in eval_steps]
        ax.plot(e_steps, e_losses, color=COLORS["eval"], marker='o',
                markersize=8, linewidth=2, label="Eval Loss")

    # Hyperparameter annotation
    annotation = build_annotation(metrics["config"], metrics["metadata"])
    if annotation:
        ax.text(0.98, 0.98, annotation, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS["annotation_bg"],
                          alpha=0.9, edgecolor='#BDBDBD'))

    # Epoch boundaries
    if metrics["config"].get("epochs"):
        total_steps = steps[-1] if steps else 0
        epochs = metrics["config"]["epochs"]
        for epoch in range(1, epochs):
            epoch_step = int(total_steps * epoch / epochs)
            ax.axvline(x=epoch_step, color='gray', linestyle='--', alpha=0.3)
            ax.text(epoch_step, ax.get_ylim()[1] * 0.98, f"E{epoch}",
                    fontsize=7, color='gray', ha='center')

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Loss Curve", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.75))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_learning_rate(metrics: dict, output_path: Path):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 4))

    train_steps = metrics["train_steps"]
    steps = [e["step"] for e in train_steps]
    lrs = [e["learning_rate"] for e in train_steps]

    ax.plot(steps, lrs, color=COLORS["lr"], linewidth=2)
    ax.fill_between(steps, lrs, alpha=0.1, color=COLORS["lr"])

    # Mark peak LR
    if lrs:
        peak_lr = max(lrs)
        peak_step = steps[lrs.index(peak_lr)]
        ax.annotate(f"Peak: {peak_lr:.2e}",
                    xy=(peak_step, peak_lr),
                    xytext=(peak_step + len(steps) * 0.1, peak_lr * 0.9),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("Learning Rate Schedule (Cosine + Warmup)", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_grad_norm(metrics: dict, output_path: Path):
    """Plot gradient norm over training."""
    fig, ax = plt.subplots(figsize=(10, 4))

    train_steps = metrics["train_steps"]
    steps = [e["step"] for e in train_steps if "grad_norm" in e]
    norms = [e["grad_norm"] for e in train_steps if "grad_norm" in e]

    if not norms:
        plt.close(fig)
        return

    ax.plot(steps, norms, color=COLORS["grad_norm"], linewidth=1.0, alpha=0.6)

    # Smoothed
    if len(norms) > 5:
        window = max(3, len(norms) // 8)
        smoothed = moving_average(norms, window)
        ax.plot(steps, smoothed, color=COLORS["grad_norm"], linewidth=2.5, alpha=0.9)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Gradient Norm", fontsize=11)
    ax.set_title("Gradient Norm", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary(metrics: dict, output_path: Path):
    """Generate a 2x2 summary plot combining all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    train_steps = metrics["train_steps"]
    steps = [e["step"] for e in train_steps]
    losses = [e["loss"] for e in train_steps]
    lrs = [e["learning_rate"] for e in train_steps]
    norms = [e.get("grad_norm", 0) for e in train_steps]

    # 1. Loss curve (top-left)
    ax = axes[0, 0]
    ax.plot(steps, losses, color=COLORS["train_raw"], linewidth=1.0, alpha=0.6)
    if len(losses) > 5:
        smoothed = moving_average(losses, max(3, len(losses) // 8))
        ax.plot(steps, smoothed, color=COLORS["train_smooth"], linewidth=2.5)
    eval_entries = metrics["eval_steps"]
    if eval_entries:
        ax.plot([e["step"] for e in eval_entries],
                [e["eval_loss"] for e in eval_entries],
                color=COLORS["eval"], marker='o', markersize=6, linewidth=1.5)
    ax.set_title("Loss", fontweight='bold')
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.2)

    # 2. Learning rate (top-right)
    ax = axes[0, 1]
    ax.plot(steps, lrs, color=COLORS["lr"], linewidth=2)
    ax.fill_between(steps, lrs, alpha=0.1, color=COLORS["lr"])
    ax.set_title("Learning Rate", fontweight='bold')
    ax.set_xlabel("Step")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax.grid(True, alpha=0.2)

    # 3. Gradient norm (bottom-left)
    ax = axes[1, 0]
    if any(n > 0 for n in norms):
        ax.plot(steps, norms, color=COLORS["grad_norm"], linewidth=1.0, alpha=0.6)
        if len(norms) > 5:
            smoothed = moving_average(norms, max(3, len(norms) // 8))
            ax.plot(steps, smoothed, color=COLORS["grad_norm"], linewidth=2.5)
    ax.set_title("Gradient Norm", fontweight='bold')
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.2)

    # 4. Training info (bottom-right) - text summary
    ax = axes[1, 1]
    ax.axis('off')
    config = metrics["config"]
    info_lines = []
    info_lines.append("Training Configuration")
    info_lines.append("=" * 30)
    if config.get("base_model"):
        info_lines.append(f"Model: {config['base_model'].split('/')[-1]}")
    if config.get("lora_r"):
        info_lines.append(f"LoRA rank: {config['lora_r']}, alpha: {config.get('lora_alpha', '?')}")
    if config.get("learning_rate"):
        info_lines.append(f"Learning rate: {config['learning_rate']}")
    if config.get("batch_size"):
        info_lines.append(f"Batch size: {config['batch_size']}/GPU")
    if config.get("num_gpus"):
        info_lines.append(f"GPUs: {config['num_gpus']} (DDP)")
    if config.get("epochs"):
        info_lines.append(f"Epochs: {config['epochs']}")
    if config.get("max_length"):
        info_lines.append(f"Max length: {config['max_length']}")
    if config.get("quantization"):
        info_lines.append(f"Quantization: {config['quantization']}")
    info_lines.append("")
    info_lines.append("Results")
    info_lines.append("=" * 30)
    if losses:
        info_lines.append(f"Final train loss: {losses[-1]:.4f}")
        info_lines.append(f"Best train loss: {min(losses):.4f}")
    if eval_entries:
        eval_losses = [e["eval_loss"] for e in eval_entries]
        info_lines.append(f"Best eval loss: {min(eval_losses):.4f}")
        info_lines.append(f"Final eval loss: {eval_losses[-1]:.4f}")
    if steps:
        info_lines.append(f"Total steps: {steps[-1]}")
    # Duration and speed
    meta = metrics["metadata"]
    if meta.get("created_at") and meta.get("completed_at"):
        try:
            start = datetime.fromisoformat(meta["created_at"])
            end = datetime.fromisoformat(meta["completed_at"])
            duration = end - start
            mins = int(duration.total_seconds() / 60)
            info_lines.append(f"Duration: {mins} min")
            if steps:
                sps = duration.total_seconds() / steps[-1]
                info_lines.append(f"Speed: {sps:.1f} s/step")
        except (ValueError, TypeError):
            pass

    ax.text(0.1, 0.95, "\n".join(info_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', alpha=0.9))

    fig.suptitle(f"Training Summary: {metrics['model_name']}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def write_summary_report(metrics: dict, output_path: Path):
    """Write a text summary report."""
    config = metrics["config"]
    meta = metrics["metadata"]
    train_steps = metrics["train_steps"]
    eval_steps = metrics["eval_steps"]

    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GSWA Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Directory: {metrics['model_name']}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Hyperparameters\n")
        f.write("-" * 40 + "\n")
        for key in ["base_model", "lora_r", "lora_alpha", "lora_dropout",
                    "learning_rate", "batch_size", "epochs", "max_length",
                    "quantization", "num_gpus", "ddp"]:
            if key in config:
                f.write(f"  {key:<20} {config[key]}\n")
        f.write("\n")

        f.write("-" * 40 + "\n")
        f.write("Training Results\n")
        f.write("-" * 40 + "\n")
        if train_steps:
            losses = [e["loss"] for e in train_steps]
            f.write(f"  {'Total steps:':<20} {train_steps[-1]['step']}\n")
            f.write(f"  {'Initial loss:':<20} {losses[0]:.4f}\n")
            f.write(f"  {'Final loss:':<20} {losses[-1]:.4f}\n")
            f.write(f"  {'Best loss:':<20} {min(losses):.4f}\n")
            f.write(f"  {'Loss reduction:':<20} {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%\n")
        if eval_steps:
            eval_losses = [e["eval_loss"] for e in eval_steps]
            f.write(f"  {'Best eval loss:':<20} {min(eval_losses):.4f}\n")
            f.write(f"  {'Final eval loss:':<20} {eval_losses[-1]:.4f}\n")
            # Overfitting indicator
            if train_steps and eval_steps:
                gap = eval_losses[-1] - losses[-1]
                f.write(f"  {'Train-eval gap:':<20} {gap:.4f}")
                if gap > 0.3:
                    f.write(" (overfitting likely)")
                elif gap > 0.15:
                    f.write(" (mild overfitting)")
                else:
                    f.write(" (good generalization)")
                f.write("\n")
        f.write("\n")

        # Duration and speed
        if meta.get("created_at") and meta.get("completed_at"):
            try:
                start = datetime.fromisoformat(meta["created_at"])
                end = datetime.fromisoformat(meta["completed_at"])
                duration = end - start
                mins = int(duration.total_seconds() / 60)
                f.write(f"  {'Duration:':<20} {mins} min\n")
                f.write(f"  {'Started:':<20} {meta['created_at']}\n")
                f.write(f"  {'Completed:':<20} {meta['completed_at']}\n")
                # Speed info
                total_steps = train_steps[-1]["step"] if train_steps else 0
                if total_steps > 0:
                    secs_per_step = duration.total_seconds() / total_steps
                    f.write(f"  {'Avg speed:':<20} {secs_per_step:.1f} s/step\n")
                    batch = config.get("batch_size", 1)
                    gpus = config.get("num_gpus", 1)
                    grad_accum = config.get("gradient_accumulation_steps", 1)
                    eff_batch = batch * gpus * grad_accum
                    samples_per_sec = eff_batch / secs_per_step
                    f.write(f"  {'Eff. batch:':<20} {batch}×{gpus}GPU×{grad_accum}accum = {eff_batch}\n")
                    f.write(f"  {'Throughput:':<20} {samples_per_sec:.2f} samples/s\n")
            except (ValueError, TypeError):
                pass
        f.write("\n")

        # Gradient norm stats
        if train_steps:
            norms = [e.get("grad_norm", 0) for e in train_steps if e.get("grad_norm")]
            if norms:
                f.write("-" * 40 + "\n")
                f.write("Gradient Norm Stats\n")
                f.write("-" * 40 + "\n")
                f.write(f"  {'Mean:':<20} {sum(norms)/len(norms):.4f}\n")
                f.write(f"  {'Max:':<20} {max(norms):.4f}\n")
                f.write(f"  {'Min:':<20} {min(norms):.4f}\n")
                if max(norms) > 5.0:
                    f.write("  WARNING: High gradient norms detected - training may be unstable\n")
                f.write("\n")

    print(f"  Saved: {output_path}")


def plot_comparison(all_metrics: list, output_dir: Path):
    """Generate comparison plots across multiple training runs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(range(len(all_metrics)))

    for i, metrics in enumerate(all_metrics):
        train_steps = metrics["train_steps"]
        if not train_steps:
            continue

        steps = [e["step"] for e in train_steps]
        losses = [e["loss"] for e in train_steps]
        config = metrics["config"]

        # Build label from key hyperparams
        label_parts = []
        if config.get("lora_r"):
            label_parts.append(f"r={config['lora_r']}")
        if config.get("learning_rate"):
            label_parts.append(f"lr={config['learning_rate']:.0e}")
        if config.get("batch_size"):
            label_parts.append(f"bs={config['batch_size']}")
        if config.get("epochs"):
            label_parts.append(f"ep={config['epochs']}")
        label = ", ".join(label_parts) if label_parts else metrics["model_name"]

        # Smoothed loss
        if len(losses) > 5:
            smoothed = moving_average(losses, max(3, len(losses) // 8))
            ax.plot(steps, smoothed, color=colors[i], linewidth=2.5, label=label)
            ax.plot(steps, losses, color=colors[i], linewidth=0.8, alpha=0.3)
        else:
            ax.plot(steps, losses, color=colors[i], linewidth=2, label=label)

        # Eval loss
        eval_entries = metrics["eval_steps"]
        if eval_entries:
            ax.plot([e["step"] for e in eval_entries],
                    [e["eval_loss"] for e in eval_entries],
                    color=colors[i], marker='o', markersize=6,
                    linestyle='none', alpha=0.7)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Comparison", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)
    plt.tight_layout()

    output_path = output_dir / "comparison_loss.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Write comparison table
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Training Run Comparison\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Run':<40} {'Final Loss':<12} {'Eval Loss':<12} {'Steps':<8} {'LR':<10} {'r':<5} {'Duration':<10} {'s/step':<8}\n")
        f.write("-" * 100 + "\n")
        for metrics in all_metrics:
            name = metrics["model_name"][:38]
            train = metrics["train_steps"]
            evals = metrics["eval_steps"]
            config = metrics["config"]
            meta = metrics["metadata"]
            final_loss = f"{train[-1]['loss']:.4f}" if train else "N/A"
            eval_loss = f"{evals[-1]['eval_loss']:.4f}" if evals else "N/A"
            total_steps = train[-1]["step"] if train else 0
            lr = f"{config.get('learning_rate', 'N/A')}"
            r = str(config.get("lora_r", "N/A"))
            # Duration and speed
            duration_str = "N/A"
            speed_str = "N/A"
            if meta.get("created_at") and meta.get("completed_at"):
                try:
                    start = datetime.fromisoformat(meta["created_at"])
                    end = datetime.fromisoformat(meta["completed_at"])
                    dur_mins = int((end - start).total_seconds() / 60)
                    duration_str = f"{dur_mins}m"
                    if total_steps > 0:
                        speed_str = f"{(end - start).total_seconds() / total_steps:.1f}"
                except (ValueError, TypeError):
                    pass
            f.write(f"{name:<40} {final_loss:<12} {eval_loss:<12} {str(total_steps):<8} {lr:<10} {r:<5} {duration_str:<10} {speed_str:<8}\n")
    print(f"  Saved: {report_path}")


def plot_single_run(metrics: dict, output_dir: Path):
    """Generate all plots for a single training run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots for: {metrics['model_name']}")
    print(f"  Output: {output_dir}/")

    plot_loss_curve(metrics, output_dir / "loss_curve.png")
    plot_learning_rate(metrics, output_dir / "learning_rate.png")
    plot_grad_norm(metrics, output_dir / "grad_norm.png")
    plot_summary(metrics, output_dir / "training_summary.png")
    write_summary_report(metrics, output_dir / "training_report.txt")

    print(f"\n  Done! {len(list(output_dir.iterdir()))} files generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from GSWA LoRA training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/plot_training.py models/gswa-lora-Mistral-20260123-012408/
  python scripts/plot_training.py models/gswa-lora-Mistral-*/ --compare
        """)
    parser.add_argument("model_dirs", nargs="+", type=Path,
                        help="Model directory (or directories for comparison)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (default: {model_dir}/Parameter_Tuning/)")
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison plots instead of individual plots")

    args = parser.parse_args()

    # Validate directories
    valid_dirs = []
    for d in args.model_dirs:
        if not d.exists():
            print(f"WARNING: Directory not found: {d}")
            continue
        valid_dirs.append(d)

    if not valid_dirs:
        print("ERROR: No valid model directories found.")
        sys.exit(1)

    # Load metrics from all directories
    all_metrics = []
    for d in valid_dirs:
        try:
            metrics = load_metrics(d)
            all_metrics.append(metrics)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"WARNING: Cannot load metrics from {d}: {e}")

    if not all_metrics:
        print("ERROR: No metrics could be loaded.")
        sys.exit(1)

    # Generate plots
    if args.compare or len(all_metrics) > 1:
        # Comparison mode
        output_dir = args.output_dir or Path("models/Parameter_Tuning_Comparison")
        plot_comparison(all_metrics, output_dir)
        # Also generate individual plots for each run
        for metrics in all_metrics:
            model_dir = Path(metrics["model_dir"])
            individual_dir = args.output_dir or model_dir / "Parameter_Tuning"
            plot_single_run(metrics, individual_dir)
    else:
        # Single run mode
        metrics = all_metrics[0]
        model_dir = Path(metrics["model_dir"])
        output_dir = args.output_dir or model_dir / "Parameter_Tuning"
        plot_single_run(metrics, output_dir)


if __name__ == "__main__":
    main()
